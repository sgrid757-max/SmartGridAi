"""
copilot_engine.py
-----------------
Backend for the floating AI Chatbot.

`ask_copilot(query, history)` sends the conversation to a real LLM
(Google Gemini first, OpenAI as fallback) and returns:

    {"text": "<answer or friendly error message>", "sources": [...], "ok": bool}

Design notes
- API keys are read server-side from environment variables (the Streamlit
  process). They are never sent to the browser.
- Uses only the Python standard library (urllib), so nothing extra to install.
- Never raises: every failure becomes a readable message, so the UI can't crash.
"""

import json
import os
import urllib.error
import urllib.request

# ---------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
REQUEST_TIMEOUT_S = int(os.getenv("COPILOT_TIMEOUT_S", "30"))
MAX_HISTORY_MESSAGES = 12      # keep prompts small and cheap
MAX_INPUT_CHARS = 4000         # reject absurdly long prompts

SYSTEM_PROMPT = (
    "You are the SmartGrid AI Chatbot inside an energy-intelligence app. "
    "You can answer general questions, and you are especially good at "
    "electricity demand, the ERCOT grid, weather impact on load, and "
    "consumer energy advice. Be clear and concise. Use Markdown when helpful. "
    "If you are unsure, say so instead of guessing."
)


class CopilotError(Exception):
    """Raised internally; converted to a friendly message in ask_copilot."""


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _clean_history(history):
    """Keep only user/assistant turns with text, trim to the last N, and make
    sure the conversation starts with a user turn (both APIs require it)."""
    turns = [
        {"role": m["role"], "content": str(m.get("content", "")).strip()}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ][-MAX_HISTORY_MESSAGES:]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


def _post_json(url, payload, headers):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        if e.code in (401, 403):
            raise CopilotError("The API key was rejected. Please check it in the sidebar settings.")
        if e.code == 429:
            raise CopilotError("Rate limit or quota reached. Please wait a moment and try again.")
        if e.code == 400:
            raise CopilotError(f"The AI service rejected the request. {body[:200]}")
        raise CopilotError(f"The AI service returned an error (HTTP {e.code}).")
    except urllib.error.URLError as e:
        raise CopilotError(f"Could not reach the AI service. Check your internet connection. ({e.reason})")
    except TimeoutError:
        raise CopilotError("The AI service took too long to respond. Please try again.")
    except json.JSONDecodeError:
        raise CopilotError("The AI service sent back an unreadable response.")


# ---------------------------------------------------------------
# Providers
# ---------------------------------------------------------------
def _call_gemini(turns, system_prompt, api_key):
    contents = [
        {"role": "user" if t["role"] == "user" else "model", "parts": [{"text": t["content"]}]}
        for t in turns
    ]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    data = _post_json(
        url,
        {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1024},
        },
        # Key goes in a header, not the URL, so it doesn't end up in logs.
        {"x-goog-api-key": api_key},
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        if reason:
            raise CopilotError(f"The request was blocked by the safety filter ({reason}). Try rephrasing.")
        raise CopilotError("The AI returned an empty response. Please try again.")
    if not text:
        raise CopilotError("The AI returned an empty response. Please try again.")
    return text


def _call_openai(turns, system_prompt, api_key):
    messages = [{"role": "system", "content": system_prompt}] + turns
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.6, "max_tokens": 1024},
        {"Authorization": f"Bearer {api_key}"},
    )
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise CopilotError("The AI returned an empty response. Please try again.")
    if not text:
        raise CopilotError("The AI returned an empty response. Please try again.")
    return text


# ---------------------------------------------------------------
# Optional: ground answers in your ERCOT PDF knowledge base (RAG)
# ---------------------------------------------------------------
def _rag_context(query):
    """Returns (context_text, sources). Silent no-op if RAG is unavailable."""
    try:
        from rag.smartgrid_pipeline import load_knowledge_base
        from rag.retrieval import retrieve_documents

        chunks, embeddings = load_knowledge_base()
        retrieved = retrieve_documents(query, chunks, embeddings, top_k=3)
        sources = [
            {"document": r["metadata"]["source"], "page": r["metadata"]["page"]}
            for r in retrieved
        ]
        context = "\n\n".join(str(r.get("text") or r.get("content") or "") for r in retrieved).strip()
        return context, sources
    except Exception:
        return "", []


# ---------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------
def ask_copilot(query, history=None, use_rag=False):
    """
    query:   the latest user message (string)
    history: list of {"role", "content"} dicts (may already include `query`)
    use_rag: set True to ground answers in the ERCOT knowledge base
    """
    query = (query or "").strip()
    if not query:
        return {"text": "Please type a question first.", "sources": [], "ok": False}
    if len(query) > MAX_INPUT_CHARS:
        return {
            "text": f"That message is too long ({len(query):,} characters). Please keep it under {MAX_INPUT_CHARS:,}.",
            "sources": [],
            "ok": False,
        }

    turns = _clean_history(history)
    # The app appends the user message before calling us; add it only if missing.
    if not turns or turns[-1]["role"] != "user" or turns[-1]["content"] != query:
        turns.append({"role": "user", "content": query})

    system_prompt, sources = SYSTEM_PROMPT, []
    if use_rag:
        context, sources = _rag_context(query)
        if context:
            system_prompt += "\n\nUse this reference material when relevant:\n" + context

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not gemini_key and not openai_key:
        return {
            "text": (
                "No AI API key is configured yet. Open **Agent & API Configuration** in the "
                "sidebar and enter a Gemini or OpenAI key, then ask again."
            ),
            "sources": [],
            "ok": False,
        }

    errors = []
    for name, key, fn in (("Gemini", gemini_key, _call_gemini), ("OpenAI", openai_key, _call_openai)):
        if not key:
            continue
        try:
            return {"text": fn(turns, system_prompt, key), "sources": sources, "ok": True}
        except CopilotError as e:
            errors.append(f"{name}: {e}")
        except Exception as e:  # last-resort safety net
            errors.append(f"{name}: unexpected error ({type(e).__name__}).")

    return {
        "text": "⚠️ Sorry, I couldn't get an answer right now.\n\n" + "\n".join(f"- {e}" for e in errors),
        "sources": [],
        "ok": False,
    }
