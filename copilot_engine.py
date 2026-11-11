import os
import re
from datetime import date
from dotenv import load_dotenv

# Internal project modules
from agentic_ai import (
    run as run_agentic_ai,
    to_pipeline_input,
    derive as derive_tool,
    ZONES
)
from ml.prediction import predict_grid_status
from rag.smartgrid_pipeline import load_knowledge_base
from rag.retrieval import retrieve_documents

# Load environment variables
load_dotenv(override=True)

COPILOT_SYSTEM_PROMPT = """You are AI Chatbot, an elite AI assistant equipped with real-time multi-model intelligence and specialized electricity grid analytics.
You provide clear, accurate, deeply reasoned, and beautifully formatted answers to ANY user inquiry (general knowledge, science, coding, electricity grids, weather thermodynamics, math, or daily questions).

Formatting Guidelines:
- Use clear markdown headers, concise bullet points, and bold text for key insights.
- Provide practical, actionable guidance when appropriate.
- Always maintain a helpful, professional, and knowledgeable tone.
"""

GEMINI_CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-flash"
]

def query_gemini(prompt: str, api_key: str, model: str = None, chat_history: list = None) -> tuple[str, str]:
    """Sends prompt securely to Google Gemini API using google.generativeai client with automatic model fallback."""
    import google.generativeai as genai

    genai.configure(api_key=api_key.strip())

    # Build model list to try: prefer specified model, followed by candidates
    models_to_try = []
    if model and model.strip():
        models_to_try.append(model.strip())
    for m in GEMINI_CANDIDATE_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    # Format multi-turn conversation context
    formatted_context = f"{COPILOT_SYSTEM_PROMPT}\n\n"
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            text_val = msg.get("content", "")
            if text_val and not text_val.startswith("❌") and not text_val.startswith("⚠️"):
                formatted_context += f"{role}: {text_val}\n"
    formatted_context += f"User: {prompt}\nAssistant:"

    last_error = None
    for m in models_to_try:
        try:
            # Using standard generativeai model call format
            gemini_model = genai.GenerativeModel(m)
            res = gemini_model.generate_content(formatted_context)
            if res and res.text:
                return res.text.strip(), f"Google Gemini ({m})"
        except Exception as err:
            last_error = err
            continue

    raise Exception(f"Gemini API request failed across models {models_to_try}: {last_error}")


def query_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini", chat_history: list = None) -> tuple[str, str]:
    """Sends prompt securely to OpenAI ChatGPT API with conversational context."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key.strip())
    msgs = [{"role": "system", "content": COPILOT_SYSTEM_PROMPT}]
    if chat_history:
        for m in chat_history[-6:]:
            r = m.get("role")
            c = m.get("content")
            if r in ["user", "assistant"] and c and not c.startswith("❌"):
                msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": prompt})

    m_name = model or "gpt-4o-mini"
    response = client.chat.completions.create(
        model=m_name,
        messages=msgs,
        temperature=0.7
    )
    if response and response.choices and response.choices[0].message:
        return response.choices[0].message.content, f"OpenAI ChatGPT ({m_name})"
    return "No response returned from OpenAI API.", "OpenAI"


def query_groq(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile", chat_history: list = None) -> tuple[str, str]:
    """Sends prompt to Groq API (OpenAI compatible)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key.strip(), base_url="https://api.groq.com/openai/v1")
    msgs = [{"role": "system", "content": COPILOT_SYSTEM_PROMPT}]
    if chat_history:
        for m in chat_history[-6:]:
            r = m.get("role")
            c = m.get("content")
            if r in ["user", "assistant"] and c and not c.startswith("❌"):
                msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": prompt})

    m_name = model or "llama-3.3-70b-versatile"
    response = client.chat.completions.create(
        model=m_name,
        messages=msgs,
        temperature=0.7
    )
    if response and response.choices and response.choices[0].message:
        return response.choices[0].message.content, f"Groq ({m_name})"
    return "No response returned from Groq API.", "Groq"


def test_ai_connection(provider: str = None, api_key: str = None, model: str = None) -> tuple[bool, str]:
    """Tests the connection to the specified AI model API and returns (status, message)."""
    load_dotenv(override=True)
    p = (provider or os.getenv("AI_PROVIDER", "gemini")).strip().lower()
    k = (api_key or os.getenv("GEMINI_API_KEY" if p == "gemini" else "OPENAI_API_KEY")).strip()

    if not k:
        return False, f"API key is empty for provider '{p}'. Please set your key."

    try:
        if "gemini" in p:
            m = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            ans, used = query_gemini("Reply with only 'API Connected'", k, model=m)
            return True, f"✅ Successfully connected to {used}!"
        elif "openai" in p:
            m = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            ans, used = query_openai("Reply with only 'API Connected'", k, model=m)
            return True, f"✅ Successfully connected to {used}!"
        elif "groq" in p:
            m = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            ans, used = query_groq("Reply with only 'API Connected'", k, model=m)
            return True, f"✅ Successfully connected to {used}!"
        else:
            return False, f"Unknown AI provider: {p}"
    except Exception as e:
        return False, f"Connection failed: {e}"


def ask_copilot(query: str, chat_history: list = None) -> dict:
    """
    Main entrypoint for the AI Chatbot.
    Dynamically routes prompt to Google Gemini API or OpenAI API with real-time error handling.
    """
    load_dotenv(override=True)
    clean_q = query.strip()
    lower_q = clean_q.lower()

    # Check keys from environment and Streamlit session_state
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    active_provider = os.getenv("AI_PROVIDER", "").strip().lower()

    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            gemini_key = st.session_state.get("gemini_api_key", gemini_key) or gemini_key
            openai_key = st.session_state.get("openai_api_key", openai_key) or openai_key
            groq_key = st.session_state.get("groq_api_key", groq_key) or groq_key
            active_provider = st.session_state.get("ai_provider", active_provider) or active_provider
    except Exception:
        pass

    # 1. Detect if the message is a grid condition description for forecasting
    has_condition_words = any(k in lower_q for k in [
        "high", "low", "avg", "temp", "celsius", "humidity", "mw", "load", "wind", "rain", "dallas", "houston", "austin", "midland", "weather"
    ])
    has_numbers = any(c.isdigit() for c in clean_q)
    is_forecast_request = has_condition_words and has_numbers

    if is_forecast_request:
        # Run Agentic Tool Calling
        extracted = run_agentic_ai(clean_q)
        pipeline_input = to_pipeline_input(extracted, clean_q)
        prediction = predict_grid_status(pipeline_input)

        # Retrieve RAG context
        context_str = ""
        sources = []
        try:
            chunks, embeddings = load_knowledge_base()
            retrieved = retrieve_documents(prediction["rag_query"], chunks, embeddings, top_k=3)
            context_str = "\n".join([f"[{r['metadata']['source']} P.{r['metadata']['page']}]: {r['text']}" for r in retrieved])
            sources = [
                {"document": r["metadata"]["source"], "page": r["metadata"]["page"], "similarity": round(float(r["score"]), 4)}
                for r in retrieved
            ]
        except Exception as e:
            print(f"[AI Chatbot] RAG retrieval notice: {e}")

        # Call Gemini or OpenAI for natural advisory
        advisory_prompt = f"Explain this electricity forecast for ERCOT {pipeline_input['ERCOT_zone']} zone: Predicted {prediction['predicted_usage']} MW, Threshold {prediction['usage_threshold']} MW, Status: {prediction['risk_status']}, Risk: {prediction['risk_probability']}%. Context: {context_str}. Provide concise consumer recommendations."

        advisory_text = ""
        engine_name = "AI Chatbot"
        if gemini_key:
            try:
                advisory_text, engine_name = query_gemini(advisory_prompt, gemini_key, chat_history=chat_history)
            except Exception as e:
                advisory_text = f"Forecast generated. (Gemini advisory notice: {e})"
        elif openai_key:
            try:
                advisory_text, engine_name = query_openai(advisory_prompt, openai_key, chat_history=chat_history)
            except Exception as e:
                advisory_text = f"Forecast generated. (OpenAI advisory notice: {e})"
        else:
            advisory_text = "Operating reserves and power generation are evaluated based on forecasted weather loads."

        risk_emoji = "⚠️ HIGH USAGE RISK" if "high" in prediction["risk_status"].lower() else "✅ NORMAL USAGE"
        formatted_response = f"""### ⚡ SmartGrid Demand Forecast: **{risk_emoji}**

- **Projected Regional Demand:** **{prediction['predicted_usage']:,.2f} MW**
- **High-Usage Threshold:** **{prediction['usage_threshold']:,.2f} MW**
- **Stress Probability:** **{prediction['risk_probability']:.2f}%**
- **Classification Status:** `{prediction['risk_status']}`

---
{advisory_text}
"""
        return {
            "text": formatted_response,
            "engine": engine_name,
            "is_forecast": True,
            "extracted": extracted,
            "pipeline_input": pipeline_input,
            "prediction": prediction,
            "sources": sources
        }

    # 2. General Questions & Conversations -> Query live AI API
    # Route to Gemini if key available or provider selected
    if gemini_key and (active_provider in ["gemini", "google", "auto", ""] or not openai_key):
        try:
            model_target = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            ans_text, eng = query_gemini(clean_q, gemini_key, model=model_target, chat_history=chat_history)
            return {
                "text": ans_text,
                "engine": eng,
                "sources": []
            }
        except Exception as e:
            return {
                "text": f"❌ **Error connecting to Google Gemini API:**\n\n`{str(e)}`\n\n*Tip: Check that your GEMINI_API_KEY is active and has access to Gemini models.*",
                "engine": "Gemini API Error",
                "sources": []
            }

    # Route to OpenAI if key available
    if openai_key and (active_provider in ["openai", "chatgpt", "auto", ""]):
        try:
            model_target = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            ans_text, eng = query_openai(clean_q, openai_key, model=model_target, chat_history=chat_history)
            return {
                "text": ans_text,
                "engine": eng,
                "sources": []
            }
        except Exception as e:
            return {
                "text": f"❌ **Error connecting to OpenAI API:**\n\n`{str(e)}`",
                "engine": "OpenAI API Error",
                "sources": []
            }

    # Route to Groq if key available
    if groq_key:
        try:
            model_target = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            ans_text, eng = query_groq(clean_q, groq_key, model=model_target, chat_history=chat_history)
            return {
                "text": ans_text,
                "engine": eng,
                "sources": []
            }
        except Exception as e:
            return {
                "text": f"❌ **Error connecting to Groq API:**\n\n`{str(e)}`",
                "engine": "Groq API Error",
                "sources": []
            }

    # 3. No API Key configured at all
    return {
        "text": """⚠️ **API Key Missing:**
Please enter your **Google Gemini API Key** or **OpenAI API Key** in the sidebar or in the `.env` file to enable dynamic AI responses!

```env
GEMINI_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
```""",
        "engine": "Setup Required",
        "sources": []
    }
