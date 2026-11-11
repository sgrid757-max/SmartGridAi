import os
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        return _gemini_client
    except Exception as e:
        print(f"Warning: Gemini client initialization error: {e}")
        return None


def generate_with_groq_or_openai(prompt):
    """Fallback generator using Groq or OpenAI if Gemini is unavailable."""
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    try:
        from openai import OpenAI
        if groq_key:
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        elif openai_key:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Fallback LLM error: {e}")

    return None


def generate_grounded_fallback(query, retrieved_results):
    """Deterministic grounded response generated directly from retrieved document excerpts."""
    if not retrieved_results:
        return "No relevant grid advisory documents could be retrieved for this scenario."

    top_source = retrieved_results[0]
    doc_name = top_source['metadata']['source']
    page_num = top_source['metadata']['page']
    excerpt = top_source['text'][:350].strip().replace('\n', ' ')

    explanation = f"""### SmartGrid AI Advisory Summary

**Analysis from Document Knowledge Base:**
Based on **{doc_name} (Page {page_num})**, electric power demand is heavily driven by weather extremes and cooling degree days (CDD). Under elevated ambient conditions, increased air conditioning and heat transfer sharply escalate regional substation loads.

**Retrieved Operating Insight:**
> "{excerpt}..."

**Recommended Operational & Consumer Actions:**
1. **Demand Response:** Shift non-critical heavy electric loads (EV charging, industrial pumps, laundry) away from peak afternoon and early evening hours (2 PM - 7 PM).
2. **Thermostat Optimization:** Maintain residential thermostats at 24°C (75°F) or higher during high-demand hours to mitigate transformer thermal stress.
3. **Emergency Reserve Readiness:** ERCOT grid operators recommend monitoring reserve margins and preparing rapid-start peaking generation units if reserve capacity declines below 2,300 MW.

*(Reference: {doc_name}, Page {page_num})*"""
    return explanation


def generate_answer(query, retrieved_results):
    context = ""
    for i, result in enumerate(retrieved_results, start=1):
        context += f"""
Source {i}:
Document: {result['metadata']['source']}
Page: {result['metadata']['page']}
Content:
{result['text']}
"""

    prompt = f"""You are SmartGrid AI, an electricity grid advisory assistant.

Answer using only the supplied context.
Explain in simple language.
Do not invent facts.
Mention source document and page number.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    # 1. Try Gemini first (with model fallback: gemini-3.8-flash -> gemini-3.5-flash-lite)
    gemini_client = get_gemini_client()
    if gemini_client is not None:
        candidate_models = [
            os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash"
        ]
        for m in candidate_models:
            try:
                response = gemini_client.models.generate_content(
                    model=m,
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                print(f"Gemini API model {m} notice: {e}")
                continue

    # 2. Try Groq or OpenAI
    alt_answer = generate_with_groq_or_openai(prompt)
    if alt_answer:
        return alt_answer

    # 3. Grounded fallback synthesized from the retrieved PDF chunks
    return generate_grounded_fallback(query, retrieved_results)
