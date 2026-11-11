import os
import json
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
from rag.llm import generate_grounded_fallback

# Load environment
load_dotenv()

COPILOT_SYSTEM_PROMPT = """You are AI Chatbot, an elite multi-model assistant combining general knowledge like ChatGPT and Gemini with specialized electricity grid intelligence.
You provide clear, accurate, deeply reasoned, and beautifully formatted answers to ANY user inquiry (general questions, science, coding, electricity grids, weather thermodynamics, math, or daily topics).

Formatting Guidelines:
- Use clear markdown headers, concise bullet points, and bold text for key insights.
- Provide practical, actionable guidance when appropriate.
- Always maintain a helpful, professional, and knowledgeable tone.
"""

def query_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini", chat_history: list = None) -> str:
    """Sends prompt securely to OpenAI ChatGPT API with conversational context."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    msgs = [{"role": "system", "content": COPILOT_SYSTEM_PROMPT}]
    if chat_history:
        for m in chat_history[-6:]:
            r = m.get("role")
            c = m.get("content")
            if r in ["user", "assistant"] and c:
                msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.7
    )
    if response and response.choices and response.choices[0].message:
        return response.choices[0].message.content
    return "No response returned from OpenAI API."


def query_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash", chat_history: list = None) -> str:
    """Sends prompt securely to Google Gemini API with conversational context."""
    # 1. Try modern google.genai SDK
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        full_text = f"{COPILOT_SYSTEM_PROMPT}\n\n"
        if chat_history:
            for m in chat_history[-6:]:
                role_label = "User" if m.get("role") == "user" else "AI Chatbot"
                full_text += f"{role_label}: {m.get('content')}\n"
        full_text += f"User: {prompt}\nAI Chatbot:"

        response = client.models.generate_content(
            model=model,
            contents=full_text
        )
        if response and response.text:
            return response.text
    except Exception as sdk_err:
        # 2. Direct REST fallback if SDK has version discrepancies
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"parts": [{"text": f"{COPILOT_SYSTEM_PROMPT}\n\nUser Question: {prompt}"}]}
            ]
        }
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


def query_groq(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile", chat_history: list = None) -> str:
    """Sends prompt to Groq API (OpenAI compatible)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    msgs = [{"role": "system", "content": COPILOT_SYSTEM_PROMPT}]
    if chat_history:
        for m in chat_history[-6:]:
            r = m.get("role")
            c = m.get("content")
            if r in ["user", "assistant"] and c:
                msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.7
    )
    if response and response.choices and response.choices[0].message:
        return response.choices[0].message.content
    return "No response returned from Groq API."


def test_ai_connection(provider: str, api_key: str, model: str = "") -> tuple[bool, str]:
    """Tests the connection to the specified AI model API and returns (status, message)."""
    if not api_key or not api_key.strip():
        return False, "API key is empty. Please enter a valid API key."

    clean_key = api_key.strip()
    try:
        if provider == "openai":
            m = model or "gpt-4o-mini"
            ans = query_openai("Respond with only 'Connection successful'", clean_key, model=m)
            return True, f"Connected to OpenAI ChatGPT ({m}) successfully!"
        elif provider == "gemini":
            m = model or "gemini-2.5-flash"
            ans = query_gemini("Respond with only 'Connection successful'", clean_key, model=m)
            return True, f"Connected to Google Gemini ({m}) successfully!"
        elif provider == "groq":
            m = model or "llama-3.3-70b-versatile"
            ans = query_groq("Respond with only 'Connection successful'", clean_key, model=m)
            return True, f"Connected to Groq ({m}) successfully!"
        else:
            return False, f"Unknown AI provider: {provider}"
    except Exception as e:
        return False, f"Connection test failed: {e}"


def get_llm_response(prompt: str, context: str = "", chat_history: list = None) -> tuple[str, str]:
    """
    Dynamically routes prompt to OpenAI, Google Gemini, or Groq API.
    Provides comprehensive error handling and loading feedback.
    """
    full_prompt = prompt
    if context:
        full_prompt = f"RELEVANT TECHNICAL KNOWLEDGE CONTEXT:\n{context}\n\nUSER QUESTION:\n{prompt}"

    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    active_provider = os.getenv("AI_PROVIDER", "auto").strip().lower()

    # Also check Streamlit session_state if available
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            openai_key = st.session_state.get("openai_api_key", openai_key) or openai_key
            gemini_key = st.session_state.get("gemini_api_key", gemini_key) or gemini_key
            groq_key = st.session_state.get("groq_api_key", groq_key) or groq_key
            active_provider = st.session_state.get("ai_provider", active_provider) or active_provider
    except Exception:
        pass

    # 1. If OpenAI ChatGPT is explicitly chosen or available
    if (active_provider in ["openai", "chatgpt"] and openai_key) or (active_provider == "auto" and openai_key):
        try:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            try:
                import streamlit as st
                model = st.session_state.get("openai_model", model)
            except Exception:
                pass
            res = query_openai(full_prompt, openai_key, model=model, chat_history=chat_history)
            return res, f"OpenAI ChatGPT ({model})"
        except Exception as e:
            return f"⚠️ **OpenAI API Error**: {e}\n\nPlease check that your `OPENAI_API_KEY` is valid and has active quota.", "OpenAI Error"

    # 2. If Google Gemini is explicitly chosen or available
    if (active_provider in ["gemini", "google"] and gemini_key) or (active_provider == "auto" and gemini_key):
        try:
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            try:
                import streamlit as st
                model = st.session_state.get("gemini_model", model)
            except Exception:
                pass
            res = query_gemini(full_prompt, gemini_key, model=model, chat_history=chat_history)
            return res, f"Google Gemini ({model})"
        except Exception as e:
            return f"⚠️ **Google Gemini API Error**: {e}\n\nPlease verify your `GEMINI_API_KEY` in settings or `.env`.", "Gemini Error"

    # 3. If Groq is explicitly chosen or available
    if (active_provider in ["groq"] and groq_key) or (active_provider == "auto" and groq_key):
        try:
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            try:
                import streamlit as st
                model = st.session_state.get("groq_model", model)
            except Exception:
                pass
            res = query_groq(full_prompt, groq_key, model=model, chat_history=chat_history)
            return res, f"Groq ({model})"
        except Exception as e:
            return f"⚠️ **Groq API Error**: {e}\n\nPlease check your `GROQ_API_KEY`.", "Groq Error"

    # 4. If an API key is specified but provider failed:
    if openai_key:
        try:
            res = query_openai(full_prompt, openai_key, model="gpt-4o-mini", chat_history=chat_history)
            return res, "OpenAI ChatGPT (gpt-4o-mini)"
        except Exception as e:
            return f"⚠️ **OpenAI API Error**: {e}", "OpenAI Error"

    if gemini_key:
        try:
            res = query_gemini(full_prompt, gemini_key, model="gemini-2.5-flash", chat_history=chat_history)
            return res, "Google Gemini (gemini-2.5-flash)"
        except Exception as e:
            return f"⚠️ **Gemini API Error**: {e}", "Gemini Error"

    # 5. No API Key configured -> Provide clear setup guide so user knows how to connect live
    help_message = """### 🔑 AI Model API Key Required

To get **real-time dynamic AI responses** from **ChatGPT (OpenAI)** or **Google Gemini**, please provide your API key:

#### ⚡ Quick Setup:
1. **In the Web App:** Open the **⚙️ AI Settings** section (in the floating chat or left sidebar) and paste your API key.
2. **In `.env`:** Add your key to the `.env` file in the project folder:
   ```env
   # OpenAI ChatGPT (gpt-4o-mini / gpt-4o):
   OPENAI_API_KEY=sk-your-openai-api-key-here

   # OR Google Gemini (gemini-2.5-flash):
   GEMINI_API_KEY=AIzaSy-your-gemini-key-here
   ```
3. Once entered, the AI Chatbot will immediately query the model live and generate real-time answers for any question!"""

    return help_message, "AI API Setup Required"


def synthesize_knowledge_fallback(query: str, context: str = "") -> str:
    """
    Comprehensive multi-topic synthesizer ensuring the assistant always answers
    intelligently across grid engineering, computer science, thermodynamics, math,
    and general knowledge in the style of ChatGPT and Gemini.
    """
    lower_q = query.lower()

    if context:
        # Context available from RAG
        return f"""### 🤖 AI Chatbot Technical Advisory

{context[:650]}...

#### 📋 Operational Recommendations:
1. **Load Shifting:** Reduce non-critical power consumption during peak load periods (2:00 PM – 7:00 PM).
2. **Thermal Management:** Maintain thermostats at 24°C (75°F) or above during summer peaks; 20°C (68°F) during winter freezes.
3. **Emergency Reserves:** ERCOT protocols require rapid-response generation units to spin up if physical operating reserves drop below 2,300 MW."""

    # Category 1: ERCOT EEA & Emergency Alerts
    if any(k in lower_q for k in ["eea", "emergency alert", "reserve margin", "operating reserve"]):
        return """### 🚨 ERCOT Energy Emergency Alerts (EEA) Protocols

ERCOT declares an Energy Emergency Alert when operating reserves decline toward critical thresholds:

| Level | Reserve Threshold | Mandatory Protocol & Grid Action |
|---|---|---|
| **EEA Level 1** | Operating Reserves $< 2,300$ MW | All available generation units online; DC tie lines import power from neighboring grids. |
| **EEA Level 2** | Operating Reserves $< 1,750$ MW | Large industrial demand-response contracts interrupted; public appeals for emergency conservation. |
| **EEA Level 3** | Operating Reserves $< 1,430$ MW | Controlled, rotating outages (firm load shed) ordered to prevent full statewide grid collapse. |

#### 🛡️ Action Plan for Energy Consumers:
- **Immediate Appliance Curtailment:** Postpone dishwashers, clothes dryers, and pool pumps until after 8:00 PM.
- **Thermostat Adjustments:** Set cooling to 78°F (25.5°C) or heating to 68°F (20°C).
- **EV Charging Pause:** Defer electric vehicle charging to overnight off-peak hours (11:00 PM – 6:00 AM)."""

    # Category 2: Cooling Degree Days & Heat Thermodynamics
    elif any(k in lower_q for k in ["cdd", "cooling degree", "heat index", "wet bulb"]):
        return """### 🌡️ Cooling Degree Days (CDD) & Thermal Grid Dynamics

#### 📐 Mathematical Definition:
Cooling Degree Days measure how much the mean daily temperature exceeds the standard thermal comfort baseline ($18.3^\\circ\\text{C}$ or $65^\\circ\\text{F}$):
$$\\text{CDD} = \\max\\left(0, \\text{Temperature}_{\\text{mean}} - 18.3\\right)$$

#### ⚡ Critical Impacts on Electricity Demand:
1. **Non-Linear HVAC Compressor Loading:** Above $\\text{CDD} \\ge 10.0$, air conditioning systems run continuously rather than cycling, causing steep aggregate demand spikes.
2. **Distribution Transformer Overheating:** High ambient overnight temperatures prevent oil-filled substation transformers from dissipating heat, increasing failure rates.
3. **Transmission Line Sag:** High ambient heat causes high-voltage conductors to expand and physically sag, decreasing maximum transmission transfer capacity."""

    # Category 3: Winter Storms & Freeze Reliability
    elif any(k in lower_q for k in ["winter", "freeze", "uri", "cold weather", "ice storm"]):
        return """### ❄️ Winter Storm Dynamics & ERCOT Reliability

During extreme Arctic freeze events (such as Winter Storm Uri in February 2021), Texas electric infrastructure faces compounded stresses:

#### 🔍 Compounding Factors:
1. **Electric Heating Surge:** Over 60% of Texas homes rely on electric resistance heating, which consumes up to 3× more power than heat pumps below freezing.
2. **Fuel Delivery Freeze-Offs:** Uninsulated natural gas wellheads, gathering lines, and compressor stations freeze up, cutting generation fuel supply when most needed.
3. **Mechanical Failures:** Un-winterized cooling water intake lines, instrumentation lines, and wind turbine blade icing take gigawatts of generation offline.

#### 🛠️ Modern Post-Uri Reforms:
- Mandatory weatherization inspections by ERCOT and PUCT.
- Firm fuel supply service (FFSS) requiring dual-fuel capability for key gas generators.
- Expanded Fast Frequency Response (FFR) and battery energy storage integration."""

    # Category 4: Solar, Wind, and Renewable Integration
    elif any(k in lower_q for k in ["solar", "wind", "renewable", "battery", "bess", "storage", "duck curve"]):
        return """### ☀️ Texas Renewable Generation & Battery Storage (BESS)

Texas leads the United States in utility-scale renewable generation, with over **38,000 MW of wind** and **18,000+ MW of solar** capacity.

#### 📉 The ERCOT Duck Curve & Net Load Ramp:
- **Noon Solar Peak:** High solar output drives down net demand during middle daylight hours.
- **Sunset Ramp (6:00 PM – 8:00 PM):** As solar generation rapidly drops to 0 MW while residential air conditioning demand peaks, conventional thermal generation and batteries must ramp up by thousands of megawatts in under 90 minutes.

#### 🔋 Battery Energy Storage Systems (BESS):
- Texas has deployed over **5,000 MW / 10,000 MWh** of 2-hour and 4-hour lithium-ion batteries.
- Batteries provide sub-second Fast Frequency Response (FFR) to arrest sudden frequency drops and discharge during peak net load ramps."""

    # Category 5: Python Code, Programming, or Feature Engineering
    elif any(k in lower_q for k in ["code", "python", "script", "function", "program", "pandas", "algorithm"]):
        return """### 💻 Python Implementation: Agentic Feature Derivation & Analysis

Here is a clean, production-ready Python implementation for calculating electricity grid meteorological features:

```python
from datetime import date
from typing import Dict, Any

ZONES = ["COAST", "EAST", "NORTH", "SOUTH", "WEST"]

def derive_grid_features(
    temperature_mean: float,
    temperature_max: float,
    temperature_min: float,
    date_iso: str,
    ercot_zone: str,
    zone_usage_mw: float = 0.0
) -> Dict[str, Any]:
    \"\"\"
    Computes derived meteorological and calendar features for ERCOT load forecasting.
    Temps in Celsius.
    \"\"\"
    target_date = date.fromisoformat(date_iso)
    cdd = round(max(0.0, temperature_mean - 18.3), 2)
    temp_range = round(temperature_max - temperature_min, 2)
    is_weekend = int(target_date.weekday() >= 5)
    zone_upper = ercot_zone.strip().upper()
    zone_encoded = ZONES.index(zone_upper) if zone_upper in ZONES else 2  # default NORTH

    return {
        "zone_usage": zone_usage_mw,
        "temperature_mean": temperature_mean,
        "temperature_max": temperature_max,
        "temperature_min": temperature_min,
        "CDD": cdd,
        "temp_range": temp_range,
        "weekend": is_weekend,
        "ERCOT_zone_encoded": zone_encoded,
        "day_of_week": target_date.weekday(),
        "month": target_date.month
    }

# Example execution:
features = derive_grid_features(
    temperature_mean=29.0,
    temperature_max=35.0,
    temperature_min=24.0,
    date_iso="2026-11-14",
    ercot_zone="NORTH",
    zone_usage_mw=52000.0
)
print("Computed Features:", features)
```

This snippet guarantees strict schema compliance with machine learning inference pipelines."""

    # Category 6: Artificial Intelligence, Machine Learning, Transformers
    elif any(k in lower_q for k in ["machine learning", "ml", "ai", "xgboost", "gradient boosting", "transformer", "chatgpt", "gemini", "rag", "neural"]):
        return r"""### 🤖 AI Architecture: Machine Learning & Agentic LLMs

#### 1. Machine Learning Demand Forecasting (XGBoost & Gradient Boosting):
- **Tabular Superiority:** Gradient boosted decision trees (XGBoost/LightGBM) outperform deep networks on structured time-series and weather data.
- **Key Features:** Mean temperature ($\sim 65\%$ importance), Cooling Degree Days ($CDD$), diurnal temperature swing ($temp\_range$), and weekend calendar indicators.
- **Classification:** Logistic sigmoid mapping predicts the probability of exceeding the $90^{\text{th}}$ percentile usage threshold.

#### 2. Agentic Tool Calling vs Simple Text Generation:
- **Deterministic Derivation:** The LLM does not perform fuzzy arithmetic. Instead, it extracts structured parameters from natural language and invokes verified tools (such as `derive`).
- **Grounding (RAG):** Domain documents are converted into dense vector embeddings. Cosine similarity retrieves official technical protocols, eliminating hallucinations."""

    # Category 7: Electrical Engineering & Grid Physics
    elif any(k in lower_q for k in ["voltage", "frequency", "hertz", "hz", "ohm", "current", "ac", "dc", "substation", "transformer"]):
        return """### ⚡ Electrical Grid Physics & Frequency Regulation

#### ⚙️ The 60.0 Hz Frequency Rule:
In the North American Eastern, Western, and Texas (ERCOT) interconnections, grid frequency must remain tightly synchronized at **60.00 Hz**:
- **Frequency $> 60.0\\text{ Hz}$:** Total generation exceeds instantaneous load; generators must throttle back.
- **Frequency $< 60.0\\text{ Hz}$:** Load exceeds generation; governor systems open turbines or batteries inject power.
- **Critical Under-Frequency ($< 59.3\\text{ Hz}$):** Under-Frequency Load Shedding (UFLS) relays automatically trip distribution feeders within milliseconds to prevent turbine blade resonance damage.

#### 📐 Core AC Electrical Formulae:
- **Single-Phase Real Power:** $P = V \\cdot I \\cdot \\cos(\\theta)$
- **Three-Phase Real Power:** $P = \\sqrt{3} \\cdot V_{\\text{LL}} \\cdot I_{\\text{L}} \\cdot \\cos(\\theta)$
- **Impedance:** $Z = R + j(X_L - X_C)$"""

    # Category 8: General Knowledge / ChatGPT & Gemini Style Reasoning
    else:
        return f"""### 🤖 AI Chatbot

Thank you for your question: **"{query}"**

#### 🔬 Executive Summary:
This topic touches on core principles of system dynamics, predictive optimization, and data-driven reasoning:

1. **Fundamental Principle:** Whether modeling an electrical power grid, software systems, or scientific phenomena, equilibrium is dictated by conservation laws and real-time feedback loops.
2. **Analytical Methodology:** High-dimensional problems are best addressed by decoupling variables, applying validated domain equations, and verifying inputs against empirical distributions.
3. **Applied Synthesis:** Integrating predictive machine learning with conversational AI enables continuous monitoring, instant situational awareness, and automated decision support.

#### 🚀 How Can I Assist You Further?
- **Weather Scenario Forecasting:** Type any condition like `"Dallas Saturday high 35C, avg 29C, wind 4 m/s, load 52000 MW"`.
- **ERCOT Regulations:** Inquire about ancillary services, reserve margins, or EEA emergency alert levels.
- **Code & Calculations:** Ask for Python scripts, feature derivation equations, or system architecture explanations!"""



def ask_copilot(query: str, chat_history: list = None) -> dict:
    """
    Main entrypoint for the Floating SmartGrid AI Copilot.
    Handles:
    - Weather / condition descriptions -> Agentic tool calling + ML Forecast + RAG
    - Domain questions -> RAG retrieval + LLM synthesis
    - General questions -> Multi-model ChatGPT/Gemini LLM response
    """
    clean_q = query.strip()
    lower_q = clean_q.lower()

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
            print(f"[Copilot] RAG retrieval notice: {e}")

        # Generate advisory via LLM or Grounded fallback
        advisory, engine_name = get_llm_response(
            prompt=f"Explain this electricity forecast for ERCOT {pipeline_input['ERCOT_zone']} zone: Predicted {prediction['predicted_usage']} MW, Threshold {prediction['usage_threshold']} MW, Status: {prediction['risk_status']}, Risk: {prediction['risk_probability']}%. Provide concise consumer recommendations.",
            context=context_str,
            chat_history=chat_history
        )

        risk_emoji = "⚠️ HIGH USAGE RISK" if "high" in prediction["risk_status"].lower() else "✅ NORMAL USAGE"
        formatted_response = f"""### ⚡ SmartGrid Demand Forecast: **{risk_emoji}**

- **Projected Regional Demand:** **{prediction['predicted_usage']:,.2f} MW**
- **High-Usage Threshold:** **{prediction['usage_threshold']:,.2f} MW**
- **Stress Probability:** **{prediction['risk_probability']:.2f}%**
- **Classification Status:** `{prediction['risk_status']}`

---
{advisory}
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

    # 2. Check if query is related to ERCOT / Grid / Outages to retrieve PDF context
    is_grid_topic = any(w in lower_q for w in [
        "ercot", "grid", "outage", "electricity", "power", "blackout", "energy", "cdd", "hdd", "reserve", "eea", "texas", "transmission", "substation"
    ])

    context_str = ""
    sources = []
    if is_grid_topic:
        try:
            chunks, embeddings = load_knowledge_base()
            retrieved = retrieve_documents(clean_q, chunks, embeddings, top_k=3)
            context_str = "\n".join([f"[{r['metadata']['source']} P.{r['metadata']['page']}]: {r['text']}" for r in retrieved])
            sources = [
                {"document": r["metadata"]["source"], "page": r["metadata"]["page"], "similarity": round(float(r["score"]), 4)}
                for r in retrieved
            ]
        except Exception as e:
            print(f"[Copilot] RAG query notice: {e}")

    # 3. Query LLM (Gemini / Groq / OpenAI) for answer
    answer, engine_name = get_llm_response(prompt=clean_q, context=context_str, chat_history=chat_history)

    return {
        "text": answer,
        "engine": engine_name,
        "is_forecast": False,
        "sources": sources
    }
