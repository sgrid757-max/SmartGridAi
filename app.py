import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import date, datetime

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Automatically bridge Streamlit secrets to os.environ so copilot_engine can read them
try:
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# Internal imports
from agentic_ai import (
    run as run_agentic_ai,
    to_pipeline_input,
    derive as derive_tool,
    ZONES,
    ZONE_DEFAULTS
)
from ml.prediction import predict_grid_status
from rag.smartgrid_pipeline import run_smartgrid, load_knowledge_base
from rag.retrieval import retrieve_documents
from rag.llm import generate_answer
from copilot_engine import ask_copilot

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartGrid AI | Energy Intelligence & Agentic Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DESIGN SYSTEM (CSS)
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:            #121212;
    --surface:       #1a1a1a;
    --surface-2:     #242424;
    --surface-3:     #2e2e2e;
    --border:        #383838;
    --border-strong: #4d4d4d;
    --text:          #e0e0e0;
    --text-muted:    #a0a0a0;
    --text-faint:    #707070;
    --accent:        #d4a373;
    --accent-soft:   rgba(212, 163, 115, 0.1);
    --accent-hover:  #e5b383;
    --good:          #81b29a;
    --good-soft:     rgba(129, 178, 154, 0.1);
    --bad:           #e07a5f;
    --bad-soft:      rgba(224, 122, 95, 0.1);
    --radius:        8px;
}

/* ---------- Base ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4, h5 {
    color: var(--text) !important;
    letter-spacing: -0.01em;
    font-weight: 600 !important;
}

p, li, span, label { color: var(--text); }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ---------- Header ---------- */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    background: var(--surface-2);
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    border: 1px solid var(--border);
}

.brand-title {
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1.1;
}

.brand-title em {
    font-style: normal;
    color: var(--accent);
}

.brand-sub {
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 2px;
}

.status-row { display: flex; gap: 8px; flex-wrap: wrap; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 500;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-muted);
}

.status-pill::before {
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--good);
}

/* ---------- Tabs ---------- */
div[data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid var(--border);
}

button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    padding: 8px 14px !important;
    border-radius: 6px 6px 0 0 !important;
}

button[data-baseweb="tab"]:hover { color: var(--text) !important; }

button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; background: var(--surface) !important; }

div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 2px !important; }
div[data-baseweb="tab-border"] { display: none; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h2 {
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}

/* ---------- Section headings ---------- */
.section-title {
    color: var(--text);
    font-size: 16px;
    font-weight: 600;
    margin: 4px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-title::before {
    content: "";
    width: 2px;
    height: 14px;
    border-radius: 1px;
    background: var(--accent);
}

/* ---------- Metric cards ---------- */
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    min-height: 110px;
}

.metric-label {
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    color: var(--text);
    font-size: 24px;
    font-weight: 600;
    margin-top: 6px;
}

.metric-note {
    color: var(--text-faint);
    font-size: 12px;
    margin-top: 4px;
}

/* ---------- Risk banners ---------- */
.status-high, .status-normal {
    padding: 12px 16px;
    border-radius: var(--radius);
    font-weight: 500;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-high {
    background: var(--bad-soft);
    color: #e6917c;
    border: 1px solid rgba(224, 122, 95, 0.3);
    border-left: 3px solid var(--bad);
}

.status-normal {
    background: var(--good-soft);
    color: #9ac2b0;
    border: 1px solid rgba(129, 178, 154, 0.3);
    border-left: 3px solid var(--good);
}

/* ---------- Agent derivation card ---------- */
.agent-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 14px;
    margin: 10px 0 12px 0;
}

.agent-card-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
}

.agent-card-title code {
    background: var(--surface);
    color: var(--accent);
    padding: 1px 4px;
    border-radius: 4px;
    font-size: 11px;
}

.agent-pill {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 5px;
    font-size: 11.5px;
    margin: 2px 4px 2px 0;
    color: var(--text);
}

.agent-pill b {
    color: var(--text-muted);
    font-weight: 500;
    font-size: 11px;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background-color: var(--surface-2);
    color: var(--text);
    border-radius: 6px;
    border: 1px solid var(--border);
    height: 38px;
    font-weight: 500;
    font-size: 13.5px;
}

.stButton > button:hover {
    background-color: var(--surface-3);
    border-color: var(--border-strong);
    color: var(--accent);
}

/* Quick chips */
.chip-row .stButton > button {
    background-color: var(--surface) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    font-size: 12.5px !important;
    height: auto !important;
    min-height: 38px;
    padding: 6px 10px !important;
    border-radius: 6px !important;
}

.chip-row .stButton > button:hover {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: var(--border-strong) !important;
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-bottom: 10px;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: var(--surface-2);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--text);
    line-height: 1.5;
    font-size: 14px;
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="textarea"],
div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}

input, textarea {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-size: 13.5px !important;
}

input::placeholder, textarea::placeholder {
    color: var(--text-faint) !important;
    -webkit-text-fill-color: var(--text-faint) !important;
}

/* ---------- Floating Chat Window ---------- */
div[class*="st-key-floating_copilot_window"] {
    position: fixed !important;
    bottom: 20px !important;
    right: 20px !important;
    width: 400px !important;
    max-width: calc(100vw - 24px) !important;
    height: 500px !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
    z-index: 999999 !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 12px !important;
}

div[class*="st-key-floating_copilot_fab_container"] button {
    background: var(--surface-2) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 999px !important;
    padding: 10px 20px !important;
    border: 1px solid var(--border-strong) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# UI HELPERS (presentation only)
# ==========================================

def metric_card(label, value, note):
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def pill(label, value):
    return f'<span class="agent-pill"><b>{label}</b>{value}</span>'


# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================

if "last_input" not in st.session_state:
    st.session_state.last_input = {
        "latitude": 32.75,
        "longitude": -96.77,
        "ERCOT_zone": "NORTH",
        "date": date.today().isoformat(),
        "temperature_mean": 25.0,
        "temperature_max": 32.0,
        "temperature_min": 18.0,
        "precipitation_mm": 0.0,
        "wind_speed_mps": 5.0,
        "snowfall_cm": 0.0,
        "humidity": 60.0
    }

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm the **SmartGrid AI Assistant**, powered by Agentic AI and ERCOT grid intelligence.\n\n"
                "Here's what I can do:\n"
                "1. **Describe conditions in plain language**, for example *'Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW'*.\n"
                "2. The **Agentic AI** extracts the features and calls the `derive` tool (Cooling Degree Days, temperature range, weekend flag, ERCOT zone), then forecasts grid stress and electricity demand.\n"
                "3. **Ask domain questions** about ERCOT reserves, EEA levels, weather impact, or consumer advisories."
            ),
            "meta": None
        }
    ]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "floating_chat_history" not in st.session_state:
    st.session_state.floating_chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your **AI Chatbot**.\n\n"
                "Ask me anything in the input box below and click **Submit**, and I will provide the best answer!"
            ),
            "sources": []
        }
    ]

if "copilot_open" not in st.session_state:
    st.session_state.copilot_open = False

if "dock_pending" not in st.session_state:
    st.session_state.dock_pending = False

# ==========================================
# 4. HEADER
# ==========================================

st.markdown("""
<div class="app-header">
    <div class="brand">
        <div class="brand-mark">⚡</div>
        <div>
            <div class="brand-title">SmartGrid <em>AI</em></div>
            <div class="brand-sub">Electricity demand forecasting, tool-calling intelligence and consumer advisory</div>
        </div>
    </div>
    <div class="status-row">
        <span class="status-pill">ML models active</span>
        <span class="status-pill">Agentic tools ready</span>
        <span class="status-pill">Floating AI Copilot active</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================
# 5. SIDEBAR CONTROLS
# ==========================================

st.sidebar.markdown("### System Settings")
st.sidebar.caption("Configure models, API keys and the manual forecaster.")

# Model & API settings expander
with st.sidebar.expander("🔑 Agent & API Configuration", expanded=False):
    groq_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Used by Agentic AI for tool calling (Groq / OpenAI API compatible)."
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input
        st.session_state["groq_api_key"] = groq_key_input

    gemini_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Used for RAG grid advisory generation and Copilot answers."
    )
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input
        st.session_state["gemini_api_key"] = gemini_key_input

    openai_key_input = st.text_input(
        "OpenAI API Key (Optional)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Used for OpenAI GPT-4o-mini Copilot queries."
    )
    if openai_key_input:
        os.environ["OPENAI_API_KEY"] = openai_key_input
        st.session_state["openai_api_key"] = openai_key_input

    agent_model = st.selectbox(
        "Agentic LLM Model",
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    os.environ["GROQ_MODEL"] = agent_model

st.sidebar.divider()

# Manual location & weather inputs
st.sidebar.markdown("### Location")
zone = st.sidebar.selectbox("ERCOT Zone", ["COAST", "NORTH", "SOUTH", "WEST"], index=["COAST", "NORTH", "SOUTH", "WEST"].index(st.session_state.last_input.get("ERCOT_zone", "NORTH")))

loc_c1, loc_c2 = st.sidebar.columns(2)
with loc_c1:
    latitude = st.number_input("Latitude", value=float(st.session_state.last_input.get("latitude", 32.75)), format="%.5f")
with loc_c2:
    longitude = st.number_input("Longitude", value=float(st.session_state.last_input.get("longitude", -96.77)), format="%.5f")

try:
    init_date = date.fromisoformat(st.session_state.last_input.get("date", date.today().isoformat()))
except Exception:
    init_date = date.today()

selected_date = st.sidebar.date_input("Prediction Date", value=init_date)

st.sidebar.divider()
st.sidebar.markdown("### Weather Conditions")

temperature_mean = st.sidebar.number_input("Average Temperature (°C)", value=float(st.session_state.last_input.get("temperature_mean", 25.0)), step=1.0)

t_c1, t_c2 = st.sidebar.columns(2)
with t_c1:
    temperature_max = st.number_input("Max Temp (°C)", value=float(st.session_state.last_input.get("temperature_max", 32.0)), step=1.0)
with t_c2:
    temperature_min = st.number_input("Min Temp (°C)", value=float(st.session_state.last_input.get("temperature_min", 18.0)), step=1.0)

humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=int(st.session_state.last_input.get("humidity", 60)))

w_c1, w_c2 = st.sidebar.columns(2)
with w_c1:
    wind_speed = st.number_input("Wind (m/s)", min_value=0.0, value=float(st.session_state.last_input.get("wind_speed_mps", 5.0)))
with w_c2:
    precipitation = st.number_input("Rain (mm)", min_value=0.0, value=float(st.session_state.last_input.get("precipitation_mm", 0.0)))

snowfall = st.sidebar.number_input("Snowfall (cm)", min_value=0.0, value=float(st.session_state.last_input.get("snowfall_cm", 0.0)))

st.sidebar.divider()
predict_button = st.sidebar.button("⚡ Run Manual Forecast", type="primary", use_container_width=True)

# Update sidebar prediction action
if predict_button:
    if temperature_min > temperature_mean:
        st.sidebar.error("Minimum temperature cannot exceed average temperature.")
    elif temperature_mean > temperature_max:
        st.sidebar.error("Average temperature cannot exceed maximum temperature.")
    else:
        manual_input = {
            "latitude": latitude,
            "longitude": longitude,
            "ERCOT_zone": zone,
            "date": selected_date.isoformat(),
            "temperature_mean": temperature_mean,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "precipitation_mm": precipitation,
            "wind_speed_mps": wind_speed,
            "snowfall_cm": snowfall,
            "humidity": humidity
        }
        with st.spinner("Analyzing energy demand and retrieving grid knowledge..."):
            try:
                res = run_smartgrid(manual_input)
                st.session_state.prediction_result = res
                st.session_state.last_input = manual_input
                st.sidebar.success("Forecast updated!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# ==========================================
# 6. MAIN NAVIGATION TABS
# ==========================================

tab_chat, tab_dashboard, tab_inspector = st.tabs([
    "AI Chatbot",
    "Forecaster Dashboard",
    "Agentic AI Inspector"
])

# -------------------------------------------------------------
# TAB 1: AI CHATBOT
# -------------------------------------------------------------
with tab_chat:
    section("AI Chatbot")
    st.caption("Ask anything: describe conditions, ask general questions, or inquire about energy and grid intelligence.")

    # Quick prompt chips
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    chip_cols = st.columns(4)
    with chip_cols[0]:
        if st.button("🔥 Dallas heatwave (Nov 2026)", key="chip_dallas", use_container_width=True):
            st.session_state.pending_prompt = "Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW"
    with chip_cols[1]:
        if st.button("🌊 Houston summer coastal peak", key="chip_houston", use_container_width=True):
            st.session_state.pending_prompt = "July 22 2026, Houston, high 40C, low 29C, avg 34C, humidity 82%, wind 3 m/s, rain 0mm, load 68000 MW"
    with chip_cols[2]:
        if st.button("❄️ Austin winter freeze", key="chip_austin", use_container_width=True):
            st.session_state.pending_prompt = "Jan 18 2026, Austin, high 2C, low -7C, avg -3C, humidity 65%, wind 10 m/s, snowfall 3cm, load 62000 MW"
    with chip_cols[3]:
        if st.button("❓ ERCOT EEA alert rules", key="chip_eea", use_container_width=True):
            st.session_state.pending_prompt = "What are the ERCOT Energy Emergency Alert (EEA) levels, reserve thresholds, and consumer actions?"
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # Display Chat History
    for msg_idx, msg in enumerate(st.session_state.chat_messages):
        avatar = "🧑" if msg["role"] == "user" else "⚡"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # If message contains structured meta (from an Agentic execution)
            if msg.get("meta"):
                meta = msg["meta"]
                ext = meta.get("extracted", {})
                pred = meta.get("prediction", {})

                if ext:
                    pills = "".join([
                        pill("Date", ext.get('date', 'Unknown')),
                        pill("Zone", ext.get('zone', 'NORTH')),
                        pill("Mean", f"{ext.get('temperature_mean')}°C"),
                        pill("Max", f"{ext.get('temperature_max')}°C"),
                        pill("Min", f"{ext.get('temperature_min')}°C"),
                        pill("CDD", ext.get('CDD')),
                        pill("Range", f"{ext.get('temp_range')}°C"),
                        pill("Weekend", 'Yes' if ext.get('weekend') == 1 else 'No'),
                        pill("Zone code", ext.get('ERCOT_zone_encoded')),
                    ])
                    st.markdown(f"""
                    <div class="agent-card">
                        <div class="agent-card-title">Agent extraction &amp; derived features &middot; tool call <code>derive()</code></div>
                        <div>{pills}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Prediction metric summary
                if pred:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Predicted Usage", f"{pred['predicted_usage']:,.2f} MW")
                    with col_m2:
                        st.metric("Usage Threshold", f"{pred['usage_threshold']:,.2f} MW")
                    with col_m3:
                        st.metric("High Usage Probability", f"{pred['risk_probability']:.2f}%")

                    if ext.get("zone_usage"):
                        delta = ext["zone_usage"] - pred["predicted_usage"]
                        st.info(f"**Reported load:** {ext['zone_usage']:,.0f} MW  |  **Model forecast:** {pred['predicted_usage']:,.0f} MW  |  **Variance:** {delta:+,.0f} MW")

                # Show sources if any
                sources = meta.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved knowledge sources ({len(sources)} documents cited)"):
                        for s in sources:
                            st.write(f"- **{s.get('document')}** (Page {s.get('page')}) — Similarity: `{s.get('similarity_score', 0):.4f}`")

    # Primary Input Box & Submit Button
    st.divider()
    with st.form("tab1_ai_chatbot_form", clear_on_submit=True):
        col_in, col_btn = st.columns([0.80, 0.20])
        with col_in:
            tab1_input_val = st.text_input(
                "Ask AI Chatbot",
                placeholder="Ask anything...",
                label_visibility="collapsed",
                key="tab1_main_input_box"
            )
        with col_btn:
            tab1_submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

    # Process incoming or pending prompt
    chat_input_val = st.chat_input("Ask anything...")
    prompt_to_process = (tab1_input_val if tab1_submitted and tab1_input_val.strip() else None) or chat_input_val or st.session_state.pending_prompt

    if prompt_to_process:
        st.session_state.pending_prompt = None

        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt_to_process,
            "meta": None
        })

        # Process message
        with st.spinner("Agentic AI analyzing scenario and synthesizing grid intelligence..."):
            lower_p = prompt_to_process.lower()
            # Detect if message contains weather or grid conditions
            is_condition_query = any(k in lower_p for k in [
                "high", "low", "avg", "temp", "celsius", "humidity", "mw", "load", "wind", "rain", "dallas", "houston", "austin", "ercot", "zone", "weather"
            ]) and any(char.isdigit() for char in prompt_to_process)

            if is_condition_query:
                # 1. Run Agentic AI Feature Extractor
                extracted_json = run_agentic_ai(prompt_to_process)
                pipeline_in = to_pipeline_input(extracted_json, prompt_to_process)

                # 2. Run ML Prediction
                pred_result = predict_grid_status(pipeline_in)

                # 3. Retrieve Documents
                try:
                    chunks, embeddings = load_knowledge_base()
                    retrieved = retrieve_documents(pred_result["rag_query"], chunks, embeddings, top_k=3)
                    advisory = generate_answer(pred_result["rag_query"], retrieved)
                    sources_list = [
                        {
                            "document": item["metadata"]["source"],
                            "page": item["metadata"]["page"],
                            "similarity_score": round(float(item["score"]), 4)
                        }
                        for item in retrieved
                    ]
                except Exception as ex:
                    advisory = f"ML Prediction completed successfully. RAG Retrieval note: {ex}"
                    sources_list = []

                # Update session state for dashboard sync
                st.session_state.prediction_result = {
                    "predicted_usage": pred_result["predicted_usage"],
                    "risk_status": pred_result["risk_status"],
                    "risk_probability": pred_result["risk_probability"],
                    "usage_threshold": pred_result["usage_threshold"],
                    "explanation": advisory,
                    "sources": sources_list
                }
                st.session_state.last_input = pipeline_in

                # Construct assistant reply text
                risk_emoji = "⚠️ HIGH USAGE RISK" if "high" in pred_result["risk_status"].lower() else "✅ NORMAL USAGE"
                reply_text = f"""### SmartGrid AI Assessment: **{risk_emoji}**

**Forecast summary**
- **Predicted demand:** **{pred_result['predicted_usage']:,.2f} MW**
- **Threshold limit:** **{pred_result['usage_threshold']:,.2f} MW**
- **Risk probability:** **{pred_result['risk_probability']:.2f}%**
- **Classification status:** `{pred_result['risk_status']}`

---
{advisory}
"""
                meta_dict = {
                    "extracted": {
                        "date": pipeline_in["date"],
                        "zone": pipeline_in["ERCOT_zone"],
                        "temperature_mean": extracted_json.get("temperature_mean"),
                        "temperature_max": extracted_json.get("temperature_max"),
                        "temperature_min": extracted_json.get("temperature_min"),
                        "CDD": extracted_json.get("CDD"),
                        "temp_range": extracted_json.get("temp_range"),
                        "weekend": extracted_json.get("weekend"),
                        "ERCOT_zone_encoded": extracted_json.get("ERCOT_zone_encoded"),
                        "zone_usage": extracted_json.get("zone_usage")
                    },
                    "prediction": pred_result,
                    "sources": sources_list
                }

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": reply_text,
                    "meta": meta_dict
                })

            else:
                # General Question: Query RAG directly
                try:
                    chunks, embeddings = load_knowledge_base()
                    retrieved = retrieve_documents(prompt_to_process, chunks, embeddings, top_k=3)
                    advisory = generate_answer(prompt_to_process, retrieved)
                    sources_list = [
                        {
                            "document": item["metadata"]["source"],
                            "page": item["metadata"]["page"],
                            "similarity_score": round(float(item["score"]), 4)
                        }
                        for item in retrieved
                    ]
                except Exception as ex:
                    advisory = f"I am unable to access the PDF knowledge base currently ({ex})."
                    sources_list = []

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": advisory,
                    "meta": {"sources": sources_list} if sources_list else None
                })

        st.rerun()

    # Chat reset button
    st.markdown('<div class="clear-row">', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns([0.8, 0.2])
    with col_c2:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: FORECASTER DASHBOARD
# -------------------------------------------------------------
with tab_dashboard:
    result = st.session_state.prediction_result
    input_data = st.session_state.last_input

    if result is None:
        section("Energy Intelligence Dashboard")
        st.write("Set the location and weather conditions in the sidebar, or run a scenario in the chatbot, to see forecast analytics here.")
        st.write("")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(metric_card("Machine learning", "Demand Forecast", "XGBoost &amp; regression modeling"), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Risk analytics", "Risk Detection", "Classification probability &amp; thresholds"), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Agentic AI + RAG", "Tool Derivation", "CDD, weekend flag, grounded advisories"), unsafe_allow_html=True)

        st.write("")
        st.info("Tip: try the Dallas scenario from the **Chatbot & Agent** tab, or press **Run Manual Forecast** in the sidebar.")

    else:
        section("Prediction Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(metric_card("Predicted electricity usage", f"{result['predicted_usage']:,.2f} MW", "Model regression forecast"), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("High-usage probability", f"{result['risk_probability']:.2f}%", "Classifier stress probability"), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Usage threshold", f"{result['usage_threshold']:,.2f} MW", "75th percentile stress boundary"), unsafe_allow_html=True)

        st.write("")

        # Risk Assessment Banner
        section("Risk Assessment")
        risk_status = str(result.get("risk_status", "")).strip().lower()
        is_high_risk = risk_status in ["high usage risk", "high usage", "high risk"]

        if is_high_risk:
            st.markdown("""
            <div class="status-high">
                ⚠️ <span>High usage risk detected. Elevated strain anticipated on ERCOT transmission infrastructure.</span>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.warning("The model has identified extreme thermal or demand conditions exceeding safe threshold limits.")
        else:
            st.markdown("""
            <div class="status-normal">
                ✅ <span>Normal usage classification. Electricity demand is projected within standard operating reserves.</span>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.success("The model has classified conditions within normal reserve margins.")

        # Usage vs Threshold Comparison
        st.divider()
        section("Usage vs Threshold")

        usage_df = pd.DataFrame({
            "Category": ["Predicted Usage", "High Usage Threshold"],
            "Value": [result["predicted_usage"], result["usage_threshold"]]
        })

        st.bar_chart(usage_df, x="Category", y="Value", horizontal=True)

        difference = result["predicted_usage"] - result["usage_threshold"]
        if difference > 0:
            st.write(f"Predicted usage is **{difference:,.2f} MW** above the high-usage threshold.")
        else:
            st.write(f"Predicted usage is **{abs(difference):,.2f} MW** below the high-usage threshold.")

        # Weather & Input Conditions Summary
        st.divider()
        section("Input Conditions")
        w_c1, w_c2, w_c3, w_c4 = st.columns(4)
        with w_c1:
            st.metric("Average Temperature", f"{input_data['temperature_mean']} °C")
        with w_c2:
            st.metric("Humidity", f"{input_data['humidity']}%")
        with w_c3:
            st.metric("Wind Speed", f"{input_data['wind_speed_mps']} m/s")
        with w_c4:
            st.metric("Precipitation", f"{input_data['precipitation_mm']} mm")

        st.caption(f"**ERCOT Zone:** {input_data.get('ERCOT_zone')}  ·  **Date:** {input_data.get('date')}  ·  **Coordinates:** {input_data.get('latitude')}, {input_data.get('longitude')}")

        # AI-Powered Grid Advisory
        st.divider()
        section("AI-Powered Grid Advisory")
        st.markdown(result["explanation"])

        # Knowledge Retrieval Sources
        if result.get("sources"):
            st.divider()
            section("Knowledge Retrieval Sources")
            st.caption("Retrieved ERCOT technical documents supporting the generated explanation.")

            for idx, src in enumerate(result["sources"], start=1):
                with st.expander(f"📄 Source {idx}: {src.get('document')} (Page {src.get('page')})"):
                    st.write("**Document:**", src.get("document"))
                    st.write("**Page:**", src.get("page"))
                    st.write("**Similarity Score:**", f"{src.get('similarity_score', 0):.4f}")

# -------------------------------------------------------------
# TAB 3: AGENTIC AI INSPECTOR (DEVELOPER / FUNCTION CALLING STUDIO)
# -------------------------------------------------------------
with tab_inspector:
    section("Agentic AI Function Calling Studio")
    st.caption("Inspect the tool-calling workflow, the `derive` parameter extraction, and the output JSON matching the project dataset.")

    test_prompt = st.text_area(
        "Condition description",
        value="Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW",
        height=90
    )

    if st.button("🚀 Execute Agentic Tool Calling", type="primary"):
        with st.spinner("Invoking Agentic AI & tool pipeline..."):
            extracted_output = run_agentic_ai(test_prompt)

            st.success("Agentic tool execution completed.")

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("**Extracted features (JSON)**")
                st.json(extracted_output)

            with col_t2:
                st.markdown("**Tool call diagnostics (`derive`)**")
                st.markdown(f"""
                - **Cooling Degree Days (CDD):** `{extracted_output.get('CDD')}` (max(0, Mean Temp − 18.3°C))
                - **Temperature Range:** `{extracted_output.get('temp_range')}` °C (High − Low)
                - **Weekend Flag:** `{extracted_output.get('weekend')}` (1 = Weekend, 0 = Weekday)
                - **ERCOT Zone Encoded:** `{extracted_output.get('ERCOT_zone_encoded')}` (index in `{ZONES}`)
                - **Reported Load:** `{extracted_output.get('zone_usage')}` MW
                - **Precipitation:** `{extracted_output.get('precipitation_mm')}` mm
                - **Wind Speed:** `{extracted_output.get('wind_speed_mps')}` m/s
                """)

                # Show pipeline mapping
                pip_in = to_pipeline_input(extracted_output, test_prompt)
                st.markdown("**Dataset feature alignment**")
                st.code(json.dumps(pip_in, indent=2), language="json")

# ==========================================
# 7. FOOTER
# ==========================================

st.divider()
st.markdown("""
<div class="app-footer">
    SmartGrid AI &middot; Autonomous energy demand forecasting, agentic AI &amp; consumer advisory<br>
    © 2026 SmartGrid AI. All rights reserved.
</div>
""", unsafe_allow_html=True)

# ==========================================
# 8. FLOATING AI CHATBOT DOCKED AT BOTTOM-RIGHT
# ==========================================

if not st.session_state.copilot_open:
    # Closed state: sleek floating action button (FAB) anchored to bottom-right
    with st.container(key="floating_copilot_fab_container"):
        if st.button("💬 AI Chatbot", key="floating_fab_trigger_btn", help="Click to open AI Chatbot"):
            st.session_state.copilot_open = True
            st.rerun()
else:
    # Open state: floating chat panel docked directly to the bottom-right on the SAME page
    with st.container(key="floating_copilot_window"):
        # Top Header Bar - Named ONLY "AI Chatbot"
        col_h_left, col_h_right = st.columns([0.76, 0.24])
        with col_h_left:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
                <span style="font-size: 20px;">🤖</span>
                <div>
                    <div style="font-weight: 700; font-size: 14.5px; color: #e0e0e0; line-height: 1.2;">AI Chatbot</div>
                    <div style="font-size: 11px; color: #a0a0a0;"><span style="color: #81b29a;">● Online</span> &middot; Ask anything</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_h_right:
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("🗑️", key="copilot_clear_top", help="Clear chat"):
                    st.session_state.floating_chat_history = []
                    st.session_state.dock_pending = False
                    st.rerun()
            with c_btn2:
                if st.button("✕", key="copilot_close_top", help="Close AI Chatbot"):
                    st.session_state.copilot_open = False
                    st.rerun()

        # Scrollable Chat Container (compact height so input box is always visible)
        with st.container(height=290):
            if not st.session_state.floating_chat_history:
                st.markdown("""
                <div style="text-align: center; padding: 24px 10px; color: #a0a0a0;">
                    <div style="font-size: 32px; margin-bottom: 6px;">🤖</div>
                    <div style="font-weight: 600; color: #e0e0e0; font-size: 14px;">AI Chatbot</div>
                    <div style="font-size: 12px; margin-top: 4px; line-height: 1.4;">
                        Ask me anything! Type your question below and click <b>Submit</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.floating_chat_history:
                    avatar = "🧑" if msg["role"] == "user" else "🤖"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])
                        if msg.get("sources"):
                            with st.expander(f"📚 Cited Sources ({len(msg['sources'])} docs)"):
                                for s in msg["sources"]:
                                    st.write(f"- **{s['document']}** (Page {s['page']})")

                # "Typing..." bubble: shown while the API request is in flight.
                # The user's message is already on screen (added before the rerun).
                if st.session_state.dock_pending:
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Typing..."):
                            try:
                                ans = ask_copilot(
                                    st.session_state.floating_chat_history[-1]["content"],
                                    st.session_state.floating_chat_history,
                                )
                            except Exception as ex:  # ask_copilot shouldn't raise, but never crash the UI
                                ans = {"text": f"⚠️ Something went wrong: {ex}", "sources": []}
                    st.session_state.floating_chat_history.append({
                        "role": "assistant",
                        "content": ans["text"],
                        "sources": ans.get("sources", [])
                    })
                    st.session_state.dock_pending = False
                    st.rerun()

        # Input box where user can ask anything and submit
        with st.form("ai_chatbot_dock_form", clear_on_submit=True):
            col_inp, col_snd = st.columns([0.76, 0.24])
            with col_inp:
                user_dock_msg = st.text_input(
                    "Ask anything",
                    placeholder="Ask anything...",
                    label_visibility="collapsed",
                    key="ai_chatbot_dock_text_input"
                )
            with col_snd:
                dock_submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

        # Step 1 of 2: store the user's message and rerun so it shows immediately.
        # Step 2 (above, inside the chat window) calls the API and shows "Typing...".
        if dock_submitted and user_dock_msg and user_dock_msg.strip() and not st.session_state.dock_pending:
            st.session_state.floating_chat_history.append({"role": "user", "content": user_dock_msg.strip()})
            st.session_state.dock_pending = True
            st.rerun()