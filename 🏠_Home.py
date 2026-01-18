import streamlit as st
import pandas as pd
import plotly.express as px

# --- VANTAGE ZERO: ELITE CORE UI ---
st.set_page_config(page_title="VANTAGE ZERO | HQ", layout="wide", page_icon="🧬")

# Institutional CSS: Glassmorphism and JetBrains Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Background & Global Font */
    .main { background-color: #05070a; color: #e0e6ed; font-family: 'JetBrains Mono', monospace; }
    
    /* Custom Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(16, 20, 26, 0.6);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: 0.3s ease;
    }
    [data-testid="stMetric"]:hover { border-color: #00ffcc; transform: translateY(-2px); }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00ffcc 0%, #0099ff 100%);
        color: black; font-weight: bold; border: none; border-radius: 5px;
        width: 100%; height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TOP NAVIGATION BAR ---
col_logo, col_stat = st.columns([3, 1])
with col_logo:
    st.title("🧬 VANTAGE ZERO")
    st.caption("INSTITUTIONAL QUANTITATIVE DFS FRAMEWORK // REV 52.0")

with col_stat:
    st.metric("SYSTEM LATENCY", "14ms", "-2ms", delta_color="normal")

st.markdown("---")

# --- DASHBOARD METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("PORTFOLIO SIZE", f"{len(st.session_state.get('nba_portfolio', [])) + len(st.session_state.get('nfl_portfolio', []))}", "ACTIVE")
m2.metric("SLATE PHASE", "DIVISIONAL", "JAN 18")
m3.metric("SIMS EXECUTED", "15.4K", "DAILY")
m4.metric("ALPHA YIELD", "8.2%", "+1.4%")

# --- MAIN HUB CONTENT ---
st.markdown("### 📡 CURRENT THEATRES OF OPERATION")
c1, c2 = st.columns(2)

with c1:
    st.info("🏀 **NBA MODULE**")
    st.write("Molecular Flex Logic: ACTIVE")
    st.write("Late-Swap Delta: 5.2%")
    if st.button("LAUNCH NBA ENGINE"):
        st.switch_page("pages/1_🏀_NBA_Alpha.py")

with c2:
    st.info("🏈 **NFL MODULE**")
    st.write("Primary Stacking: ENABLED")
    st.write("Weather Variance: CHI (18°F)")
    if st.button("LAUNCH NFL ENGINE"):
        st.switch_page("pages/2_🏈_NFL_Alpha.py")

st.markdown("---")

# --- LIVE SLATE TICKER ---
st.markdown("#### 🕒 DIVISIONAL COUNTDOWN")
st.code("""
HOU @ NE | KICKOFF: 15:00 ET | STATUS: PRE-FLIGHT
LAR @ CHI | KICKOFF: 18:30 ET | STATUS: WEATHER WATCH
""")
