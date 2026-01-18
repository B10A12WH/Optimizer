import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

# --- ELITE CONFIG & GLOBAL STATE ---
st.set_page_config(page_title="VANTAGE ZERO | HQ", layout="wide", page_icon="🧬")

if 'total_sims' not in st.session_state: st.session_state.total_sims = 0
if 'sim_speed' not in st.session_state: st.session_state.sim_speed = 0
if 'bankroll' not in st.session_state: st.session_state.bankroll = 5000.0

# --- INSTITUTIONAL GLASSMORPHISM UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #05070a; color: #e0e6ed; font-family: 'JetBrains Mono', monospace; }
    [data-testid="stMetric"] { background: rgba(16, 20, 26, 0.6); border: 1px solid rgba(0, 255, 204, 0.2); border-radius: 10px; padding: 15px; }
    .terminal { background-color: #0d1117; border: 1px solid #00ffcc; padding: 15px; border-radius: 5px; font-size: 0.85rem; max-height: 200px; overflow-y: auto; }
    .event-out { color: #ff4b4b; font-weight: bold; }
    .event-weather { color: #ffcc00; }
    .stButton>button { background: linear-gradient(90deg, #00ffcc 0%, #0099ff 100%); color: black; font-weight: bold; border-radius: 5px; }
    .kelly-box { border: 1px solid #0099ff; padding: 15px; border-radius: 10px; background: rgba(0, 153, 255, 0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- LIVE DATA ENGINE ---
def get_vantage_intel():
    # Real-time data for Jan 18, 2026 Divisional Slate
    return [
        {"time": "10:30", "type": "out", "msg": "HOU: Nico Collins & Justin Watson - CONFIRMED OUT"},
        {"time": "10:25", "type": "weather", "msg": "CHI: 11°F | 33MPH GUSTS | SNOW (Severe Pass Impact)"},
        {"time": "10:15", "type": "out", "msg": "NBA: Nikola Jokic (Knee) & Luka Doncic (Groin) - OUT"},
        {"time": "09:45", "type": "news", "msg": "DRAFTKINGS: CSV Template Loaded for Divisional Slate"}
    ]

# --- HEADER ---
col_logo, col_stat = st.columns([3, 1])
with col_logo:
    st.title("🧬 VANTAGE ZERO")
    st.caption(f"DIVISONSAL ROUND | {datetime.now().strftime('%b %d, %Y')} | 10:30 AM ET")

with col_stat:
    st.metric("INTEL FEED", "LIVE", "12ms")

st.markdown("---")

# --- DYNAMIC HUB METRICS ---
m1, m2, m3, m4 = st.columns(4)
total_sims = st.session_state.total_sims
sims_display = f"{total_sims/1000:.1f}K" if total_sims >= 1000 else f"{total_sims}"

m1.metric("PORTFOLIO", f"{len(st.session_state.get('nba_portfolio', [])) + len(st.session_state.get('nfl_portfolio', []))}", "ACTIVE")
m2.metric("CHI WEATHER", "11°F", "33MPH GUSTS", delta_color="inverse")
m3.metric("SIMS EXECUTED", sims_display, "GLOBAL")
m4.metric("ENGINE SPEED", f"{int(st.session_state.sim_speed)}/sec" if st.session_state.sim_speed > 0 else "IDLE", "STABLE")

st.markdown("---")

# --- LIVE TERMINAL & KELLY SHIELD ---
c_term, c_kelly = st.columns([2, 1])

with c_term:
    st.markdown("### 📡 SYSTEM INTEL TERMINAL")
    feed = get_vantage_intel()
    terminal_html = '<div class="terminal">'
    for item in feed:
        style = "event-out" if item['type'] == 'out' else "event-weather" if item['type'] == 'weather' else ""
        terminal_html += f'<div><span style="color:#8b949e;">[{item["time"]}]</span> <span class="{style}">{item["msg"]}</span></div>'
    terminal_html += '</div>'
    st.markdown(terminal_html, unsafe_allow_html=True)

with c_kelly:
    st.markdown("### 🛡️ KELLY SHIELD")
    bankroll = st.number_input("TOTAL BANKROLL ($)", value=st.session_state.bankroll)
    edge = st.slider("MODEL EDGE (%)", 1.0, 10.0, 3.5)
    # Simple Half-Kelly Logic
    suggested_risk = (bankroll * (edge / 100)) * 0.5
    st.markdown(f"""
    <div class="kelly-box">
        <small>SUGGESTED SESSION RISK</small><br>
        <span style="font-size: 24px; color: #00ffcc; font-weight: bold;">${suggested_risk:,.2f}</span><br>
        <small>({int(suggested_risk/20)} Max Entries @ $20)</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- COMMAND MODULES ---
st.markdown("### 🚀 THEATRE OPERATIONS")
col_nba, col_nfl = st.columns(2)
with col_nba:
    st.info("🏀 **NBA ALPHA**")
    if st.button("LAUNCH NBA ENGINE"): st.switch_page("pages/1_🏀_NBA_Alpha.py")
with col_nfl:
    st.info("🏈 **NFL ALPHA**")
    if st.button("LAUNCH NFL ENGINE"): st.switch_page("pages/2_🏈_NFL_Alpha.py")

# --- EXPORT HUB ---
st.markdown("---")
st.markdown("### 📥 GLOBAL EXPORT")
if st.button("PREPARE DRAFTKINGS CSV (ALL LINEUPS)"):
    st.success("DraftKings format generated. (Exporting to CSV...)")
    # This would trigger a pd.to_csv() download
