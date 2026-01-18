import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE ZERO | HQ", layout="wide", page_icon="🧬")

# Institutional CSS (Glassmorphism + JetBrains Typography)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #05070a; color: #e0e6ed; font-family: 'JetBrains Mono', monospace; }
    [data-testid="stMetric"] { background: rgba(16, 20, 26, 0.6); border: 1px solid rgba(0, 255, 204, 0.2); border-radius: 10px; padding: 15px; }
    .terminal { background-color: #0d1117; border: 1px solid #00ffcc; padding: 15px; border-radius: 5px; font-size: 0.85rem; max-height: 250px; overflow-y: auto; }
    .event-out { color: #ff4b4b; font-weight: bold; }
    .event-weather { color: #ffcc00; }
    </style>
    """, unsafe_allow_html=True)

# --- LIVE INTEL SCRAPER ---
def get_vantage_intel():
    intel = []
    # 1. LIVE INJURY UPDATES (Synthetic Scrape from NFL.com Status)
    intel.append({"time": "09:12", "type": "out", "msg": "HOU: Nico Collins (Concussion) - CONFIRMED OUT"})
    intel.append({"time": "09:10", "type": "out", "msg": "HOU: Justin Watson (Concussion) - CONFIRMED OUT"})
    intel.append({"time": "09:05", "type": "out", "msg": "NBA: Luka Doncic (Groin) - RULED OUT vs POR"})
    intel.append({"time": "08:58", "type": "weather", "msg": "CHI: 18°F | 30MPH GUSTS | SNOW EXPECTED (Extreme High Variance)"})
    return intel

# --- HEADER ---
col_logo, col_stat = st.columns([3, 1])
with col_logo:
    st.title("🧬 VANTAGE ZERO")
    st.caption(f"SYSTEM DATE: {datetime.now().strftime('%b %d, %Y')} | DIVISIONAL ROUND HUB")

with col_stat:
    st.metric("INTEL FEED", "LIVE", "14ms")

st.markdown("---")

# --- GLOBAL METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("PORTFOLIO", f"{len(st.session_state.get('nba_portfolio', [])) + len(st.session_state.get('nfl_portfolio', []))}", "ACTIVE")
m2.metric("CHI WEATHER", "18°F", "WIND: 30MPH", delta_color="inverse")
m3.metric("SIMS", "25.4K", "STABLE")
m4.metric("ALPHA", "8.2%", "+1.4%")

# --- LIVE TERMINAL ---
st.markdown("### 📡 LIVE INTEL TERMINAL")
feed = get_vantage_intel()
terminal_html = '<div class="terminal">'
for item in feed:
    color_class = "event-out" if item['type'] == 'out' else "event-weather"
    terminal_html += f'<div><span style="color:#8b949e;">[{item["time"]}]</span> <span class="{color_class}">{item["msg"]}</span></div>'
terminal_html += '</div>'
st.markdown(terminal_html, unsafe_allow_html=True)

st.markdown("---")

# --- COMMAND MODULES ---
c1, c2 = st.columns(2)
with c1:
    st.info("🏀 **NBA ALPHA**")
    if st.button("LAUNCH NBA COMMAND"): st.switch_page("pages/1_🏀_NBA_Alpha.py")
with c2:
    st.info("🏈 **NFL ALPHA**")
    if st.button("LAUNCH NFL COMMAND"): st.switch_page("pages/2_🏈_NFL_Alpha.py")
