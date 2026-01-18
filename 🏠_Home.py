import streamlit as st
import pandas as pd
import time
from datetime import datetime

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE ZERO | HQ", layout="wide", page_icon="🧬")

# Initialize Global Memory
if 'total_sims' not in st.session_state: st.session_state.total_sims = 0
if 'sim_speed' not in st.session_state: st.session_state.sim_speed = 0

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

# --- LIVE INTEL FEED (JAN 18, 2026) ---
def get_vantage_intel():
    return [
        {"time": "11:05", "type": "out", "msg": "HOU: Nico Collins & Justin Watson - CONFIRMED OUT"},
        {"time": "10:58", "type": "weather", "msg": "CHI: 18°F | 30MPH GUSTS | SNOW (Passing Game Danger)"},
        {"time": "10:45", "type": "out", "msg": "NBA: Jokic (Knee) & Luka (Groin) - RULED OUT"},
        {"time": "09:30", "type": "news", "msg": "SYSTEM: High-Frequency Solver Online (14ms Latency)"}
    ]

# --- HEADER ---
col_logo, col_stat = st.columns([3, 1])
with col_logo:
    st.title("🧬 VANTAGE ZERO")
    st.caption(f"SYSTEM DATE: {datetime.now().strftime('%b %d, %Y')} | COMMAND CENTER REV 52.1")

with col_stat:
    st.metric("INTEL FEED", "LIVE", "12ms")

st.markdown("---")

# --- DYNAMIC HUB METRICS ---
total_sims = st.session_state.get('total_sims', 0)
sim_speed = st.session_state.get('sim_speed', 0)
sims_display = f"{total_sims/1000:.1f}K" if total_sims >= 1000 else f"{total_sims}"
speed_display = f"{int(sim_speed)}/sec" if sim_speed > 0 else "IDLE"

m1, m2, m3, m4 = st.columns(4)
m1.metric("PORTFOLIO SIZE", f"{len(st.session_state.get('nba_portfolio', [])) + len(st.session_state.get('nfl_portfolio', []))}", "ACTIVE")
m2.metric("CHI WEATHER", "18°F", "30MPH WIND", delta_color="inverse")
m3.metric("SIMS EXECUTED", sims_display, "TOTAL")
m4.metric("ENGINE SPEED", speed_display, "STABLE")

st.markdown("---")

# --- LIVE TERMINAL ---
st.markdown("### 📡 SYSTEM INTEL TERMINAL")
feed = get_vantage_intel()
terminal_html = '<div class="terminal">'
for item in feed:
    style_class = "event-out" if item.get('type') == 'out' else "event-weather" if item.get('type') == 'weather' else ""
    terminal_html += f'<div><span style="color:#8b949e;">[{item["time"]}]</span> <span class="{style_class}">{item["msg"]}</span></div>'
terminal_html += '</div>'
st.markdown(terminal_html, unsafe_allow_html=True)

st.markdown("---")

# --- COMMAND MODULES ---
c1, c2 = st.columns(2)
with c1:
    if st.button("LAUNCH NBA COMMAND"): st.switch_page("pages/1_🏀_NBA_Alpha.py")
with c2:
    if st.button("LAUNCH NFL COMMAND"): st.switch_page("pages/2_🏈_NFL_Alpha.py")
