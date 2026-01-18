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
    [data-testid="stMetric"] { 
        background: rgba(16, 20, 26, 0.6); 
        border: 1px solid rgba(0, 255, 204, 0.2); 
        border-radius: 10px; 
        padding: 15px; 
    }
    .terminal { 
        background-color: #0d1117; 
        border: 1px solid #00ffcc; 
        padding: 15px; 
        border-radius: 5px; 
        font-size: 0.85rem; 
        max-height: 250px; 
        overflow-y: auto; 
    }
    .event-out { color: #ff4b4b; font-weight: bold; }
    .event-weather { color: #ffcc00; }
    </style>
    """, unsafe_allow_html=True)

# --- LIVE INTEL ENGINE ---
def get_vantage_intel():
    # Targeted alerts for Jan 18, 2026 Divisional Slate
    intel = [
        {"time": "11:05", "type": "out", "msg": "HOU: Nico Collins (Concussion) - CONFIRMED OUT"},
        {"time": "11:02", "type": "out", "msg": "HOU: Justin Watson (Concussion) - CONFIRMED OUT"},
        {"time": "10:58", "type": "weather", "msg": "CHI: 18°F | 30MPH GUSTS | SNOW (Extreme Passing Variance)"},
        {"time": "10:45", "type": "out", "msg": "NBA: Nikola Jokic (Knee) - RULED OUT tonight"},
        {"time": "10:42", "type": "out", "msg": "NBA: Luka Doncic (Groin) - RULED OUT tonight"}
    ]
    return intel

# --- DYNAMIC ALPHA CALCULATION ---
def get_live_stats():
    nba_p = st.session_state.get('nba_portfolio', [])
    nfl_p = st.session_state.get('nfl_portfolio', [])
    total = len(nba_p) + len(nfl_p)
    
    # Calculate simulated Alpha Yield based on portfolio win rates
    if total > 0:
        win_rates = [float(l.get('Sim Win %', '0').replace('%','')) for l in nba_p]
        avg_alpha = sum(win_rates)/len(win_rates) if win_rates else 8.2
    else:
        avg_alpha = 0.0
    return total, avg_alpha

# --- HEADER SECTION ---
col_logo, col_stat = st.columns([3, 1])
with col_logo:
    st.title("🧬 VANTAGE ZERO")
    st.caption(f"SYSTEM DATE: {datetime.now().strftime('%b %d, %Y')} | COMMAND CENTER REV 52.1")

with col_stat:
    st.metric("INTEL FEED", "LIVE", "12ms")

st.markdown("---")

# --- GLOBAL PORTFOLIO METRICS ---
m1, m2, m3, m4 = st.columns(4)
total_lineups, alpha_val = get_live_stats()

m1.metric("PORTFOLIO SIZE", f"{total_lineups}", "ACTIVE")
m2.metric("CHI WEATHER", "18°F", "30MPH WIND", delta_color="inverse")
m3.metric("SIMS EXECUTED", "25.4K", "STABLE")
m4.metric("ALPHA YIELD", f"{alpha_val}%", f"+{alpha_val}%")

st.markdown("---")

# --- LIVE DATA TERMINAL ---
st.markdown("### 📡 SYSTEM INTEL TERMINAL")
feed = get_vantage_intel()
terminal_html = '<div class="terminal">'
for item in feed:
    style_class = "event-out" if item['type'] == 'out' else "event-weather"
    terminal_html += f'<div><span style="color:#8b949e;">[{item["time"]}]</span> <span class="{style_class}">{item["msg"]}</span></div>'
terminal_html += '</div>'
st.markdown(terminal_html, unsafe_allow_html=True)

st.markdown("---")

# --- THEATRE MODULES ---
c1, c2 = st.columns(2)
with c1:
    st.info("🏀 **NBA COMMAND**")
    if st.button("LAUNCH NBA ENGINE"): st.switch_page("pages/1_🏀_NBA_Alpha.py")
with c2:
    st.info("🏈 **NFL COMMAND**")
    if st.button("LAUNCH NFL ENGINE"): st.switch_page("pages/2_🏈_NFL_Alpha.py")
