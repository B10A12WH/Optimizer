import streamlit as st
import pandas as pd
import time

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE ZERO | HQ", layout="wide", page_icon="🧬")

# [Your existing CSS stays here...]

# --- DYNAMIC NEWS FEED LOGIC ---
def get_news_feed():
    # In a professional setup, this would be a requests.get() from an API
    news = [
        {"time": "14:22", "event": "🚨 LAR: Cooper Kupp (Ankle) upgraded to PROBABLE"},
        {"time": "14:15", "event": "📊 HOU@NE: Line moved from +1.5 to +2.5"},
        {"time": "13:58", "event": "❄️ CHI: Wind gusting 22MPH. Passing projections -4%"},
        {"time": "13:45", "event": "🏀 NBA: Luka Doncic (Heel) ruled OUT for tonight"},
    ]
    return news

# --- HEADER SECTION ---
col_logo, col_stat = st.columns([3, 1])
with col_logo:
    st.title("🧬 VANTAGE ZERO")
    st.caption("INSTITUTIONAL QUANTITATIVE DFS FRAMEWORK // REV 52.0")

with col_stat:
    st.metric("SERVER STATUS", "STABLE", "14ms", delta_color="normal")

st.markdown("---")

# --- GLOBAL PORTFOLIO INTELLIGENCE ---
st.markdown("### 📊 GLOBAL RISK MONITOR")
m1, m2, m3, m4 = st.columns(4)

# Calculate total exposure from session state
nba_count = len(st.session_state.get('nba_portfolio', []))
nfl_count = len(st.session_state.get('nfl_portfolio', []))

m1.metric("TOTAL LINEUPS", f"{nba_count + nfl_count}", "ACTIVE")
m2.metric("SLATE PHASE", "DIVISIONAL", "JAN 18")
m3.metric("SIMS RUN", "25.4K", "DAILY")
m4.metric("HUB LATENCY", "LOW", "OPTIMAL")

st.markdown("---")

# --- THE "VET-KILLER" NEWS TERMINAL ---
st.markdown("### 📡 LIVE INTEL FEED")
with st.container():
    # This styling creates a scrolling "terminal" look
    st.markdown("""
        <style>
        .terminal {
            background-color: #0d1117;
            border: 1px solid #00ffcc;
            padding: 15px;
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            max-height: 200px;
            overflow-y: auto;
        }
        .time { color: #8b949e; }
        .event { color: #00ffcc; margin-left: 10px; }
        </style>
    """, unsafe_allow_html=True)
    
    feed_html = '<div class="terminal">'
    for item in get_news_feed():
        feed_html += f'<div><span class="time">[{item["time"]}]</span><span class="event">{item["event"]}</span></div>'
    feed_html += '</div>'
    st.markdown(feed_html, unsafe_allow_html=True)

st.markdown("---")

# --- THEATRE NAVIGATION ---
st.markdown("### 🚀 LAUNCH SEQUENCES")
c1, c2 = st.columns(2)
with c1:
    if st.button("OPEN NBA COMMAND"): st.switch_page("pages/1_🏀_NBA_Alpha.py")
with c2:
    if st.button("OPEN NFL COMMAND"): st.switch_page("pages/2_🏈_NFL_Alpha.py")
