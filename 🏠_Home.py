import streamlit as st
from datetime import datetime

# --- SYSTEM CONFIG ---
st.set_page_config(page_title="VANTAGE ZERO | HQ", layout="wide", page_icon="🧬")

# Ensure all state variables exist so nothing "disappears"
for key, val in {'total_sims': 0, 'sim_speed': 0, 'bankroll': 5000.0}.items():
    if key not in st.session_state: st.session_state[key] = val

# --- UI BRANDING ---
st.title("🧬 VANTAGE ZERO")
st.caption(f"SYSTEM CLOCK: {datetime.now().strftime('%b %d, %Y')} | MULTI-THEATRE COMMAND")

# Global Stats Bar
m1, m2, m3 = st.columns(3)
m1.metric("BANKROLL", f"${st.session_state.bankroll:,.2f}")
m2.metric("SIMS", f"{st.session_state.total_sims/1000:.1f}K")
m3.metric("STATUS", "READY", "12ms")

st.markdown("---")

# --- DUAL THEATRE ACCESS ---
st.subheader("🚀 ACTIVE MISSIONS")
col_nba, col_nfl = st.columns(2)

with col_nba:
    st.info("🏀 **NBA ALPHA ENGINE**")
    st.write("Target: Ceiling simulations for Sunday night slate.")
    # This button programmatically switches the page
    if st.button("OPEN NBA COMMAND"):
        st.switch_page("pages/1_🏀_NBA_Alpha.py")

with col_nfl:
    st.info("🏈 **NFL ALPHA ENGINE**")
    st.write("Target: Divisional Round | Weather Adjusted [CHI: 11°F].")
    if st.button("OPEN NFL COMMAND"):
        st.switch_page("pages/2_🏈_NFL_Alpha.py")

# --- SYSTEM TERMINAL ---
st.markdown("---")
st.subheader("📡 SYSTEM FEED")
st.code(f"""
[LOG] NBA Engine Standby...
[LOG] NFL Engine Optimized for Chicago Wind (33MPH)...
[LOG] Session State Persistent across {len(st.session_state)} variables...
""", language="bash")
