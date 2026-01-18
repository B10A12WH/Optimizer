import streamlit as st
from datetime import datetime

# --- SYSTEM CONFIG (LONG-TERM SCALING) ---
st.set_page_config(page_title="VANTAGE ZERO | HQ", layout="wide", page_icon="🧬")

# Core Memory Persistence
for key, val in {'total_sims': 0, 'sim_speed': 0, 'bankroll': 5000.0}.items():
    if key not in st.session_state: st.session_state[key] = val

# --- DYNAMIC INTEL ENGINE ---
def get_live_intel():
    # This remains a dynamic "Variable Layer"
    return [
        {"type": "out", "msg": "HOU: Collins & Watson - OUT [Jan 18]"},
        {"type": "weather", "msg": "CHI: 11°F | 33MPH GUSTS (High Velocity Warning)"},
        {"type": "sys", "msg": "System Integrity: Modular v54.0 Active"}
    ]

# --- UI ARCHITECTURE ---
st.title("🧬 VANTAGE ZERO")
st.caption(f"SYSTEM CLOCK: {datetime.now().strftime('%b %d, %Y')} | AGNOSTIC QUANT HUB")

# Global Metrics
c1, c2, c3 = st.columns(3)
c1.metric("BANKROLL SHIELD", f"${st.session_state.bankroll:,.2f}")
c2.metric("SIMULATION VOLUME", f"{st.session_state.total_sims/1000:.1f}K", "TOTAL")
c3.metric("ENGINE LATENCY", "12ms", "OPTIMAL")

st.markdown("---")

# Intelligent Command Center
col_term, col_risk = st.columns([2, 1])

with col_term:
    st.subheader("📡 Variable Feed")
    for item in get_live_intel():
        color = "#ff4b4b" if item['type'] == 'out' else "#ffcc00"
        st.markdown(f"<span style='color:{color};'>●</span> {item['msg']}", unsafe_allow_html=True)

with col_risk:
    st.subheader("🛡️ Risk Governance")
    # Implements half-kelly for long-term bankroll preservation
    edge = st.slider("Model Alpha (%)", 1.0, 10.0, 3.5)
    risk_amt = (st.session_state.bankroll * (edge / 100)) * 0.5 
    st.info(f"Recommended Limit: ${risk_amt:,.2f}")

st.markdown("---")
# Theatre Operations (Static links to your modular engines)
if st.button("LAUNCH NFL ENGINE"): st.switch_page("pages/2_🏈_NFL_Alpha.py")
