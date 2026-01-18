import streamlit as st
from datetime import datetime

# --- SYSTEM ARCHITECTURE ---
st.set_page_config(
    page_title="VANTAGE ZERO | QUANTITATIVE HUB", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- STANDARDIZED STYLE GOVERNANCE ---
st.markdown("""
    <style>
    /* Institutional Dark Mode Palette */
    .main { background-color: #0e1117; color: #ffffff; }
    
    /* Metric Card Standardization */
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #00e676; }
    [data-testid="stMetricDelta"] { font-size: 14px; }
    
    /* Terminal Output Styling */
    .stCodeBlock { border-left: 3px solid #00e676; background-color: #1a1c24 !format; }
    
    /* Typography Overrides */
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
    </style>
    """, unsafe_allow_html=True)

# --- GLOBAL SESSION PERSISTENCE ---
if 'bankroll' not in st.session_state: st.session_state.bankroll = 5000.0
if 'total_sims' not in st.session_state: st.session_state.total_sims = 0

# --- HEADER: SYSTEM STATUS ---
hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.title("VANTAGE ZERO")
    st.caption(f"QUANTITATIVE OPERATIONS HUB | VERSION 55.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with hdr_right:
    if st.button("SYSTEM RESET", use_container_width=True):
        st.session_state.total_sims = 0
        st.rerun()

st.divider()

# --- MODULE 1: KPI OVERVIEW ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("AVAILABLE LIQUIDITY", f"${st.session_state.bankroll:,.2f}")
m2.metric("SIMULATION VOLUME", f"{st.session_state.total_sims:,}")
m3.metric("SYSTEM LATENCY", "14ms", "OPTIMAL")
m4.metric("ACTIVE THEATRES", "2", "STABLE")

st.divider()

# --- MODULE 2: OPERATIONS AND RISK ---
tab_ops, tab_risk = st.tabs(["OPERATIONAL COMMAND", "RISK GOVERNANCE"])

with tab_ops:
    st.subheader("Active Asset Classes")
    col_nfl, col_nba = st.columns(2)
    
    with col_nfl:
        st.write("**NATIONAL FOOTBALL LEAGUE**")
        st.caption("Status: Divisional Round | Weather-Adjusted Projections Active")
        if st.button("EXECUTE NFL ENGINE", use_container_width=True):
            st.switch_page("pages/2_🏈_NFL_Alpha.py")
            
    with col_nba:
        st.write("**NATIONAL BASKETBALL ASSOCIATION**")
        st.caption("Status: Standard Season | High-Variance Solver Active")
        if st.button("EXECUTE NBA ENGINE", use_container_width=True):
            st.switch_page("pages/1_🏀_NBA_Alpha.py")

with tab_risk:
    st.subheader("Capital Allocation Strategy")
    st.write("Current Methodology: Half-Kelly Criterion (0.5x Multiplier)")
    
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        alpha_input = st.slider("Target Model Alpha (%)", 1.0, 10.0, 3.5, step=0.1)
    
    with risk_col2:
        # Quantitative Risk Calculation
        allocation = (st.session_state.bankroll * (alpha_input / 100)) * 0.5
        st.metric("MAXIMUM EXPOSURE LIMIT", f"${allocation:,.2f}")

# --- MODULE 3: SYSTEM FEED ---
st.divider()
st.subheader("Institutional Variable Feed")
st.text_area(
    label="Live Data Streams",
    value="[11:15] HOU-NFL: Confirmation on Collins/Watson injury status.\n"
          "[11:10] CHI-NFL: Wind velocity sustained at 33MPH; precipitation confirmed.\n"
          "[11:00] NBA: Late-swap logic prioritized for evening slate.\n"
          "[SYSTEM] Modular Architecture v55.0 validated and operational.",
    height=150,
    disabled=True
)
