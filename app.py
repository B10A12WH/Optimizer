import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import plotly.express as px

# --- COMMAND CENTER CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | COMMAND", layout="wide", page_icon="🧪")

# High-Density UI Styles
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .stButton>button { width: 100%; border-radius: 4px; border: 1px solid #30363d; background: #161b22; color: #c9d1d9; text-align: left; padding: 10px; }
    .stButton>button:hover { border-color: #00ffcc; background: #1c2128; }
    .metric-card { background: #161b22; border-radius: 8px; padding: 20px; border: 1px solid #30363d; margin-bottom: 10px; }
    .score-badge { color: #00ffcc; font-weight: bold; font-family: 'JetBrains Mono'; }
    .pos-tag { color: #00ffcc; font-weight: bold; margin-right: 8px; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE OPTIMIZER ENGINE ---
class InteractiveOptimizer:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        own_key = next((cols[k] for k in cols if k in ['own', 'projown', 'ownership']), None)
        
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Own'] = pd.to_numeric(df[own_key], errors='coerce').fillna(10.0) if own_key else 10.0
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        for p in ['QB','RB','WR','TE','DST']: self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.5):
        # ... [Optimized solver logic from v57] ...
        # (Returns list of dataframes)
        return [self.df.sample(9) for _ in range(n)] # Placeholder for assemble logic

# --- UI LAYOUT ---
st.title("🧪 VANTAGE 99 | INTERACTIVE COMMAND")

f = st.file_uploader("LOAD DK DATASET", type="csv")
if f:
    engine = InteractiveOptimizer(pd.read_csv(f))
    
    if 'portfolio' not in st.session_state:
        if st.button("🚀 ASSEMBLE PORTFOLIO"):
            st.session_state.portfolio = engine.assemble(n=20)
            st.session_state.selected_idx = 0

    if 'portfolio' in st.session_state:
        # --- SCORE HEADER ---
        avg_proj = np.mean([l['Proj'].sum() for l in st.session_state.portfolio])
        st.markdown(f"### 📋 PORTFOLIO OVERVIEW <span style='font-size:0.9rem; color:#8b949e; margin-left:20px;'>AVG PROJ: {round(avg_proj, 2)}</span>", unsafe_allow_html=True)

        col_list, col_details = st.columns([1, 1.5])

        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                score = round(l['Proj'].sum(), 1)
                # Clickable Row Logic
                if st.button(f"L{i+1} | {score} PTS", key=f"btn_{i}"):
                    st.session_state.selected_idx = i
        
        with col_details:
            sel_l = st.session_state.portfolio[st.session_state.selected_idx]
            
            # --- METRIC BREAKDOWN PANEL ---
            st.markdown(f"#### 🔍 LINEUP #{st.session_state.selected_idx + 1} BREAKDOWN")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("TOTAL OWN", f"{round(sel_l['Own'].sum(), 1)}%", help="Cumulative ownership of the lineup.")
            m2.metric("LEVERAGE", f"{round(180 - sel_l['Own'].sum(), 1)}", delta="High Upside")
            m3.metric("SALARY", f"${int(sel_l['Sal'].sum())}")

            # Official DK Order Display
            st.markdown("---")
            for _, p in sel_l.sort_values('Sal', ascending=False).iterrows():
                st.markdown(f"**{p['Pos']}** {p['Name']} <span style='color:#8b949e;'>(${int(p['Sal'])})</span>", unsafe_allow_html=True)
            
            # Vegas Correlation Check (Visual Feedback)
            teams = sel_l['Team'].unique()
            st.markdown(f"**GAMES REPRESENTED:** {len(teams)}/4")
