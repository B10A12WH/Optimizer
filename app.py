import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import plotly.graph_objects as go
import time

# --- STYLING: HIGH-END SAAS AESTHETIC ---
st.set_page_config(page_title="VANTAGE 99 | ELITE", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* Master-Detail Split */
    .stHorizontalBlock { gap: 2rem; }
    
    /* Neon Glassmorphism Cards */
    .lineup-button {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .lineup-button:hover { border-color: #00ffcc; background: #1c2128; box-shadow: 0 0 10px rgba(0, 255, 204, 0.2); }
    
    /* Position Badges */
    .pos-tag { 
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 10px;
        color: #0b0e14;
        background: #00ffcc;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

class EliteOptimizer:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        
        # Header Agnostic Parsing
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        # Binary Position Matrix
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
        
        # Divisional Blacklist (Jan 18, 2026)
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.5, jitter=0.22):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        scale = np.clip(raw_p * jitter, 0.01, None)
        
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            sim_p = np.random.normal(raw_p, scale).clip(min=0)
            A, bl, bu = [], [], []
            
            # 1. Total Roster Count
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            # 2. Salary Cap
            A.append(sals); bl.append(49200); bu.append(50000)
            
            # 3. DRAFTKINGS CLASSIC CONSTRAINTS (Includes FLEX Logic)
            # Position | Min | Max (Allows 1 Flex spot for RB/WR/TE)
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            # 4. Exposure Governor
            for idx, name in enumerate(self.df['Name']):
                if counts[name] >= (n * exp):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                lineup = self.df.iloc[idx].copy()
                portfolio.append(lineup)
                for name in lineup['Name']: counts[name] += 1
        return portfolio

# --- DASHBOARD LAYOUT ---
st.title("🧪 VANTAGE 99 | ELITE COMMAND")
f = st.file_uploader("UPLOAD DATASET", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteOptimizer(df_raw)
    
    if 'portfolio' not in st.session_state:
        if st.button("🚀 INITIATE PORTFOLIO REBALANCING"):
            st.session_state.portfolio = engine.assemble(n=20)
            st.session_state.selected_idx = 0

    if 'portfolio' in st.session_state:
        col_nav, col_main = st.columns([1, 2])
        
        with col_nav:
            st.markdown('<p class="section-header">Portfolio Overview</p>', unsafe_allow_html=True)
            for i, l in enumerate(st.session_state.portfolio):
                score = round(l['Proj'].sum(), 1)
                if st.button(f"L{i+1} | {score} PTS", key=f"btn_{i}"):
                    st.session_state.selected_idx = i
        
        with col_main:
            sel_l = st.session_state.portfolio[st.session_state.selected_idx]
            st.markdown(f'<p class="section-header">Scouting Report: Lineup #{st.session_state.selected_idx + 1}</p>', unsafe_allow_html=True)
            
            # --- OFFICIAL DK ORDERING LOGIC ---
            qb = sel_l[sel_l['Pos'] == 'QB'].iloc[0]
            rbs = sel_l[sel_l['Pos'] == 'RB'].sort_values('Sal', ascending=False)
            wrs = sel_l[sel_l['Pos'] == 'WR'].sort_values('Sal', ascending=False)
            te = sel_l[sel_l['Pos'] == 'TE'].sort_values('Sal', ascending=False).iloc[0]
            dst = sel_l[sel_l['Pos'] == 'DST'].iloc[0]
            
            # Identify FLEX (The one player not in core starters)
            core_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
            flex = sel_l[~sel_l['ID'].isin(core_ids)].iloc[0]
            
            roster_order = [
                ("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]), 
                ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]), 
                ("TE", te), ("FLEX", flex), ("DST", dst)
            ]
            
            # Render Modern List
            for label, p in roster_order:
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px; border-bottom: 1px solid #30363d;">
                    <span><span class="pos-tag">{label}</span> <b>{p['Name']}</b></span>
                    <span style="color: #8b949e;">{p['Team']} • ${int(p['Sal'])}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # --- METRIC AUDIT PANEL ---
            st.markdown('<p class="section-header" style="margin-top:25px;">Strategic Audit</p>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("SALARY UTIL", f"${int(sel_l['Sal'].sum())}", f"{int(50000 - sel_l['Sal'].sum())} Left")
            m2.metric("PROJ SCORE", round(sel_l['Proj'].sum(), 2))
            m3.metric("GAME DENSITY", f"{len(sel_l['Team'].unique())} Teams")
