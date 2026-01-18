import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v64.0", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stButton>button { width: 100%; border-radius: 4px; border: 1px solid #30363d; background: #161b22; color: #c9d1d9; text-align: left; }
    .stButton>button:hover { border-color: #00ffcc; color: #00ffcc; }
    .audit-val { font-family: 'JetBrains Mono'; font-weight: bold; color: #00ffcc; font-size: 1.1rem; }
    .late-swap-ready { border: 1px solid #00ffcc; color: #00ffcc; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

class EliteOptimizerV64:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        # JAN 18 GAME WINDOWS: Houston@NE (3 PM), Rams@Chicago (6:30 PM)
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
            
        # JAN 18 SCRUB (Inactive/Injury Blacklist)
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.5):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        late_mask = self.df['is_late'].values.astype(np.float64)
        
        # Weather-Adjusted Jitter (Chicago winds + cold)
        scale = np.clip(raw_p * 0.25, 0.01, None)
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            # Late-Swap Bias: Favor late-game players for the FLEX spot
            adj_p = raw_p + (late_mask * 0.05)
            sim_p = np.random.normal(adj_p, scale).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49200); bu.append(50000)
            
            # HARD CAP POSITIONS (Strict 1-2-3-1-1-1 Format)
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            # Exposure Governor
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

# --- MAIN RENDERER ---
st.title("🧬 VANTAGE 99 | v64.0 COMMAND")

f = st.file_uploader("LOAD DK DATASET", type="csv")
if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteOptimizerV64(df_raw)
    
    if 'portfolio' not in st.session_state:
        if st.button("🚀 INITIATE ASSEMBLY"):
            st.session_state.portfolio = engine.assemble(n=20)
            st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state:
        col_list, col_scout = st.columns([1, 1.8])

        with col_list:
            st.markdown("### 📋 PORTFOLIO INDEX")
            for i, l in enumerate(st.session_state.portfolio):
                # Check for Late-Swap Ready (Player from 6:30 PM game in lineup)
                has_late = l['is_late'].sum() > 0
                swap_tag = '<span class="late-swap-ready">SWAP</span>' if has_late else ""
                if st.button(f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            st.markdown(f"### 🔍 SCOUTING REPORT: LINEUP #{st.session_state.sel_idx+1}")
            
            # --- POSITIONAL AUDIT ---
            p_counts = l['Pos'].value_counts()
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.markdown(f"QB<br><span class='audit-val'>{p_counts.get('QB', 0)}/1</span>", unsafe_allow_html=True)
            a2.markdown(f"RB<br><span class='audit-val'>{p_counts.get('RB', 0)}/3</span>", unsafe_allow_html=True)
            a3.markdown(f"WR<br><span class='audit-val'>{p_counts.get('WR', 0)}/4</span>", unsafe_allow_html=True)
            a4.markdown(f"TE<br><span class='audit-val'>{p_counts.get('TE', 0)}/2</span>", unsafe_allow_html=True)
            a5.markdown(f"DST<br><span class='audit-val'>{p_counts.get('DST', 0)}/1</span>", unsafe_allow_html=True)
            
            # --- DK ENTRY ORDER SORT ---
            st.markdown("---")
            qb = l[l['Pos'] == 'QB'].iloc[0]
            rbs = l[l['Pos'] == 'RB'].sort_values('Sal', ascending=False)
            wrs = l[l['Pos'] == 'WR'].sort_values('Sal', ascending=False)
            te = l[l['Pos'] == 'TE'].sort_values('Sal', ascending=False).iloc[0]
            dst = l[l['Pos'] == 'DST'].iloc[0]
            
            core_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
            flex = l[~l['ID'].isin(core_ids)].iloc[0]
            
            display_order = [("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]), ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]), ("TE", te), ("FLEX", flex), ("DST", dst)]
            
            for label, p in display_order:
                st.write(f"**{label}** | {p['Name']} ({p['Team']}) — ${int(p['Sal'])}")
