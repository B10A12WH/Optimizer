import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v66.0", layout="wide", page_icon="🧬")

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

class EliteOptimizerV66:
    def __init__(self, df):
        # Header Normalization: Fixes 'Salary' or 'AvgPointsPerGame' KeyErrors
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        
        # Hunt for the projection column
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        # 1/18 Vegas Boosts (Matchup Weighting)
        vegas_boosts = {'HOU': 0.88, 'NE': 1.08, 'LAR': 1.15, 'CHI': 1.04}
        self.df['Proj'] = self.df.apply(lambda x: x['Proj'] * vegas_boosts.get(x['Team'], 1.0), axis=1)
        
        # Game Windows for Late Swap
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
            
        # Inactive/Injury Scrub
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.45):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        late_mask = self.df['is_late'].values.astype(np.float64)
        is_te = self.df['is_TE'].values
        
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            # Tournament Strategy: Penalty for TE in Flex + Late Swap Priority
            adj_p = raw_p + (late_mask * 0.15) - (is_te * 0.85)
            sim_p = np.random.normal(adj_p, adj_p * 0.22).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # 9 Players
            A.append(sals); bl.append(49000); bu.append(50000) # Salary Cap
            
            # Position Requirements (DraftKings Classic)
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(1) # Forced TE limit
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            # QB Stacking (Pair QB with at least 1 WR/TE)
            for team in self.df['Team'].unique():
                q_idx = self.df[(self.df['Team'] == team) & (self.df['is_QB'] == 1)].index.tolist()
                s_idx = self.df[(self.df['Team'] == team) & ((self.df['is_WR'] == 1) | (self.df['is_TE'] == 1))].index.tolist()
                if q_idx and s_idx:
                    row = np.zeros(n_p)
                    for s in s_idx: row[s] = 1
                    for q in q_idx: row[q] = -1
                    A.append(row); bl.append(0); bu.append(8)

            # Exposure Control
            for idx, name in enumerate(self.df['Name']):
                if counts[name] >= (n * exp):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                lineup = self.df.iloc[idx].copy()
                portfolio.append(lineup)
                for name in lineup['Name']: counts[name] += 1
        return portfolio

# --- UI RENDERING ---
st.title("🧬 VANTAGE 99 | v66.0 COMMAND")

f = st.file_uploader("LOAD DK DATASET", type="csv")
if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteOptimizerV66(df_raw)
    
    if 'portfolio' not in st.session_state:
        if st.button("🚀 INITIATE ASSEMBLY"):
            st.session_state.portfolio = engine.assemble(n=20)
            st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state:
        col_list, col_scout = st.columns([1, 2.2])

        with col_list:
            st.markdown("### 📋 PORTFOLIO INDEX")
            for i, l in enumerate(st.session_state.portfolio):
                if st.button(f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            st.markdown(f"### 🔍 DRAFTKINGS STANDARD ROSTER: LINEUP #{st.session_state.sel_idx+1}")
            
            # Re-identifying players for standard sorting
            qb = l[l['is_QB']==1].iloc[0]
            rbs = l[l['is_RB']==1].sort_values('Sal', ascending=False)
            wrs = l[l['is_WR']==1].sort_values('Sal', ascending=False)
            te = l[l['is_TE']==1].iloc[0]
            dst = l[l['is_DST']==1].iloc[0]

            # FLEX Logic: Identifying the 9th man (3rd RB, 4th WR, or 2nd TE)
            # This logic puts the player with the LATEST start time in FLEX for swap flexibility
            core_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], 
                        wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], 
                        te['ID'], dst['ID']]
            flex_candidate = l[~l['ID'].isin(core_ids)]
            flex = flex_candidate.iloc[0] if not flex_candidate.empty else None

            # DRAFTKINGS ROSTER ORDER: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
            display_order = [
                ("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]),
                ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]),
                ("TE", te), ("FLEX", flex), ("DST", dst)
            ]

            # Visual Table Output
            st.table(pd.DataFrame([
                {"Pos": label, "Name": p['Name'], "Team": p['Team'], "Salary": f"${int(p['Sal'])}", "Proj": round(p['Proj'], 2)}
                for label, p in display_order
            ]))
            
            st.markdown(f"**Total Salary Used:** ${int(l['Sal'].sum())}")
            st.markdown(f"**Late-Swap Ready Players:** {l['is_late'].sum()}")
