import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v71.0 GPP", layout="wide", page_icon="🏆")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stButton>button { width: 100%; border-radius: 4px; border: 1px solid #30363d; background: #161b22; color: #c9d1d9; text-align: left; }
    .stButton>button:hover { border-color: #00ffcc; color: #00ffcc; }
    .audit-val { font-family: 'JetBrains Mono'; font-weight: bold; color: #00ffcc; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

class EliteGPPOptimizerV71:
    def __init__(self, df):
        # 1. ROBUST COLUMN DETECTION
        cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        self.df = df.copy()
        
        # Search for core columns using multiple common naming conventions
        p_key = next((cols[k] for k in cols if k in ['proj', 'fppg', 'points', 'avgpointspergame']), df.columns[0])
        sal_key = next((cols[k] for k in cols if k in ['salary', 'sal', 'cost']), 'Salary')
        team_key = next((cols[k] for k in cols if k in ['team', 'teamabbrev', 'tm']), 'TeamAbbrev')
        own_key = next((cols[k] for k in cols if k in ['ownership', 'own', 'projown', 'roster', 'roster%']), None)
        
        # 2. DATA CLEANING & SAFE MAPPING
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[sal_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[team_key].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        # FIXED: Safe Ownership Assignment (Avoids v70.0 Traceback)
        if own_key:
            self.df['Own'] = pd.to_numeric(df[own_key], errors='coerce').fillna(15.0)
        else:
            self.df['Own'] = 15.0  # Assigns scalar 15 to entire column safely

        # Matchup Weighting (Vegas Shootout Targets)
        vegas_boosts = {'HOU': 0.88, 'NE': 1.08, 'LAR': 1.15, 'CHI': 1.04}
        self.df['Proj'] = self.df.apply(lambda x: x['Proj'] * vegas_boosts.get(x['Team'], 1.0), axis=1)
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
            
        # JAN 18 SCRUB (Inactives Filter)
        inactives = ['Nico Collins', 'Justin Watson', 'Mack Hollins', 'Nick McCloud', 'Trent Brown']
        self.df = self.df[~self.df['Name'].isin(inactives)].reset_index(drop=True)

    def assemble(self, n=20, exp=0.40, diversity=5):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        owns = self.df['Own'].values.astype(np.float64)
        late_mask = self.df['is_late'].values.astype(np.float64)
        is_te = self.df['is_TE'].values
        
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        used_lineups = []

        for i in range(n):
            # SIMULATION CORE: 22% Jitter for Ceiling outcomes
            adj_p = (raw_p + (late_mask * 0.20) - (is_te * 0.95)).flatten()
            sim_p = np.random.normal(loc=adj_p, scale=np.abs(adj_p * 0.22), size=adj_p.shape).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) 
            A.append(sals); bl.append(49000); bu.append(50000)
            
            # GPP LEVERAGE: Ownership Cap
            A.append(owns); bl.append(0); bu.append(135.0)
            
            # Positional Constraints
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            # QB STACKING: QB + WR/TE Correlation
            for team in self.df['Team'].unique():
                q_idx = self.df[(self.df['Team'] == team) & (self.df['is_QB'] == 1)].index.tolist()
                s_idx = self.df[(self.df['Team'] == team) & ((self.df['is_WR'] == 1) | (self.df['is_TE'] == 1))].index.tolist()
                if q_idx and s_idx:
                    row = np.zeros(n_p); row[s_idx] = 1; row[q_idx] = -1
                    A.append(row); bl.append(0); bu.append(8)

            # Diversity Rule: Minimum 4 unique players per lineup
            for past_idx in used_lineups:
                row = np.zeros(n_p); row[list(past_idx)] = 1
                A.append(row); bl.append(0); bu.append(diversity)

            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                portfolio.append(self.df.iloc[idx].copy())
                used_lineups.append(tuple(idx))
                for name in portfolio[-1]['Name']: counts[name] += 1
        return portfolio

# --- MAIN RENDERING ---
st.title("🏆 VANTAGE 99 | v71.0 GPP COMMAND")
f = st.file_uploader("LOAD DK DATASET", type="csv")
if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteGPPOptimizerV71(df_raw)
    
    if st.button("🚀 INITIATE GPP ASSEMBLY"):
        st.session_state.portfolio = engine.assemble(n=20)
        st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state:
        l = st.session_state.portfolio[st.session_state.sel_idx]
        qb = l[l['is_QB']==1].iloc[0]; dst = l[l['is_DST']==1].iloc[0]; te = l[l['is_TE']==1].iloc[0]
        rbs = l[l['is_RB']==1].sort_values('Sal', ascending=False)
        wrs = l[l['is_WR']==1].sort_values('Sal', ascending=False)
        core_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
        flex = l[~l['ID'].isin(core_ids)].iloc[0]
        
        display_order = [("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]), ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]), ("TE", te), ("FLEX", flex), ("DST", dst)]
        
        st.subheader(f"LINEUP #{st.session_state.sel_idx+1} (GPP CEILING)")
        st.table(pd.DataFrame([{"Pos": lbl, "Name": p['Name'], "Team": p['Team'], "Salary": f"${int(p['Sal'])}", "Proj": round(p['Proj'], 2)} for lbl, p in display_order]))
        st.write(f"**Total Ownership:** {round(l['Own'].sum(), 1)}% | **GPP Strategy:** ACTIVE")
