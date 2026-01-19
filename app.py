import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re

# --- ELITE UI & MULTI-SPORT CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | MULTI-SPORT ALPHA", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] { background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d; border-radius: 12px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

class VantageUnifiedOptimizer:
    def __init__(self, df, sport="NBA"):
        self.df = df.copy()
        self.sport = sport
        self._clean_data()

    def _clean_data(self):
        # Auto-detect common CSV headers
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'avgpoints'], cols)], errors='coerce').fillna(5.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(15.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm'], cols)].astype(str)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def run_alpha_sims(self, n_sims=5000, own_cap=125):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values
        # GPP ADVANTAGE: Ownership Leverage Filter
        # Penalizes high-owned players in sims to find low-owned "Ceiling" outcomes
        leverage_matrix = np.random.normal(raw_p, raw_p * 0.22, size=(n_sims, n_p)).clip(min=0)
        leverage_matrix = leverage_matrix * (1 - (self.df['Own'].values / 250))

        # Constraints Logic
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8 if self.sport=="NBA" else 9); bu.append(8 if self.sport=="NBA" else 9)
        A.append(self.df['Sal'].values); bl.append(49200); bu.append(50000)
        A.append(self.df['Own'].values); bl.append(0); bu.append(own_cap)

        # Sport-Specific Roster Rules
        if self.sport == "NFL":
            for p in ['QB','RB','WR','TE','DST']:
                A.append((self.df['Pos'] == p).astype(int).values)
                if p == 'QB': bl.append(1); bu.append(1)
                elif p == 'RB': bl.append(2); bu.append(3)
                elif p == 'WR': bl.append(3); bu.append(4)
                else: bl.append(1); bu.append(1)
        else: # NBA
            for p in ['PG','SG','SF','PF','C']:
                A.append(self.df['Pos'].str.contains(p).astype(int).values)
                bl.append(1); bu.append(6)

        lineup_pool = {}
        with st.status(f"🔬 SIMULATING {self.sport} GAME SCRIPTS...", expanded=True) as status:
            for i in range(1000): # High-quality solve for top 1k scenarios
                res = milp(c=-leverage_matrix[i], constraints=LinearConstraint(np.vstack(A), bl, bu),
                           integrality=np.ones(n_p), bounds=Bounds(0, 1), options={'mip_rel_gap': 0.01})
                if res.success:
                    ids = tuple(sorted(np.where(res.x > 0.5)[0]))
                    lineup_pool[ids] = lineup_pool.get(ids, 0) + 1
        
        return sorted(lineup_pool.items(), key=lambda x: x[1], reverse=True)[:10]

# --- MAIN APP INTERFACE ---
st.sidebar.title("🛠️ CONTROL CENTER")
mode = st.sidebar.radio("CHOOSE SPORT", ["NBA", "NFL"])
own_threshold = st.sidebar.slider("MAX TEAM OWNERSHIP %", 80, 200, 130)

st.title(f"🏆 VANTAGE 99 | {mode} ALPHA LAB")
f = st.file_uploader("UPLOAD SALARY CSV (DraftKings Format)", type="csv")

if f:
    df_raw = pd.read_csv(f)
    engine = VantageUnifiedOptimizer(df_raw, sport=mode)
    
    if st.button(f"🚀 RUN {mode} GPP SIMULATIONS"):
        results = engine.run_alpha_sims(own_cap=own_threshold)
        st.subheader("🥇 TOP PROJECTED CEILING LINEUPS")
        for ids, freq in results:
            ldf = engine.df.iloc[list(ids)]
            with st.expander(f"LINEUP - Win Freq: {round(freq/10, 1)}% | Proj: {round(ldf['Proj'].sum(), 1)}"):
                st.table(ldf[['Pos', 'Name', 'Team', 'Sal', 'Proj', 'Own']])
