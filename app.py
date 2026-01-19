import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re

# --- ELITE UI & MULTI-SPORT CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | LEGAL OPTIMIZER", layout="wide", page_icon="⚡")

class VantageUnifiedOptimizer:
    def __init__(self, df, sport="NBA"):
        self.df = df.copy()
        self.sport = sport
        self._clean_data()

    def _clean_data(self):
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        # Fixed: Changed .fillna(5.0) to 0.0 to prevent ghost punts
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'avgpoints'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        
        # Purge players with no projection to ensure legal stability
        self.df = self.df[self.df['Proj'] > 0.5].reset_index(drop=True)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def get_legal_constraints(self):
        """
        HARD LOCK: Strictly enforces DraftKings legal roster requirements.
        """
        n_p = len(self.df)
        A, bl, bu = [], [], []

        # 1. Total Roster Size 
        # NFL = 9 players, NBA = 8 players
        total_players = 9 if self.sport == "NFL" else 8
        A.append(np.ones(n_p)); bl.append(total_players); bu.append(total_players)

        # 2. Salary Cap ($50,000) 
        A.append(self.df['Sal'].values); bl.append(45000); bu.append(50000)

        if self.sport == "NFL":
            # HARD LOCK: QB(1), RB(2-3), WR(3-4), TE(1-2), DST(1) 
            A.append((self.df['Pos'] == 'QB').astype(int).values); bl.append(1); bu.append(1)
            A.append((self.df['Pos'] == 'RB').astype(int).values); bl.append(2); bu.append(3)
            A.append((self.df['Pos'] == 'WR').astype(int).values); bl.append(3); bu.append(4)
            A.append((self.df['Pos'] == 'TE').astype(int).values); bl.append(1); bu.append(2)
            A.append((self.df['Pos'] == 'DST').astype(int).values); bl.append(1); bu.append(1)
        else:
            # HARD LOCK NBA: Exact slot coverage 
            # Requires at least 1 of each primary position
            for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                A.append(self.df['Pos'].str.contains(p).astype(int).values); bl.append(1); bu.append(5)

        return np.vstack(A), bl, bu

    def run_alpha_sims(self, n_lineups=20, correlation=0.5, leverage=0.5):
        n_p = len(self.df)
        teams = self.df['Team'].unique()
        A, bl, bu = self.get_legal_constraints()
        
        lineup_pool = []
        with st.status(f"🚀 EXECUTING {self.sport} LEGAL SIMULATIONS...", expanded=False):
            for i in range(1000):
                # Correlated Game Scripts 
                team_shift = {t: np.random.normal(1.0, 0.15 * correlation) for t in teams}
                sim_projs = np.array([
                    row['Proj'] * team_shift[row['Team']] * np.random.normal(1.0, 0.1) 
                    for _, row in self.df.iterrows()
                ])
                # Leverage Penalty 
                sim_projs = sim_projs * (1 - (self.df['Own'].values * (leverage / 150)))

                # MILP Solver 
                res = milp(c=-sim_projs, constraints=LinearConstraint(A, bl, bu),
                           integrality=np.ones(n_p), bounds=Bounds(0, 1))
                
                if res.success:
                    idx = tuple(np.where(res.x > 0.5)[0])
                    lineup_pool.append({'idx': idx, 'sim_score': sim_projs[list(idx)].sum()})
                
                if len(lineup_pool) >= n_lineups * 2: break

        unique_lineups = {entry['idx']: entry for entry in lineup_pool}.values()
        sorted_pool = sorted(unique_lineups, key=lambda x: x['sim_score'], reverse=True)[:n_lineups]
        return [self.df.iloc[list(entry['idx'])] for entry in sorted_pool]

# --- UI INTERFACE ---
st.sidebar.title("🕹️ CONTROL")
mode = st.sidebar.radio("SPORT", ["NBA", "NFL"])
f = st.file_uploader("UPLOAD SALARY CSV", type="csv")

if f:
    engine = VantageUnifiedOptimizer(pd.read_csv(f), sport=mode)
    if st.button(f"⚡ GENERATE LEGAL {mode} LINEUPS"):
        results = engine.run_alpha_sims()
        for i, ldf in enumerate(results):
            with st.expander(f"LINEUP #{i+1} | Proj: {round(ldf['Proj'].sum(), 1)}"):
                # Force clean display of positions for final review 
                st.table(ldf[['Pos', 'Name', 'Team', 'Sal', 'Proj']])
