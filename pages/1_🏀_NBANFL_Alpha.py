import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- PLATFORM CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | MULTI-SPORT", layout="wide", page_icon="🚀")

class VantageOptimizer:
    def __init__(self, df, sport="NFL"):
        self.df = df.copy()
        self.sport = sport
        self._standardize_data()
        
    def _standardize_data(self):
        # Universal Column Hunting
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg'], cols)], errors='coerce').fillna(0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(10)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm'], cols)].astype(str)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def get_constraints(self, n_p):
        A, bl, bu = [], [], []
        sals = self.df['Sal'].values
        
        # Total Players & Salary
        A.append(np.ones(n_p)); bl.append(8 if self.sport=="NBA" else 9); bu.append(8 if self.sport=="NBA" else 9)
        A.append(sals); bl.append(49000); bu.append(50000)

        if self.sport == "NFL":
            for p in ['QB', 'RB', 'WR', 'TE', 'DST']:
                mask = (self.df['Pos'] == p).astype(int).values
                if p == 'QB': bl.append(1); bu.append(1)
                if p == 'RB': bl.append(2); bu.append(3)
                if p == 'WR': bl.append(3); bu.append(4)
                if p == 'TE': bl.append(1); bu.append(2)
                if p == 'DST': bl.append(1); bu.append(1)
                A.append(mask)
        else: # NBA DraftKings Rules
            for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                mask = self.df['Pos'].str.contains(p).astype(int).values
                bl.append(1); bu.append(4)
                A.append(mask)
        
        return np.vstack(A), bl, bu

    def run_sim_build(self, n_lineups=10, iterations=10000):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values
        # GPP IMPROVEMENT: We use a higher std dev (0.25) for NBA to account for "Blowout" or "Hot Hand" variance
        variance = 0.25 if self.sport == "NBA" else 0.22
        sim_matrix = np.random.normal(loc=raw_p, scale=raw_p * variance, size=(iterations, n_p)).clip(min=0)
        
        # Ownership Leverage: Penalize "Chalk" slightly in sims to find pivots
        sim_matrix = sim_matrix * (1 - (self.df['Own'].values / 200))

        pool = []
        A, bl, bu = self.get_constraints(n_p)
        
        # We solve for the top 1000 unique "Game Scripts" to ensure speed + variety
        for i in range(1000):
            res = milp(c=-sim_matrix[i], constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                pool.append({'idx': tuple(idx), 'score': raw_p[idx].sum()})
        
        return [self.df.iloc[list(entry['idx'])] for entry in sorted(pool, key=lambda x: x['score'], reverse=True)[:n_lineups]]

# --- UI INTERFACE ---
st.sidebar.title("🎮 CONTROL CENTER")
sport_mode = st.sidebar.radio("SELECT SPORT", ["NFL", "NBA"])
sim_count = st.sidebar.slider("SIMULATIONS", 1000, 10000, 10000)

st.title(f"🏆 VANTAGE 99 | {sport_mode} GPP ENGINE")
file = st.file_uploader(f"UPLOAD {sport_mode} CSV", type="csv")

if file:
    data = pd.read_csv(file)
    engine = VantageOptimizer(data, sport=sport_mode)
    
    if st.button(f"🚀 GENERATE {sport_mode} GPP LINEUPS"):
        with st.spinner(f"Simulating {sim_count} Game Scripts..."):
            st.session_state.lineups = engine.run_sim_build(n_lineups=15, iterations=sim_count)
            st.success("GPP Strategy Applied: High-Ownership Tax & Variance Modeling Complete.")

    if 'lineups' in st.session_state:
        for i, df_lineup in enumerate(st.session_state.lineups):
            with st.expander(f"LINEUP {i+1} | Proj: {round(df_lineup['Proj'].sum(), 2)}"):
                st.table(df_lineup[['Pos', 'Name', 'Team', 'Sal', 'Proj', 'Own']])
