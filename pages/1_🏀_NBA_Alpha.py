import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, linear_sum_assignment
import re
from datetime import datetime

st.set_page_config(page_title="VANTAGE ZERO | NBA ALPHA", layout="wide", page_icon="🏀")

# --- 1. LIVE VEGAS DATA (JAN 18, 2026) ---
# Calculations based on O/U and Spreads
VEGAS_NBA = {
    "ORL": {"ou": 229.5, "spread": -3.0}, # Implied: 116.25
    "MEM": {"ou": 229.5, "spread": 3.0},  # Implied: 113.25
    "CHI": {"ou": 222.5, "spread": -6.0}, # Implied: 114.25
    "BKN": {"ou": 222.5, "spread": 6.0},  # Implied: 108.25
    "SAC": {"ou": 228.5, "spread": -2.5}, # Implied: 115.5
    "POR": {"ou": 228.5, "spread": 2.5}   # Implied: 113.0
}
# 2026 Season Mean PPG
SEASON_MEANS = {"ORL": 116.3, "MEM": 114.8, "CHI": 117.3, "BKN": 108.9, "SAC": 110.6, "POR": 116.2}

class VantageZeroNBA:
    def __init__(self, df):
        self.df = df.copy()
        self.df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(10.0)
        self.df['Own'] = pd.to_numeric(df['Ownership'], errors='coerce') if 'Ownership' in df.columns else 15.0
        
        # --- 2. DYNAMIC MATCHUP WEIGHTING ---
        # Adjusts projections for high-scoring 'shootouts'
        def get_lift(row):
            t = row['TeamAbbrev']
            if t in VEGAS_NBA and t in SEASON_MEANS:
                implied = (VEGAS_NBA[t]['ou'] / 2) - (VEGAS_NBA[t]['spread'] / 2)
                return implied / SEASON_MEANS[t]
            return 1.0
        self.df['Proj'] = self.df.apply(lambda r: r['Proj'] * get_lift(r), axis=1)

        # Position Logic
        pos_list = ['PG', 'SG', 'SF', 'PF', 'C']
        for p in pos_list: self.df[f'is_{p}'] = self.df['Position'].str.contains(p).astype(int)
        self.df['is_G'] = self.df[['is_PG', 'is_SG']].max(axis=1)
        self.df['is_F'] = self.df[['is_SF', 'is_PF']].max(axis=1)

    def run_engine(self, sims=2000, own_cap=125, jitter=0.22):
        n_p = len(self.df)
        projs = self.df['Proj'].values.astype(float)
        sals = self.df['Salary'].values.astype(float)
        owns = self.df['Own'].values.astype(float)
        
        # --- 3. TOURNAMENT CONSTRAINTS ---
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8); bu.append(8) # Size
        A.append(sals); bl.append(49600); bu.append(50000) # Salary
        A.append(owns); bl.append(0); bu.append(own_cap)  # LEVERAGE CAP
        
        for c in ['is_PG', 'is_SG', 'is_SF', 'is_PF', 'is_C']:
            A.append(self.df[c].values); bl.append(1); bu.append(8)
        A.append(self.df['is_G'].values); bl.append(3); bu.append(8)
        A.append(self.df['is_F'].values); bl.append(3); bu.append(8)
        
        sim_matrix = np.random.normal(projs, projs * jitter, size=(sims, n_p)).clip(min=0)
        lineup_freq = {}

        for i in range(sims):
            res = milp(c=-sim_matrix[i], constraints=LinearConstraint(np.vstack(A), bl, bu),
                       integrality=np.ones(n_p), bounds=Bounds(0, 1), options={'mip_rel_gap': 0.05})
            if res.success:
                idx = tuple(sorted(np.where(res.x > 0.5)[0]))
                lineup_freq[idx] = lineup_freq.get(idx, 0) + 1
        
        # Format Results
        top_ids = sorted(lineup_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [{"Win %": f"{round((c/sims)*100, 2)}%", "Sal": self.df.iloc[list(ids)]['Salary'].sum(),
                 "Lineup": ", ".join(self.df.iloc[list(ids)]['Name'].tolist())} for ids, c in top_ids]

# --- UI INTERFACE ---
st.title("🧬 VANTAGE ZERO | NBA ALPHA")
f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")
if f:
    df_raw = pd.read_csv(f)
    for i, row in df_raw.head(10).iterrows():
        if 'Name' in row.values and 'Salary' in row.values:
            vz = VantageZeroNBA(df_raw.iloc[i+1:].reset_index(drop=True))
            if st.button("🚀 EXECUTE NBA SIMULATION"):
                st.table(pd.DataFrame(vz.run_engine()))
            break
