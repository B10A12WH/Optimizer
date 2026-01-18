import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, linear_sum_assignment
import re
from datetime import datetime

# --- VANTAGE ZERO: NBA CORE REBUILD (V50.0) ---
st.set_page_config(page_title="VANTAGE ZERO | NBA", layout="wide", page_icon="🏀")

class VantageZeroNBA:
    def __init__(self, df):
        # 1. CLEANING & VECTORIZATION
        self.df = df.copy()
        self.df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(5.0)
        self.df['Own'] = pd.to_numeric(df['Ownership'], errors='coerce') if 'Ownership' in df.columns else 15.0
        
        # 2. POSITION MATRIX (The Foundation)
        pos_list = ['PG', 'SG', 'SF', 'PF', 'C']
        for p in pos_list:
            self.df[f'is_{p}'] = self.df['Position'].str.contains(p).astype(int)
        self.df['is_G'] = self.df[['is_PG', 'is_SG']].max(axis=1)
        self.df['is_F'] = self.df[['is_SF', 'is_PF']].max(axis=1)
        
        # 3. CHRONO-LOGIC (For Late Swap)
        def parse_dk_time(val):
            m = re.search(r'(\d{1,2}:\d{2}[APM]+)', str(val))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        self.df['Time'] = self.df['Game Info'].apply(parse_dk_time)
        
        # 4. INSTITUTIONAL SCRUB (Jan 18 Status)
        # Scrubbing high-profile scratches to save sim time
        out_list = ['Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jayson Tatum']
        self.df = self.df[~self.df['Name'].isin(out_list)].reset_index(drop=True)

    def run_engine(self, sims=5000, own_cap=125, jitter=0.22):
        n_p = len(self.df)
        projs = self.df['Proj'].values.astype(float)
        sals = self.df['Salary'].values.astype(float)
        owns = self.df['Own'].values.astype(float)
        
        # PRE-BUILD GLOBAL CONSTRAINTS (Matrix form)
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8); bu.append(8) # Roster Size
        A.append(sals); bl.append(49700); bu.append(50000) # Salary Floor
        A.append(owns); bl.append(0); bu.append(own_cap)  # Leverage Cap
        
        # Positional Minimums
        for c in ['is_PG', 'is_SG', 'is_SF', 'is_PF', 'is_C']:
            A.append(self.df[c].values); bl.append(1); bu.append(8)
        A.append(self.df['is_G'].values); bl.append(3); bu.append(8)
        A.append(self.df['is_F'].values); bl.append(3); bu.append(8)
        A_stack = np.vstack(A)

        lineup_freq = {}
        
        # VECTORIZED JITTER (Pre-calculate all 5,000 game scripts)
        # Using 1.5x Multiplier on Studs ($9k+) to test their floor
        dyn_jitter = np.where(sals >= 9000, jitter * 1.5, jitter)
        sim_matrix = np.random.normal(projs, projs * dyn_jitter, size=(sims, n_p)).clip(min=0)

        status = st.status("🧬 VANTAGE ZERO: SIMULATING OUTCOMES...", expanded=True)
        for i in range(sims):
            res = milp(c=-sim_matrix[i], constraints=LinearConstraint(A_stack, bl, bu),
                       integrality=np.ones(n_p), bounds=Bounds(0, 1),
                       options={'mip_rel_gap': 0.05, 'presolve': True})
            
            if res.success:
                idx = tuple(sorted(np.where(res.x > 0.5)[0]))
                lineup_freq[idx] = lineup_freq.get(idx, 0) + 1
        
        status.update(label="✅ SIMULATION COMPLETE", state="complete")
        
        # MOLECULAR SLOT ASSIGNMENT (The Hungarian Algorithm)
        top_ids = sorted(lineup_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        results = []
        for ids, count in top_ids:
            ldf = self.df.iloc[list(ids)].copy()
            slots = ['PG','SG','SF','PF','C','G','F','UTIL']
            # Logic: Min-cost to put late players in late slots
            cost = np.full((8, 8), 100.0)
            for pi, (_, p) in enumerate(ldf.iterrows()):
                for si, slot_pos in enumerate(['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']):
                    if p[slot_pos] == 1: cost[pi, si] = 10
                # UTIL logic: Latest game time = 0 cost
                cost[pi, 7] = 0 if p['Time'] == ldf['Time'].max() else 50
            
            rows, cols = linear_sum_assignment(cost)
            lineup_dict = {"Sim Win %": f"{round((count/sims)*100, 2)}%", "Salary": ldf['Salary'].sum()}
            for r, c in zip(rows, cols): lineup_dict[slots[c]] = ldf.iloc[r]['Name']
            results.append(lineup_dict)
            
        return results

# --- INTERFACE ---
st.title("🧬 VANTAGE ZERO | NBA ENGINE")
f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")
if f:
    df_raw = pd.read_csv(f)
    # Detect DK Header
    for i, row in df_raw.head(10).iterrows():
        if 'Name' in row.values and 'Salary' in row.values:
            df_final = df_raw.iloc[i+1:].copy()
            df_final.columns = df_raw.iloc[i].values
            break
    
    vz = VantageZeroNBA(df_final.reset_index(drop=True))
    if st.button("🚀 EXECUTE ALPHA SIMULATION"):
        final_lineups = vz.run_engine()
        st.table(pd.DataFrame(final_lineups))
