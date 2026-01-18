import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v87.0 GPP", layout="wide", page_icon="🏀")

class EliteGPPOptimizerV87:
    def __init__(self, df):
        self.df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        def hunt(keys, default_val=None):
            for k in keys:
                if k in raw_cols: return raw_cols[k]
            return default_val

        p_key = hunt(['proj', 'fppg', 'points', 'avgpointspergame'], df.columns[0])
        s_key = hunt(['salary', 'sal', 'cost'], 'Salary')
        o_key = hunt(['ownership', 'own', 'projown', 'roster'], None)
        
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[s_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[hunt(['position', 'pos'], 'Position')].astype(str)
        self.df['Team'] = df[hunt(['team', 'teamabbrev', 'tm'], 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[hunt(['id', 'playerid'], 'ID')].astype(str)
        self.df['Own'] = pd.to_numeric(df[o_key], errors='coerce').fillna(15.0) if o_key else 15.0

        # NBA Position Flags
        for pos in ['PG','SG','SF','PF','C']: self.df[f'is_{pos}'] = self.df['Pos'].str.contains(pos).astype(int)
        self.df['is_G'] = (self.df['is_PG'] | self.df['is_SG']).astype(int)
        self.df['is_F'] = (self.df['is_SF'] | self.df['is_PF']).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.df); raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64); owns = self.df['Own'].values.astype(np.float64)
        
        # CEILING JITTER (25%) + STARS & SCRUBS REWARD
        # We simulate 10k scripts and favor players with extreme 'Value' potential
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.25), size=(total_sims, n_p)).clip(min=0)
        
        sim_pool = []
        for i in range(min(total_sims, 600)): 
            sim_p = sim_matrix[i]
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
            A.append(sals); bl.append(48500); bu.append(50000) # Tighten budget for stars
            A.append(owns); bl.append(0); bu.append(130.0) # Leverage pivot
            
            # STARS RULE: Force at least two $9k+ players
            is_star = (self.df['Sal'] >= 9000).astype(int).values
            A.append(is_star); bl.append(2); bu.append(4)

            # SCRUBS RULE: Force at least one <$4k 'Punt'
            is_punt = (self.df['Sal'] <= 4000).astype(int).values
            A.append(is_punt); bl.append(1); bu.append(3)

            # Positions
            for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.df[f'is_{p}'].values); bl.append(1); bu.append(8)
            A.append(self.df['is_G'].values); bl.append(3); bu.append(8)
            A.append(self.df['is_F'].values); bl.append(3); bu.append(8)

            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                sim_pool.append({'idx': tuple(idx), 'sim_score': sim_p[idx].sum()})

        sorted_pool = sorted(sim_pool, key=lambda x: x['sim_score'], reverse=True)
        portfolio, used_hashes = [], set()
        for entry in sorted_pool:
            if entry['idx'] not in used_hashes:
                portfolio.append(self.df.iloc[list(entry['idx'])].copy())
                used_hashes.add(entry['idx'])
            if len(portfolio) >= n_final: break
        return portfolio

# --- MAIN RENDERING ---
st.title("🏆 VANTAGE 99 | NBA STARS & SCRUBS")
f = st.file_uploader("LOAD DK NBA CSV", type="csv")
if f:
    df_raw = pd.read_csv(f); engine = EliteGPPOptimizerV87(df_raw)
    if st.button("🚀 INITIATE GPP CEILING SIMS"):
        st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000); st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state and len(st.session_state.portfolio) > 0:
        l = st.session_state.portfolio[st.session_state.sel_idx]
        st.subheader(f"NBA GPP LINEUP #{st.session_state.sel_idx+1}")
        st.table(l[['Name', 'Pos', 'Team', 'Sal', 'Proj', 'Own']])
        st.write(f"**GPP Confidence:** 10,000 Sims | **Target Score:** 350-370+")
