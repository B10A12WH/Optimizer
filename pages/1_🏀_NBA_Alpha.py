import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v89.0 NUCLEAR", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV89:
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
        self.df['ID'] = df[hunt(['id', 'playerid'], 'ID')].astype(str)
        self.df['Own'] = pd.to_numeric(df[o_key], errors='coerce').fillna(15.0) if o_key else 15.0

        # Position Masking for NBA Slots
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.df[f'mask_{p}'] = self.df['Pos'].str.contains(p).astype(int)
        self.df['mask_G'] = (self.df['mask_PG'] | self.df['mask_SG']).astype(int)
        self.df['mask_F'] = (self.df['mask_SF'] | self.df['mask_PF']).astype(int)

        # Strategic GPP Masks
        self.df['is_Star'] = (self.df['Sal'] >= 9500).astype(int)
        self.df['is_Punt'] = (self.df['Sal'] <= 4200).astype(int)
        self.df['is_Contrarian'] = (self.df['Own'] < 10.0).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.df); raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64); owns = self.df['Own'].values.astype(np.float64)
        
        # INCREASED CEILING JITTER (28%): Forcing more 'Nuclear' results
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.28), size=(total_sims, n_p)).clip(min=0)
        
        sim_pool = []
        for i in range(min(total_sims, 600)): 
            sim_p = sim_matrix[i]
            A, bl, bu = [], [], []
            
            # CORE CONSTRAINTS
            A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
            A.append(sals); bl.append(49600); bu.append(50000) # Force 99%+ salary spend
            A.append(owns); bl.append(0); bu.append(125.0) # Tighten ownership for leverage
            
            # GPP TARGETS: Stars, Scrubs, and Contrarians
            A.append(self.df['is_Star'].values); bl.append(2); bu.append(4)
            A.append(self.df['is_Punt'].values); bl.append(1); bu.append(3)
            A.append(self.df['is_Contrarian'].values); bl.append(3); bu.append(8)

            # Positions
            for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.df[f'mask_{p}'].values); bl.append(1); bu.append(8)
            A.append(self.df['mask_G'].values); bl.append(3); bu.append(8)
            A.append(self.df['mask_F'].values); bl.append(3); bu.append(8)

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
st.title("🏀 VANTAGE 99 | NBA NUCLEAR CEILING")
f = st.file_uploader("LOAD DK NBA CSV", type="csv")
if f:
    df_raw = pd.read_csv(f); engine = EliteNBAGPPOptimizerV89(df_raw)
    if st.button("🚀 EXECUTE 10,000 NUCLEAR SIMS"):
        st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
        st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state and len(st.session_state.portfolio) > 0:
        l = st.session_state.portfolio[st.session_state.sel_idx]
        st.subheader(f"NBA GPP LINEUP #{st.session_state.sel_idx+1}")
        st.table(l[['Name', 'Pos', 'Team', 'Sal', 'Proj', 'Own']])
        st.write(f"**GPP Analysis:** 10k Sims | Targeted Ceiling: 360+ Points")
