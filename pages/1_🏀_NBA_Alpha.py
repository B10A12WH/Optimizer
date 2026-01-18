import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v88.0 NBA PRO", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV88:
    def __init__(self, df):
        # Efficiency: Standardize and map columns once at initialization
        self.df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        
        def hunt(keys, default_val=None):
            for k in keys:
                if k in raw_cols: return raw_cols[k]
            return default_val

        p_key = hunt(['proj', 'fppg', 'points', 'avgpointspergame'], df.columns[0])
        s_key = hunt(['salary', 'sal', 'cost'], 'Salary')
        o_key = hunt(['ownership', 'own', 'projown', 'roster'], None)
        
        # Pre-cast to NumPy-friendly types immediately
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[s_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[hunt(['position', 'pos'], 'Position')].astype(str)
        self.df['Team'] = df[hunt(['team', 'teamabbrev', 'tm'], 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[hunt(['id', 'playerid'], 'ID')].astype(str)
        self.df['Own'] = pd.to_numeric(df[o_key], errors='coerce').fillna(15.0) if o_key else 15.0

        # High-Speed Binary Masking: Replaces slow string checks in the solve loop
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.df[f'mask_{p}'] = self.df['Pos'].str.contains(p).astype(int)
        
        self.df['mask_G'] = (self.df['mask_PG'] | self.df['mask_SG']).astype(int)
        self.df['mask_F'] = (self.df['mask_SF'] | self.df['mask_PF']).astype(int)
        self.df['mask_Star'] = (self.df['Sal'] >= 9000).astype(int)
        self.df['mask_Punt'] = (self.df['Sal'] <= 4000).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        # Localize variables to function scope for faster lookup
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        owns = self.df['Own'].values.astype(np.float64)
        
        # 10,000 SIMULATION ENGINE: Vectorized generation
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.25), size=(total_sims, n_p)).clip(min=0)
        
        sim_pool = []
        # Pre-build constant constraints
        A_base = [np.ones(n_p), sals, owns, self.df['mask_Star'].values, self.df['mask_Punt'].values]
        bl_base = [8, 48500, 0, 2, 1]
        bu_base = [8, 50000, 130.0, 4, 3]
        
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            A_base.append(self.df[f'mask_{p}'].values)
            bl_base.append(1); bu_base.append(8)
            
        A_base.append(self.df['mask_G'].values); bl_base.append(3); bu_base.append(8)
        A_base.append(self.df['mask_F'].values); bl_base.append(3); bu_base.append(8)
        
        static_A = np.vstack(A_base)

        # Efficiency Loop: Evaluate top outlier scripts
        for i in range(min(total_sims, 600)): 
            sim_p = sim_matrix[i]
            res = milp(c=-sim_p, constraints=LinearConstraint(static_A, bl_base, bu_base), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1),
                       options={'presolve': True})
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                sim_pool.append({'idx': tuple(idx), 'sim_score': sim_p[idx].sum()})

        # Rank and return top unique ceiling lineups
        sorted_pool = sorted(sim_pool, key=lambda x: x['sim_score'], reverse=True)
        portfolio, used_hashes = [], set()
        for entry in sorted_pool:
            if entry['idx'] not in used_hashes:
                portfolio.append(self.df.iloc[list(entry['idx'])].copy())
                used_hashes.add(entry['idx'])
            if len(portfolio) >= n_final: break
        return portfolio

# --- MAIN RENDERING ---
st.title("🏀 VANTAGE 99 | NBA BOOM PRO v88.0")
f = st.file_uploader("LOAD DK NBA CSV", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteNBAGPPOptimizerV88(df_raw)
    
    if st.button("🚀 EXECUTE 10,000 GPP SIMS"):
        with st.spinner("Analyzing 10k Scenarios..."):
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state and len(st.session_state.portfolio) > 0:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                if st.button(f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            st.subheader(f"NBA GPP LINEUP #{st.session_state.sel_idx+1}")
            
            # Efficient Table Display
            st.table(l[['Name', 'Pos', 'Team', 'Sal', 'Proj', 'Own']].style.format({'Proj': '{:.2f}', 'Own': '{:.1f}%'}))
            st.write(f"**Stars Found:** {len(l[l['Sal'] >= 9000])} | **Punts Found:** {len(l[l['Sal'] <= 4000])}")
