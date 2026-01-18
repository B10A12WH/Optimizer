import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v83.0 NBA", layout="wide", page_icon="🏀")

class EliteGPPOptimizerV83:
    def __init__(self, df):
        self.df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        
        def hunt(keys, default_val=None):
            for k in keys:
                if k in raw_cols: return raw_cols[k]
            return default_val

        # 1. NBA COLUMN DETECTION
        p_key = hunt(['proj', 'fppg', 'points', 'avgpointspergame'], df.columns[0])
        s_key = hunt(['salary', 'sal', 'cost'], 'Salary')
        o_key = hunt(['ownership', 'own', 'projown', 'roster'], None)
        
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[s_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[hunt(['position', 'pos'], 'Position')].astype(str)
        self.df['Team'] = df[hunt(['team', 'teamabbrev', 'tm'], 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[hunt(['id', 'playerid'], 'ID')].astype(str)
        self.df['Own'] = pd.to_numeric(df[o_key], errors='coerce').fillna(15.0) if o_key else 15.0

        # 2. POSITION ELIGIBILITY (Including Dual-Eligible Players)
        self.df['is_PG'] = self.df['Pos'].str.contains('PG').astype(int)
        self.df['is_SG'] = self.df['Pos'].str.contains('SG').astype(int)
        self.df['is_SF'] = self.df['Pos'].str.contains('SF').astype(int)
        self.df['is_PF'] = self.df['Pos'].str.contains('PF').astype(int)
        self.df['is_C'] = self.df['Pos'].str.contains('C').astype(int)
        
        # Guard/Forward Combo Spots
        self.df['is_G'] = (self.df['is_PG'] | self.df['is_SG']).astype(int)
        self.df['is_F'] = (self.df['is_SF'] | self.df['is_PF']).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        owns = self.df['Own'].values.astype(np.float64)
        
        # 10,000 SIMULATION CORE: Tighter jitter for NBA (18% vs 22%)
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.18), size=(total_sims, n_p)).clip(min=0)
        
        sim_pool = []
        for i in range(min(total_sims, 600)): 
            sim_p = sim_matrix[i]
            A, bl, bu = [], [], []
            
            # DK NBA CLASSIC RULES: 8 Players, $50k Salary
            A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
            A.append(sals); bl.append(48000); bu.append(50000)
            A.append(owns); bl.append(0); bu.append(150.0) 
            
            # Mandatory Individual Spots (Min 1 each)
            for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                A.append(self.df[f'is_{p}'].values); bl.append(1); bu.append(8)

            # Combined Guard (G) and Forward (F) Requirements
            # DraftKings requires 3 total Guards (PG, SG, and 1 G) and 3 total Forwards (SF, PF, and 1 F)
            A.append(self.df['is_G'].values); bl.append(3); bu.append(8)
            A.append(self.df['is_F'].values); bl.append(3); bu.append(8)

            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                sim_pool.append({'idx': tuple(idx), 'sim_score': sim_p[idx].sum()})

        # Rank and return top 10 unique ceilings
        sorted_pool = sorted(sim_pool, key=lambda x: x['sim_score'], reverse=True)
        portfolio, used_hashes = [], set()
        for entry in sorted_pool:
            if entry['idx'] not in used_hashes:
                portfolio.append(self.df.iloc[list(entry['idx'])].copy())
                used_hashes.add(entry['idx'])
            if len(portfolio) >= n_final: break
        return portfolio

# --- MAIN RENDERING ---
st.title("🏆 VANTAGE 99 | NBA GPP COMMAND")
f = st.file_uploader("LOAD DK NBA DATASET", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteGPPOptimizerV83(df_raw)
    
    if st.button("🚀 INITIATE 10,000 NBA SIMS"):
        with st.spinner("Analyzing NBA High-Variance Scripts..."):
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
            st.subheader(f"NBA LINEUP #{st.session_state.sel_idx+1}")
            st.table(l[['Name', 'Pos', 'Team', 'Sal', 'Proj', 'Own']])
            st.write(f"**Total Salary:** ${int(l['Sal'].sum())} | **Projected Score:** {round(l['Proj'].sum(), 2)}")
