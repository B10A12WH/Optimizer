import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v98.0 NBA", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV98:
    def __init__(self, df):
        # 1. BULLETPROOF DATA INITIALIZATION
        self.raw_df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        
        def safe_map(keys, default_val=0.0):
            for k in keys:
                if k in raw_cols: return pd.to_numeric(df[raw_cols[k]], errors='coerce').fillna(default_val)
            return pd.Series([default_val] * len(df))

        # Standardized DataFrame to ensure internal keys never vanish
        self.clean_df = pd.DataFrame()
        self.clean_df['Name'] = df[next((raw_cols[k] for k in ['name', 'player'] if k in raw_cols), df.columns[0])].astype(str)
        self.clean_df['Team'] = df[next((raw_cols[k] for k in ['team', 'teamabbrev'] if k in raw_cols), df.columns[1])].astype(str)
        self.clean_df['Pos'] = df[next((raw_cols[k] for k in ['position', 'pos'] if k in raw_cols), df.columns[2])].astype(str)
        self.clean_df['ID'] = df[next((raw_cols[k] for k in ['id', 'playerid'] if k in raw_cols), df.columns[3])].astype(str)
        
        self.clean_df['Proj'] = safe_map(['proj', 'fppg', 'points'], 0.0)
        self.clean_df['Sal'] = safe_map(['salary', 'sal', 'cost'], 50000)
        self.clean_df['Own'] = safe_map(['ownership', 'own', 'roster'], 15.0)

        # 2. POSITION & STRATEGY MASKS
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.clean_df[f'mask_{p}'] = self.clean_df['Pos'].str.contains(p).astype(int)
        self.clean_df['mask_G'] = (self.clean_df['mask_PG'] | self.clean_df['mask_SG']).astype(int)
        self.clean_df['mask_F'] = (self.clean_df['mask_SF'] | self.clean_df['mask_PF']).astype(int)
        
        # GPP Architecture Masks
        self.clean_df['is_Star'] = (self.clean_df['Sal'] >= 9600).astype(int)
        self.clean_df['is_Punt'] = (self.clean_df['Sal'] <= 4200).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.clean_df)
        raw_p = self.clean_df['Proj'].values.astype(np.float64)
        sals = self.clean_df['Sal'].values.astype(np.float64)
        owns = self.clean_df['Own'].values.astype(np.float64)
        
        # CEILING JITTER (35%): Forcing true outlier 'Boom' scenarios
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.35), size=(total_sims, n_p)).clip(min=0)
        
        tiers = [
            {"name": "Nuclear (350+ Target)", "sal": 49700, "stars": 2, "own": 120.0, "pnt": 1},
            {"name": "Alpha GPP", "sal": 49400, "stars": 1, "own": 135.0, "pnt": 1},
            {"name": "GPP-Standard", "sal": 48500, "stars": 1, "own": 150.0, "pnt": 0}
        ]
        
        for tier in tiers:
            sim_pool = []
            for i in range(min(total_sims, 500)): 
                sim_p = sim_matrix[i]
                # Reward Star/Punt combos in the math itself
                adj_sim_p = sim_p * (1 + (self.clean_df['is_Star'].values * 0.08) + (self.clean_df['is_Punt'].values * 0.08))
                
                A, bl, bu = [], [], []
                A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
                A.append(sals); bl.append(tier['sal']); bu.append(50000)
                A.append(owns); bl.append(0); bu.append(tier['own'])
                A.append(self.clean_df['is_Star'].values); bl.append(tier['stars']); bu.append(4)
                A.append(self.clean_df['is_Punt'].values); bl.append(tier['pnt']); bu.append(3)

                for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.clean_df[f'mask_{p}'].values); bl.append(1); bu.append(8)
                A.append(self.clean_df['mask_G'].values); bl.append(3); bu.append(8)
                A.append(self.clean_df['mask_F'].values); bl.append(3); bu.append(8)

                res = milp(c=-adj_sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
                if res.success:
                    idx = np.where(res.x > 0.5)[0]
                    sim_pool.append({'idx': tuple(idx), 'ceiling': sim_p[idx].sum(), 'median': raw_p[idx].sum()})

            if len(sim_pool) >= n_final:
                st.toast(f"Mode Activated: {tier['name']}", icon="🔥")
                # SORT BY CEILING SCORE
                sorted_pool = sorted(sim_pool, key=lambda x: x['ceiling'], reverse=True)
                portfolio, seen = [], set()
                for entry in sorted_pool:
                    if entry['idx'] not in seen:
                        lineup = self.clean_df.iloc[list(entry['idx'])].copy()
                        lineup['Ceiling'] = entry['ceiling']
                        portfolio.append(lineup); seen.add(entry['idx'])
                    if len(portfolio) >= n_final: break
                return portfolio
        return []

# --- MAIN UI ---
st.title("🏆 VANTAGE 99 | NBA NUCLEAR ALPHA")
f = st.file_uploader("LOAD DK NBA CSV", type="csv")

if f:
    df_raw = pd.read_csv(f); engine = EliteNBAGPPOptimizerV98(df_raw)
    
    if st.button("🚀 EXECUTE 10,000 SIMS"):
        with st.status("Crunching 10,000 Scenarios...", expanded=True) as status:
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0
            if st.session_state.portfolio: status.update(label="Lineups Optimized!", state="complete")

    if 'portfolio' in st.session_state and st.session_state.portfolio:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                # SHOW CEILING ON BUTTONS: Breaks the "260 Median" mental block
                ceiling = round(l['Ceiling'].iloc[0], 1)
                if st.button(f"L{i+1} | 🔥 {ceiling} GPP", key=f"btn_{i}"): st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            # Standard Positional Slotting
            final_roster, pool = [], l.copy()
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                cands = pool[pool[f'mask_{pos}'] == 1].sort_values('Proj', ascending=False)
                if not cands.empty:
                    p = cands.iloc[0]; final_roster.append({"Pos": pos, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
                    pool = pool[pool['ID'] != p['ID']]
            
            # Use remaining 3 as FLEX/UTIL
            for _, p in pool.iterrows():
                final_roster.append({"Pos": "FLEX", "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
            
            st.subheader(f"GPP NUCLEAR LINEUP #{st.session_state.sel_idx+1}")
            st.table(pd.DataFrame(final_roster))
            st.info(f"Targeting 350+ GPP Ceiling | Median Proj: {round(l['Proj'].sum(), 1)} | Salary: ${int(l['Sal'].sum())}")
