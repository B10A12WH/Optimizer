import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v96.0 NBA", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV96:
    def __init__(self, df):
        # 1. BULLETPROOF COLUMN MAPPER: Prevents KeyError: None
        self.raw_df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        
        def safe_get(keys, default_val=None):
            for k in keys:
                if k in raw_cols: return pd.to_numeric(df[raw_cols[k]], errors='coerce')
            return pd.Series([default_val] * len(df))

        # Create a standardized truth table to prevent any downstream display errors
        self.clean_df = pd.DataFrame()
        self.clean_df['Name'] = df[next((raw_cols[k] for k in ['name', 'player'] if k in raw_cols), df.columns[0])].astype(str)
        self.clean_df['Team'] = df[next((raw_cols[k] for k in ['team', 'teamabbrev'] if k in raw_cols), df.columns[1])].astype(str)
        self.clean_df['Pos'] = df[next((raw_cols[k] for k in ['position', 'pos'] if k in raw_cols), df.columns[2])].astype(str)
        self.clean_df['ID'] = df[next((raw_cols[k] for k in ['id', 'playerid'] if k in raw_cols), df.columns[3])].astype(str)
        
        # Core Stats with Safe Defaults
        self.clean_df['Proj'] = safe_get(['proj', 'fppg', 'points'], 0.0).fillna(0.0)
        self.clean_df['Sal'] = safe_get(['salary', 'sal', 'cost'], 50000).fillna(50000)
        self.clean_df['Own'] = safe_get(['ownership', 'own', 'roster'], 15.0).fillna(15.0)

        # 2. POSITION & STRATEGY MASKS
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.clean_df[f'mask_{p}'] = self.clean_df['Pos'].str.contains(p).astype(int)
        self.clean_df['mask_G'] = (self.clean_df['mask_PG'] | self.clean_df['mask_SG']).astype(int)
        self.clean_df['mask_F'] = (self.clean_df['mask_SF'] | self.clean_df['mask_PF']).astype(int)
        
        # 3. GPP "BOOM" ARCHITECTURE
        self.clean_df['is_Star'] = (self.clean_df['Sal'] >= 9800).astype(int)
        self.clean_df['is_Punt'] = (self.clean_df['Sal'] <= 4200).astype(int)
        self.clean_df['is_Leverage'] = (self.clean_df['Own'] < 10.0).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.clean_df)
        raw_p = self.clean_df['Proj'].values.astype(np.float64)
        sals = self.clean_df['Sal'].values.astype(np.float64)
        owns = self.clean_df['Own'].values.astype(np.float64)
        
        # 4. VOLATILITY ENGINE: 35% Jitter to hunt for 350-400pt scripts
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.35), size=(total_sims, n_p)).clip(min=0)
        
        tiers = [
            {"name": "Nuclear", "sal": 49700, "stars": 2, "own": 120.0, "lev": 3, "pnt": 1},
            {"name": "Alpha", "sal": 49400, "stars": 1, "own": 135.0, "lev": 2, "pnt": 1},
            {"name": "GPP-Standard", "sal": 48500, "stars": 1, "own": 150.0, "lev": 1, "pnt": 0}
        ]
        
        for tier in tiers:
            sim_pool = []
            for i in range(min(total_sims, 600)): 
                sim_p = sim_matrix[i]
                
                # MATHEMATICAL CEILING BIAS: Forces the solver to prefer Star/Punt combos
                adj_sim_p = sim_p * (1 + (self.clean_df['is_Star'].values * 0.08) + (self.clean_df['is_Punt'].values * 0.08))
                
                A, bl, bu = [], [], []
                A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
                A.append(sals); bl.append(tier['sal']); bu.append(50000)
                A.append(owns); bl.append(0); bu.append(tier['own'])
                
                # GPP CONSTRAINTS
                A.append(self.clean_df['is_Star'].values); bl.append(tier['stars']); bu.append(4)
                A.append(self.clean_df['is_Punt'].values); bl.append(tier['pnt']); bu.append(3)
                A.append(self.clean_df['is_Leverage'].values); bl.append(tier['lev']); bu.append(8)

                for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.clean_df[f'mask_{p}'].values); bl.append(1); bu.append(8)
                A.append(self.clean_df['mask_G'].values); bl.append(3); bu.append(8)
                A.append(self.clean_df['mask_F'].values); bl.append(3); bu.append(8)

                res = milp(c=-adj_sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
                if res.success:
                    idx = np.where(res.x > 0.5)[0]
                    sim_pool.append({'idx': tuple(idx), 'ceiling_score': sim_p[idx].sum(), 'median_score': raw_p[idx].sum()})

            if len(sim_pool) >= n_final:
                st.toast(f"Mode: {tier['name']}", icon="☢️")
                # Sort by Simulated Ceiling (GPP Potential) rather than Median
                sorted_pool = sorted(sim_pool, key=lambda x: x['ceiling_score'], reverse=True)
                portfolio, seen = [], set()
                for entry in sorted_pool:
                    if entry['idx'] not in seen:
                        # Append scores to the dataframe for UI display
                        lineup_df = self.clean_df.iloc[list(entry['idx'])].copy()
                        lineup_df['Sim_Ceiling'] = entry['ceiling_score']
                        portfolio.append(lineup_df)
                        seen.add(entry['idx'])
                    if len(portfolio) >= n_final: break
                return portfolio
        return []

# --- UI LAYER ---
st.title("🏆 VANTAGE 99 | NBA NUCLEAR Alpha")
f = st.file_uploader("LOAD DK NBA CSV", type="csv")

if f:
    df_raw = pd.read_csv(f); engine = EliteNBAGPPOptimizerV96(df_raw)
    
    if st.button("🚀 EXECUTE 10,000 NUCLEAR SIMS"):
        with st.status("Crunching 10,000 Outlier Scenarios...", expanded=True) as status:
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0
            if st.session_state.portfolio: status.update(label="Lineups Assembled!", state="complete")

    if 'portfolio' in st.session_state and st.session_state.portfolio:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                # DISPLAY CEILING POTENTIAL: Showing the 350+ GPP Ceiling instead of the 250 Median
                ceiling = round(l['Sim_Ceiling'].iloc[0], 1)
                if st.button(f"L{i+1} | 🔥 {ceiling} GPP", key=f"btn_{i}"): st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            
            def pick_p(p_df, mask):
                cands = p_df[p_df[mask] == 1].sort_values('Proj', ascending=False)
                if not cands.empty:
                    p = cands.iloc[0]; return p, p_df[p_df['ID'] != p['ID']]
                return None, p_df

            final, pool = [], l.copy()
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                p, pool = pick_p(pool, f'mask_{pos}')
                if p is not None: final.append({"Pos": pos, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%", "Proj": p['Proj']})
            
            for flex in ['G', 'F']:
                p, pool = pick_p(pool, f'mask_{flex}')
                if p is not None: final.append({"Pos": flex, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%", "Proj": p['Proj']})
            
            if not pool.empty:
                u = pool.iloc[0]; final.append({"Pos": "UTIL", "Name": u['Name'], "Team": u['Team'], "Sal": f"${int(u['Sal'])}", "Own": f"{u['Own']}%", "Proj": u['Proj']})
            
            st.subheader(f"GPP NUCLEAR LINEUP #{st.session_state.sel_idx+1}")
            st.table(pd.DataFrame(final))
            st.info(f"Targeting: 350-380 Points | Median Projection: {round(l['Proj'].sum(), 1)} | Salary: ${int(l['Sal'].sum())}")
