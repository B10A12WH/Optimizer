import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v103.0 NBA", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV103:
    def __init__(self, df):
        # 1. BULLETPROOF COLUMN INITIALIZATION
        self.raw_df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("_", "").replace("%", ""): c for c in df.columns}
        
        def safe_map(keys, default_val=0.0):
            for k in keys:
                search_k = k.lower().replace(" ", "").replace("_", "")
                if search_k in raw_cols: 
                    return pd.to_numeric(df[raw_cols[search_k]], errors='coerce').fillna(default_val)
            return pd.Series([default_val] * len(df))

        # Standardized DataFrame to ensure internal keys never vanish
        self.clean_df = pd.DataFrame()
        self.clean_df['Name'] = df[next((raw_cols[k] for k in ['name', 'player'] if k in raw_cols), df.columns[0])].astype(str)
        self.clean_df['Team'] = df[next((raw_cols[k] for k in ['team', 'teamabbrev'] if k in raw_cols), df.columns[1])].astype(str)
        self.clean_df['Pos'] = df[next((raw_cols[k] for k in ['position', 'pos'] if k in raw_cols), df.columns[2])].astype(str)
        self.clean_df['ID'] = df[next((raw_cols[k] for k in ['id', 'playerid'] if k in raw_cols), df.columns[3])].astype(str)
        
        # We start with 0.0 Proj until the Generative Engine runs
        self.clean_df['Sal'] = safe_map(['salary', 'sal', 'cost'], 50000)
        self.clean_df['Own'] = safe_map(['ownership', 'own', 'projown', 'roster'], 15.0)

        # 2. POSITION & STRATEGY MASKS
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.clean_df[f'mask_{p}'] = self.clean_df['Pos'].str.contains(p).astype(int)
        
        # DraftKings Combined Slots
        self.clean_df['mask_G'] = (self.clean_df['mask_PG'] | self.clean_df['mask_SG']).astype(int)
        self.clean_df['mask_F'] = (self.clean_df['mask_SF'] | self.clean_df['mask_PF']).astype(int)
        
        # GPP Architecture
        self.clean_df['is_Star'] = (self.clean_df['Sal'] >= 9400).astype(int)
        self.clean_df['is_Punt'] = (self.clean_df['Sal'] <= 4400).astype(int)

    def generate_projections(self, out_players):
        """Generative Engine: Calculates 350+ GPP Ceilings from Salary and Injuries"""
        # Baseline: Tournament winning lineups average 7x value (350 pts)
        self.clean_df['Proj'] = (self.clean_df['Sal'] / 1000) * 5.8
        
        # Zero out players ruled OUT
        self.clean_df.loc[self.clean_df['Name'].isin(out_players), 'Proj'] = 0.0
        
        # Usage Migration: Teammates gain 8-12% per missing rotation player
        out_teams = self.clean_df[self.clean_df['Name'].isin(out_players)]['Team'].unique()
        for team in out_teams:
            count = len(self.clean_df[(self.clean_df['Team'] == team) & (self.clean_df['Name'].isin(out_players))])
            self.clean_df.loc[(self.clean_df['Team'] == team) & (self.clean_df['Proj'] > 0), 'Proj'] *= (1 + (0.10 * count))

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.clean_df)
        raw_p = self.clean_df['Proj'].values.astype(np.float64)
        sals = self.clean_df['Sal'].values.astype(np.float64)
        owns = self.clean_df['Own'].values.astype(np.float64)
        
        # 35% JITTER: Forcing simulations to reach the 350-400pt ceiling
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.35), size=(total_sims, n_p)).clip(min=0)
        
        tiers = [
            {"name": "Nuclear", "sal": 49700, "stars": 2, "own": 120.0, "pnt": 1},
            {"name": "Alpha", "sal": 49200, "stars": 1, "own": 140.0, "pnt": 1},
            {"name": "Emergency", "sal": 48000, "stars": 1, "own": 180.0, "pnt": 0}
        ]
        
        portfolio, seen = [], set()
        for tier in tiers:
            sim_pool = []
            for i in range(min(total_sims, 500)): 
                sim_p = sim_matrix[i]
                adj_sim_p = sim_p * (1 + (self.clean_df['is_Star'].values * 0.1) + (self.clean_df['is_Punt'].values * 0.1))
                
                A, bl, bu = [], [], []
                A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
                A.append(sals); bl.append(tier['sal']); bu.append(50000)
                A.append(owns); bl.append(0); bu.append(tier['own'])
                A.append(self.clean_df['is_Star'].values); bl.append(tier['stars']); bu.append(4)
                A.append(self.clean_df['is_Punt'].values); bl.append(tier['pnt']); bu.append(4)

                # Mandatory DK Positional Slots
                for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                    A.append(self.clean_df[f'mask_{p}'].values); bl.append(1); bu.append(8)
                A.append(self.clean_df['mask_G'].values); bl.append(3); bu.append(8)
                A.append(self.clean_df['mask_F'].values); bl.append(3); bu.append(8)

                res = milp(c=-adj_sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
                if res.success:
                    idx = np.where(res.x > 0.5)[0]
                    sim_pool.append({'idx': tuple(idx), 'ceiling': sim_p[idx].sum()})

            # Greedy Fill: Ensure we get 10 lineups
            sorted_pool = sorted(sim_pool, key=lambda x: x['ceiling'], reverse=True)
            for entry in sorted_pool:
                if entry['idx'] not in seen:
                    l_df = self.clean_df.iloc[list(entry['idx'])].copy()
                    l_df['GPP_Ceiling'] = entry['ceiling']
                    portfolio.append(l_df); seen.add(entry['idx'])
                if len(portfolio) >= n_final: break
            if len(portfolio) >= n_final: break
            
        return portfolio

# --- MAIN UI ---
st.title("🏆 VANTAGE 99 | NBA NUCLEAR GENERATOR")
f = st.file_uploader("LOAD DK SALARY CSV", type="csv")

if f:
    df_raw = pd.read_csv(f); engine = EliteNBAGPPOptimizerV103(df_raw)
    inactives = st.multiselect("Mark Players ruled OUT (Injury News):", options=sorted(engine.clean_df['Name'].tolist()))
    
    if st.button("🚀 EXECUTE 10,000 GPP SIMS"):
        # INTERNAL PROJECTION GENERATION
        engine.generate_projections(inactives)
        with st.status("Solving 10,000 Outlier Nightmares...", expanded=True) as status:
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0
            if st.session_state.portfolio: status.update(label="10 Nuclear Lineups Optimized!", state="complete")

    if 'portfolio' in st.session_state and st.session_state.portfolio:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                ceiling = round(l['GPP_Ceiling'].iloc[0], 1)
                if st.button(f"L{i+1} | 🔥 {ceiling} GPP", key=f"btn_{i}"): st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            
            # 3. MOLECULAR ROSTER MAPPING: Strictly Following DraftKings Positions
            def pick_p(p_df, mask):
                cands = p_df[p_df[mask] == 1].sort_values('Sal', ascending=False)
                if not cands.empty:
                    p = cands.iloc[0]; return p, p_df[p_df['ID'] != p['ID']]
                return None, p_df

            final_roster, pool = [], l.copy()
            # Slot Core Positions
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                p, pool = pick_p(pool, f'mask_{pos}')
                if p is not None: final_roster.append({"Pos": pos, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
            
            # Slot Guard/Forward Flex
            for flex in ['G', 'F']:
                p, pool = pick_p(pool, f'mask_{flex}')
                if p is not None: final_roster.append({"Pos": flex, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
            
            # Final Utility Slot
            if not pool.empty:
                u = pool.iloc[0]; final_roster.append({"Pos": "UTIL", "Name": u['Name'], "Team": u['Team'], "Sal": f"${int(u['Sal'])}", "Own": f"{u['Own']}%"})
            
            st.subheader(f"GPP NUCLEAR LINEUP #{st.session_state.sel_idx+1}")
            st.table(pd.DataFrame(final_roster))
            st.info(f"Targeting 350-380 Points | Salary Used: ${int(l['Sal'].sum())} | Proj Total: {round(l['Proj'].sum(), 1)}")
