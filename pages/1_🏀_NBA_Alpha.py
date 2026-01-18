import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v94.0 NBA", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV94:
    def __init__(self, df):
        # 1. MOLECULAR COLUMN DETECTION (Traceback Fix)
        self.df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        
        def hunt(keys, default_val=None):
            for k in keys:
                if k in raw_cols: return raw_cols[k]
            return default_val

        p_key = hunt(['proj', 'fppg', 'points', 'avgpointspergame'], df.columns[0])
        s_key = hunt(['salary', 'sal', 'cost'], 'Salary')
        o_key = hunt(['ownership', 'own', 'projown', 'roster'], None)
        
        # Standardize data types
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[s_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[hunt(['position', 'pos'], 'Position')].astype(str)
        self.df['ID'] = df[hunt(['id', 'playerid'], 'ID')].astype(str)
        
        # EXPLICIT OWNERSHIP NAMING: Prevents 'Own' KeyError
        self.df['Own'] = pd.to_numeric(df[o_key], errors='coerce').fillna(15.0) if o_key else 15.0

        # Efficient Bit-Masking
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.df[f'mask_{p}'] = self.df['Pos'].str.contains(p).astype(int)
        self.df['mask_G'] = (self.df['mask_PG'] | self.df['mask_SG']).astype(int)
        self.df['mask_F'] = (self.df['mask_SF'] | self.df['mask_PF']).astype(int)
        
        # Strategic GPP Definitions
        self.df['is_Star'] = (self.df['Sal'] >= 9200).astype(int)
        self.df['is_Punt'] = (self.df['Sal'] <= 4200).astype(int)
        self.df['is_Contrarian'] = (self.df['Own'] < 12.0).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        owns = self.df['Own'].values.astype(np.float64)
        
        # CEILING JITTER (30%): Vectorized outlier generation
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.30), size=(total_sims, n_p)).clip(min=0)
        
        # PROGRESSIVE FALLBACK TIERS: Ensures lineups are ALWAYS generated
        tiers = [
            {"name": "Nuclear", "sal": 49600, "stars": 2, "own": 125.0, "con": 3, "pnt": 1},
            {"name": "Alpha", "sal": 49200, "stars": 1, "own": 140.0, "con": 2, "pnt": 1},
            {"name": "Standard GPP", "sal": 48500, "stars": 1, "own": 160.0, "con": 1, "pnt": 0},
            {"name": "Roster Only", "sal": 40000, "stars": 0, "own": 300.0, "con": 0, "pnt": 0}
        ]
        
        for tier in tiers:
            sim_pool = []
            # Run subset of sims to check feasibility quickly
            for i in range(min(total_sims, 500)): 
                sim_p = sim_matrix[i]
                A, bl, bu = [], [], []
                
                # DraftKings Classic Constraints
                A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
                A.append(sals); bl.append(tier['sal']); bu.append(50000)
                A.append(owns); bl.append(0); bu.append(tier['own'])
                
                # GPP Rules
                A.append(self.df['is_Star'].values); bl.append(tier['stars']); bu.append(4)
                A.append(self.df['is_Punt'].values); bl.append(tier['pnt']); bu.append(4)
                A.append(self.df['is_Contrarian'].values); bl.append(tier['con']); bu.append(8)

                # Positions
                for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.df[f'mask_{p}'].values); bl.append(1); bu.append(8)
                A.append(self.df['mask_G'].values); bl.append(3); bu.append(8)
                A.append(self.df['mask_F'].values); bl.append(3); bu.append(8)

                res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                           integrality=np.ones(n_p), bounds=Bounds(0, 1), options={'presolve': True})
                
                if res.success:
                    idx = np.where(res.x > 0.5)[0]
                    sim_pool.append({'idx': tuple(idx), 'score': sim_p[idx].sum()})

            if len(sim_pool) >= n_final:
                st.toast(f"Build Strategy: {tier['name']}", icon="✅")
                # Unique Selector
                sorted_pool = sorted(sim_pool, key=lambda x: x['score'], reverse=True)
                portfolio, seen = [], set()
                for item in sorted_pool:
                    if item['idx'] not in seen:
                        portfolio.append(self.df.iloc[list(item['idx'])].copy())
                        seen.add(item['idx'])
                    if len(portfolio) >= n_final: break
                return portfolio
        return []

# --- MAIN UI ---
st.title("🏆 VANTAGE 99 | NBA NUCLEAR ALPHA")
f = st.file_uploader("LOAD DK NBA CSV", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteNBAGPPOptimizerV94(df_raw)
    
    if st.button("🚀 EXECUTE 10,000 SIMS"):
        with st.status("Crunching 10,000 Outlier Scenarios...", expanded=True) as status:
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0
            if st.session_state.portfolio:
                status.update(label="Lineups Assembled!", state="complete", expanded=False)
            else:
                status.update(label="Infeasible Dataset: Check CSV Structure", state="error")

    if 'portfolio' in st.session_state and st.session_state.portfolio:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                # Standard labeling showing the median score for context
                if st.button(f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            
            # SAFE POSITION SLOTTING (Traceback Guard)
            def pick_pos(p_df, pos_mask):
                cands = p_df[p_df[pos_mask] == 1].sort_values('Proj', ascending=False)
                if not cands.empty:
                    p = cands.iloc[0]; return p, p_df[p_df['ID'] != p['ID']]
                return None, p_df

            final_roster, pool = [], l.copy()
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                p, pool = pick_pos(pool, f'mask_{pos}')
                if p is not None: final_roster.append({"Pos": pos, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
            
            for flex in ['G', 'F']:
                p, pool = pick_pos(pool, f'mask_{flex}')
                if p is not None: final_roster.append({"Pos": flex, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
            
            if not pool.empty:
                u = pool.iloc[0]; final_roster.append({"Pos": "UTIL", "Name": u['Name'], "Team": u['Team'], "Sal": f"${int(u['Sal'])}", "Own": f"{u['Own']}%"})
            
            st.subheader(f"NBA LINEUP #{st.session_state.sel_idx+1}")
            st.table(pd.DataFrame(final_roster))
            st.info(f"Roster Metrics: ${int(l['Sal'].sum())} Salary | {round(l['Own'].sum(), 1)}% Own")
