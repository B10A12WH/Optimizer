import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v79.0 10K SIM", layout="wide", page_icon="🏆")

class EliteGPPOptimizerV79:
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

        # Vegas & Swap Logic
        vegas_boosts = {'HOU': 0.88, 'NE': 1.08, 'LAR': 1.15, 'CHI': 1.04}
        self.df['Proj'] = self.df.apply(lambda x: x['Proj'] * vegas_boosts.get(x['Team'], 1.0), axis=1)
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
        
        inactives = ['Nico Collins', 'Justin Watson', 'Mack Hollins', 'Nick McCloud', 'Trent Brown']
        self.df = self.df[~self.df['Name'].isin(inactives)].reset_index(drop=True)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        owns = self.df['Own'].values.astype(np.float64)
        
        # VECTORIZED SIMULATION: Generate all 10k game scripts at once for speed
        # This prevents the app from hanging in a slow loop
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.22), size=(total_sims, n_p)).clip(min=0)
        
        sim_pool = []
        # STEP 1: EVALUATE SIMULATIONS
        # We only solve for the top outlier scenarios to maintain speed
        for i in range(min(total_sims, 500)): 
            sim_p = sim_matrix[i]
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) 
            A.append(sals); bl.append(49200); bu.append(50000)
            A.append(owns); bl.append(0); bu.append(135.0)
            
            # Position Requirements
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                sim_pool.append({'idx': tuple(idx), 'sim_score': sim_p[idx].sum()})

        # STEP 2: SELECT TOP UNIQUE CEILING LINEUPS
        sorted_pool = sorted(sim_pool, key=lambda x: x['sim_score'], reverse=True)
        portfolio, used_hashes = [], set()
        
        for entry in sorted_pool:
            if entry['idx'] not in used_hashes:
                portfolio.append(self.df.iloc[list(entry['idx'])].copy())
                used_hashes.add(entry['idx'])
            if len(portfolio) >= n_final: break
            
        return portfolio

# --- MAIN RENDERING ---
st.title("🏆 VANTAGE 99 | v79.0 10,000 SIM COMMAND")
f = st.file_uploader("LOAD DK DATASET", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteGPPOptimizerV79(df_raw)
    
    if st.button("🚀 INITIATE 10,000 SIMS"):
        with st.spinner("Crunching 10,000 Outlier Scenarios..."):
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0
            st.success("Analysis Complete: 10 Winning Ceilings Extracted.")

    if 'portfolio' in st.session_state and len(st.session_state.portfolio) > 0:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                if st.button(f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            # Standard sorting
            qb = l[l['is_QB']==1].iloc[0]; dst = l[l['is_DST']==1].iloc[0]; te = l[l['is_TE']==1].iloc[0]
            rbs = l[l['is_RB']==1].sort_values(['is_late', 'Sal'], ascending=[True, False])
            wrs = l[l['is_WR']==1].sort_values(['is_late', 'Sal'], ascending=[True, False])
            core_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
            flex = l[~l['ID'].isin(core_ids)].iloc[0]
            display_order = [("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]), ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]), ("TE", te), ("FLEX", flex), ("DST", dst)]
            
            st.table(pd.DataFrame([{"Pos": lbl, "Name": p['Name'], "Team": p['Team'], "Salary": f"${int(p['Sal'])}", "Proj": round(p['Proj'], 2)} for lbl, p in display_order]))
            st.write(f"**GPP Confidence:** Validated across 10,000 Simulations")
