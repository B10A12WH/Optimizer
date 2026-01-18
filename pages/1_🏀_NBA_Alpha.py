import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v93.0 NUCLEAR", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV93:
    def __init__(self, df):
        self.df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        
        def hunt(keys, default_val=None):
            for k in keys:
                if k in raw_cols: return raw_cols[k]
            return default_val

        # Efficient Data Mapping
        p_key = hunt(['proj', 'fppg', 'points', 'avgpointspergame'], df.columns[0])
        s_key = hunt(['salary', 'sal', 'cost'], 'Salary')
        o_key = hunt(['ownership', 'own', 'projown', 'roster'], None)
        
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[s_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[hunt(['position', 'pos'], 'Position')].astype(str)
        self.df['ID'] = df[hunt(['id', 'playerid'], 'ID')].astype(str)
        
        # TRACEBACK FIX: Explicitly naming the ownership column for display logic
        self.df['Own'] = pd.to_numeric(df[o_key], errors='coerce').fillna(15.0) if o_key else 15.0

        # Vectorized Position Masking
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.df[f'mask_{p}'] = self.df['Pos'].str.contains(p).astype(int)
        self.df['mask_G'] = (self.df['mask_PG'] | self.df['mask_SG']).astype(int)
        self.df['mask_F'] = (self.df['mask_SF'] | self.df['mask_PF']).astype(int)

        # Nuclear Strategy Masks
        self.df['is_Star'] = (self.df['Sal'] >= 9500).astype(int)
        self.df['is_Punt'] = (self.df['Sal'] <= 4200).astype(int)
        self.df['is_Contrarian'] = (self.df['Own'] < 10.0).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        owns = self.df['Own'].values.astype(np.float64)
        
        # INCREASED CEILING JITTER (30%): Forcing true outlier 'Boom' scenarios
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.30), size=(total_sims, n_p)).clip(min=0)
        
        tiers = [
            {"name": "Nuclear", "sal": 49700, "stars": 2, "own": 120.0, "con": 3},
            {"name": "Alpha", "sal": 49400, "stars": 1, "own": 135.0, "con": 2},
            {"name": "GPP-Standard", "sal": 48500, "stars": 1, "own": 150.0, "con": 1}
        ]
        
        for tier in tiers:
            sim_pool = []
            for i in range(min(total_sims, 500)): 
                sim_p = sim_matrix[i]
                A, bl, bu = [], [], []
                
                # Core Roster
                A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
                A.append(sals); bl.append(tier['sal']); bu.append(50000)
                A.append(owns); bl.append(0); bu.append(tier['own'])
                
                # GPP Targets
                A.append(self.df['is_Star'].values); bl.append(tier['stars']); bu.append(4)
                A.append(self.df['is_Punt'].values); bl.append(1); bu.append(3)
                A.append(self.df['is_Contrarian'].values); bl.append(tier['con']); bu.append(8)

                # Positions
                for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.df[f'mask_{p}'].values); bl.append(1); bu.append(8)
                A.append(self.df['mask_G'].values); bl.append(3); bu.append(8)
                A.append(self.df['mask_F'].values); bl.append(3); bu.append(8)

                res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
                if res.success:
                    idx = np.where(res.x > 0.5)[0]
                    sim_pool.append({'idx': tuple(idx), 'sim_score': sim_p[idx].sum()})

            if len(sim_pool) >= n_final:
                st.toast(f"Mode: {tier['name']}", icon="🔥")
                sorted_pool = sorted(sim_pool, key=lambda x: x['sim_score'], reverse=True)
                portfolio, seen = [], set()
                for entry in sorted_pool:
                    if entry['idx'] not in seen:
                        portfolio.append(self.df.iloc[list(entry['idx'])].copy())
                        seen.add(entry['idx'])
                    if len(portfolio) >= n_final: break
                return portfolio
        return []

# --- MAIN UI ---
st.title("🏆 VANTAGE 99 | NBA NUCLEAR ALPHA")
f = st.file_uploader("LOAD DK NBA CSV", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteNBAGPPOptimizerV93(df_raw)
    
    if st.button("🚀 EXECUTE 10,000 SIMS"):
        with st.status("Crunching Outlier Scenarios...", expanded=True) as status:
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0
            if st.session_state.portfolio:
                status.update(label="Lineups Optimized!", state="complete", expanded=False)
            else:
                status.update(label="Infeasible Dataset", state="error")

    if 'portfolio' in st.session_state and st.session_state.portfolio:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                if st.button(f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            lineup = st.session_state.portfolio[st.session_state.sel_idx]
            
            # SAFE POSITION SLOTTING
            def pick_pos(p_df, pos_flag):
                cands = p_df[p_df[pos_flag] == 1].sort_values('Proj', ascending=False)
                if not cands.empty:
                    p = cands.iloc[0]
                    return p, p_df[p_df['ID'] != p['ID']]
                return None, p_df

            final_roster, pool = [], lineup.copy()
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                player, pool = pick_pos(pool, f'mask_{pos}')
                if player is not None: final_roster.append({"Pos": pos, "Name": player['Name'], "Team": player['Team'], "Sal": int(player['Sal']), "Own": f"{player['Own']}%"})
            
            for flex in ['mask_G', 'mask_F']:
                player, pool = pick_pos(pool, flex)
                if player is not None: final_roster.append({"Pos": flex[5:], "Name": player['Name'], "Team": player['Team'], "Sal": int(player['Sal']), "Own": f"{player['Own']}%"})
            
            if not pool.empty:
                u = pool.iloc[0]; final_roster.append({"Pos": "UTIL", "Name": u['Name'], "Team": u['Team'], "Sal": int(u['Sal']), "Own": f"{u['Own']}%"})
            
            st.subheader(f"NBA LINEUP #{st.session_state.sel_idx+1}")
            st.table(pd.DataFrame(final_roster))
            st.info(f"Roster Metrics: ${int(lineup['Sal'].sum())} Salary Used | {round(lineup['Own'].sum(), 1)}% Total Ownership")
