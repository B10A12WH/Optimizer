import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v101.0 NBA", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV101:
    def __init__(self, df):
        # 1. MOLECULAR COLUMN MAPPING
        self.raw_df = df.copy()
        cols = {c.lower().replace(" ", "").replace("_", "").replace("%", ""): c for c in df.columns}
        
        # Standardized DataFrame Creation (Internal Schema)
        self.clean_df = pd.DataFrame()
        self.clean_df['Name'] = df[next((cols[k] for k in ['name', 'player'] if k in cols), df.columns[0])].astype(str)
        self.clean_df['Team'] = df[next((cols[k] for k in ['team', 'teamabbrev'] if k in cols), df.columns[1])].astype(str)
        self.clean_df['Pos'] = df[next((cols[k] for k in ['position', 'pos'] if k in cols), df.columns[2])].astype(str)
        self.clean_df['ID'] = df[next((cols[k] for k in ['id', 'playerid'] if k in cols), df.columns[3])].astype(str)
        
        # Safe Numeric Mapping
        self.clean_df['Sal'] = pd.to_numeric(df[next((cols[k] for k in ['salary', 'sal', 'cost'] if k in cols), 'Salary')], errors='coerce').fillna(50000)
        self.clean_df['Own'] = pd.to_numeric(df[next((cols[k] for k in ['ownership', 'own', 'roster'] if k in cols), None)], errors='coerce').fillna(15.0) if any(x in cols for x in ['own','ownership']) else 15.0

        # Position Masking
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            self.clean_df[f'mask_{p}'] = self.clean_df['Pos'].str.contains(p).astype(int)
        self.clean_df['mask_G'] = (self.clean_df['mask_PG'] | self.clean_df['mask_SG']).astype(int)
        self.clean_df['mask_F'] = (self.clean_df['mask_SF'] | self.clean_df['mask_PF']).astype(int)

    def apply_usage_projections(self, out_players):
        """Generates projections from salary + injury report"""
        # Baseline: NBA GPP winners usually average 6.5x value
        # We start with a 5.5x baseline and boost from there
        self.clean_df['Proj'] = (self.clean_df['Sal'] / 1000) * 5.5
        
        # Zero out players ruled OUT
        self.clean_df.loc[self.clean_df['Name'].isin(out_players), 'Proj'] = 0.0
        
        # Injury Boost Logic: If a teammate is out, others get more usage
        out_teams = self.clean_df[self.clean_df['Name'].isin(out_players)]['Team'].unique()
        for team in out_teams:
            num_out = len(self.clean_df[(self.clean_df['Team'] == team) & (self.clean_df['Name'].isin(out_players))])
            # Boost remaining players on the same team by 8% per missing teammate
            self.clean_df.loc[(self.clean_df['Team'] == team) & (~self.clean_df['Name'].isin(out_players)), 'Proj'] *= (1 + (0.08 * num_out))

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.clean_df)
        raw_p = self.clean_df['Proj'].values.astype(np.float64)
        sals = self.clean_df['Sal'].values.astype(np.float64)
        owns = self.clean_df['Own'].values.astype(np.float64)
        
        # CEILING JITTER (38%): Forcing the 10k sims to hunt outlier 350-400pt scenarios
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.38), size=(total_sims, n_p)).clip(min=0)
        
        tiers = [
            {"name": "Nuclear", "sal": 49700, "stars": 2, "own": 120.0, "pnt": 1},
            {"name": "Alpha", "sal": 49400, "stars": 1, "own": 140.0, "pnt": 1},
            {"name": "Emergency", "sal": 48000, "stars": 1, "own": 180.0, "pnt": 0}
        ]
        
        portfolio, seen = [], set()
        for tier in tiers:
            sim_pool = []
            for i in range(min(total_sims, 500)):
                sim_p = sim_matrix[i]
                A, bl, bu = [], [], []
                A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
                A.append(sals); bl.append(tier['sal']); bu.append(50000)
                A.append(owns); bl.append(0); bu.append(tier['own'])
                
                # GPP Architecture
                A.append((self.clean_df['Sal'] >= 9400).astype(int).values); bl.append(tier['stars']); bu.append(4)
                A.append((self.clean_df['Sal'] <= 4400).astype(int).values); bl.append(tier['pnt']); bu.append(3)

                for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.clean_df[f'mask_{p}'].values); bl.append(1); bu.append(8)
                A.append(self.clean_df['mask_G'].values); bl.append(3); bu.append(8)
                A.append(self.clean_df['mask_F'].values); bl.append(3); bu.append(8)

                res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
                if res.success:
                    idx = np.where(res.x > 0.5)[0]
                    sim_pool.append({'idx': tuple(idx), 'ceiling': sim_p[idx].sum()})

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
st.title("🏆 VANTAGE 99 | NBA GENERATIVE NUCLEAR")
f = st.file_uploader("LOAD DK SALARY CSV", type="csv")

if f:
    df_raw = pd.read_csv(f); engine = EliteNBAGPPOptimizerV101(df_raw)
    
    # 2. THE INJURY REPORT
    st.markdown("### 🚑 INJURY REPORT")
    all_players = sorted(engine.clean_df['Name'].tolist())
    inactives = st.multiselect("Mark Players ruled OUT (Injury Report):", options=all_players)
    
    if st.button("🚀 GENERATE & SIMULATE (10,000 SCENARIOS)"):
        # The engine creates projections internally based on your selections
        engine.apply_usage_projections(inactives)
        with st.status("Crunching Outlier Nightmares...", expanded=True) as status:
            st.session_state.portfolio = engine.assemble(n_final=10, total_sims=10000)
            st.session_state.sel_idx = 0
            if st.session_state.portfolio: status.update(label="10 Winning Ceilings Extracted!", state="complete")

    if 'portfolio' in st.session_state and st.session_state.portfolio:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            st.markdown("### 📋 PORTFOLIO")
            for i, l in enumerate(st.session_state.portfolio):
                ceiling = round(l['GPP_Ceiling'].iloc[0], 1)
                # DISPLAYING CEILING (350+ range)
                if st.button(f"L{i+1} | 🔥 {ceiling} GPP", key=f"btn_{i}"): st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            final_roster, pool = [], l.copy()
            # Standard positional slotting to match manual entry
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                cands = pool[pool[f'mask_{pos}'] == 1].sort_values('Sal', ascending=False)
                if not cands.empty:
                    p = cands.iloc[0]; final_roster.append({"Pos": pos, "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
                    pool = pool[pool['ID'] != p['ID']]
            for _, p in pool.iterrows():
                final_roster.append({"Pos": "FLEX", "Name": p['Name'], "Team": p['Team'], "Sal": f"${int(p['Sal'])}", "Own": f"{p['Own']}%"})
            
            st.subheader(f"GPP NUCLEAR LINEUP #{st.session_state.sel_idx+1}")
            st.table(pd.DataFrame(final_roster))
            st.info(f"Targeting 350+ GPP Ceiling | Generated Proj Total: {round(l['Proj'].sum(), 1)}")
