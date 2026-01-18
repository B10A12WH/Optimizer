import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v76.0 GPP", layout="wide", page_icon="🏆")

class EliteGPPOptimizerV76:
    def __init__(self, df):
        cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        self.df = df.copy()
        
        p_key = next((cols[k] for k in cols if k in ['proj', 'fppg', 'points', 'avgpointspergame']), df.columns[0])
        sal_key = next((cols[k] for k in cols if k in ['salary', 'sal', 'cost']), 'Salary')
        
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[sal_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        self.df['Own'] = pd.to_numeric(df.get('Ownership', 15.0), errors='coerce').fillna(15.0)

        # Matchup Weighting (LAR @ CHI Shootout Target)
        vegas_boosts = {'HOU': 0.88, 'NE': 1.08, 'LAR': 1.15, 'CHI': 1.04}
        self.df['Proj'] = self.df.apply(lambda x: x['Proj'] * vegas_boosts.get(x['Team'], 1.0), axis=1)
        
        # 3:00 PM vs 6:30 PM Start Times
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']: self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
        
        # Hard Scrub: Mack Hollins & Nico Collins are OUT
        inactives = ['Nico Collins', 'Justin Watson', 'Mack Hollins', 'Nick McCloud']
        self.df = self.df[~self.df['Name'].isin(inactives)].reset_index(drop=True)

    def assemble(self, n=10, diversity=4):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        owns = self.df['Own'].values.astype(np.float64)
        late_mask = self.df['is_late'].values.astype(np.float64) # Identify late game players
        
        portfolio, used_lineups = [], []

        for i in range(n):
            # SIMULATION: 22% Variance + Late Swap Bias
            # We add a tiny 'shadow' projection to late players to favor them for FLEX
            sim_p = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.22), size=raw_p.shape).clip(min=0)
            sim_p += (late_mask * 0.01) 
            
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

            # Unique Lineup Control
            for past_idx in used_lineups:
                row = np.zeros(n_p); row[list(past_idx)] = 1
                A.append(row); bl.append(0); bu.append(diversity)

            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                portfolio.append(self.df.iloc[idx].copy())
                used_lineups.append(tuple(idx))
        return portfolio

# --- MAIN RENDERING ---
st.title("🏆 VANTAGE 99 | v76.0 GPP PORTFOLIO")
f = st.file_uploader("LOAD DK DATASET", type="csv")
if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteGPPOptimizerV76(df_raw)
    
    if st.button("🚀 INITIATE 10-LINEUP GPP ASSEMBLY"):
        st.session_state.portfolio = engine.assemble(n=10)
        st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state and len(st.session_state.portfolio) > 0:
        col_list, col_scout = st.columns([1, 2.5])
        with col_list:
            st.markdown("### 📋 PORTFOLIO INDEX")
            for i, l in enumerate(st.session_state.portfolio):
                score = round(l['Proj'].sum(), 1)
                if st.button(f"L{i+1} | {score} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            # --- IMPROVED FLEX DISPLAY LOGIC ---
            # Group into core spots, then assign the LATEST kickoff player to FLEX
            qb = l[l['is_QB']==1].iloc[0]; dst = l[l['is_DST']==1].iloc[0]; te = l[l['is_TE']==1].iloc[0]
            rbs = l[l['is_RB']==1].sort_values(['is_late', 'Sal'], ascending=[True, False])
            wrs = l[l['is_WR']==1].sort_values(['is_late', 'Sal'], ascending=[True, False])
            
            # Identify core players (earliest start times)
            core_rbs = rbs.iloc[:2]; core_wrs = wrs.iloc[:3]
            core_ids = [qb['ID'], core_rbs.iloc[0]['ID'], core_rbs.iloc[1]['ID'], 
                        core_wrs.iloc[0]['ID'], core_wrs.iloc[1]['ID'], core_wrs.iloc[2]['ID'], 
                        te['ID'], dst['ID']]
            
            # The player NOT in core_ids is the FLEX (guaranteed to be latest possible)
            flex = l[~l['ID'].isin(core_ids)].iloc[0]
            
            display_order = [("QB", qb), ("RB", core_rbs.iloc[0]), ("RB", core_rbs.iloc[1]), 
                             ("WR", core_wrs.iloc[0]), ("WR", core_wrs.iloc[1]), ("WR", core_wrs.iloc[2]), 
                             ("TE", te), ("FLEX", flex), ("DST", dst)]
            
            st.subheader(f"LINEUP #{st.session_state.sel_idx+1} (LATE-SWAP READY)")
            st.table(pd.DataFrame([{"Pos": lbl, "Name": p['Name'], "Team": p['Team'], "Salary": f"${int(p['Sal'])}", "Proj": round(p['Proj'], 2)} for lbl, p in display_order]))
            st.markdown(f"**Total Projected Score:** {round(l['Proj'].sum(), 2)}")
