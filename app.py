import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import time

# --- VANTAGE 99: INDEPENDENT QUANT ENGINE (V46.0) ---
st.set_page_config(page_title="VANTAGE 99 | QUANT LAB", layout="wide", page_icon="🧪")

class QuantNFLOptimizer:
    def __init__(self, df):
        # 1. CLEAN HEADERS
        cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
        
        # 2. THE "INDEPENDENCE" LOGIC
        # If no ProjPts exists, we build a median projection from DK's AvgPointsPerGame
        # and then apply a "GPP Upside" multiplier based on the player's position.
        if not any(k in cols for k in ['projpts', 'proj', 'points']):
            avg_key = cols.get('avgpointspergame', next(iter(cols.values())))
            df['BaseProj'] = pd.to_numeric(df[avg_key], errors='coerce').fillna(2.0)
            
            # Position-Based Variance Multipliers (Mimics Range of Outcomes)
            # WRs/TEs have higher "Ceiling" potential than RBs due to PPR/100yd bonuses
            df['UpsideFactor'] = 1.15 # Default
            df.loc[df['Position'] == 'WR', 'UpsideFactor'] = 1.35
            df.loc[df['Position'] == 'QB', 'UpsideFactor'] = 1.20
            df.loc[df['Position'] == 'TE', 'UpsideFactor'] = 1.40
            
            df['Proj'] = df['BaseProj'] * df['UpsideFactor']
            st.info("🧬 QUANT MODE: Independent Ceiling Projections Calculated.")
        else:
            proj_key = next((cols[k] for k in cols if k in ['projpts', 'proj', 'points']), None)
            df['Proj'] = pd.to_numeric(df[proj_key], errors='coerce').fillna(0.0)

        # 3. CORE ATTRIBUTES
        df['Salary'] = pd.to_numeric(df[cols.get('salary', 'Salary')], errors='coerce').fillna(50000)
        df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        df['Name'] = df[cols.get('name', 'Name')].astype(str)
        df['ID'] = df[cols.get('id', 'ID')].astype(str)
        df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        
        for p in ['QB','RB','WR','TE','DST']: 
            df[f'is_{p}'] = (df['Pos'] == p).astype(int)
        
        self.df = df[df['Proj'] > 1.0].reset_index(drop=True)

    def generate_portfolio(self, n_sims=500, portfolio_size=20, max_exposure=0.5, jitter=0.25, stack_size=1):
        n_p = len(self.df)
        proj_vals = self.df['Proj'].values.astype(float)
        sal_vals = self.df['Salary'].values.astype(float)
        final_lineups = []
        exposure_counts = {name: 0 for name in self.df['Name']}
        
        bar = st.progress(0)
        for i in range(n_sims):
            if len(final_lineups) >= portfolio_size: break
            if i % (max(1, n_sims//10)) == 0: bar.progress(i/n_sims)
            
            # CEILING SIMULATION (Standard Deviation Jitter)
            # This mimics "Stokastic" simulations by creating 500 different 'game scripts'
            sim = np.random.normal(proj_vals, proj_vals * jitter).clip(min=0)
            
            # Linear Optimization Setup
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # Total players
            A.append(sal_vals); bl.append(49000.0); bu.append(50000.0) # Salary Floor/Cap
            
            # Positional Constraints
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)
            
            # QB STACKING (GPP Requirement)
            for qidx, row in self.df[self.df['is_QB'] == 1].iterrows():
                m = np.zeros(n_p)
                teammate_mask = (self.df['Team'] == row['Team']) & (self.df['Pos'].isin(['WR', 'TE']))
                m[teammate_mask] = 1
                m[qidx] = -stack_size 
                A.append(m); bl.append(0); bu.append(10)

            # Exposure Governor
            for idx, name in enumerate(self.df['Name']):
                if exposure_counts[name] >= (portfolio_size * max_exposure):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx_list = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx_list].copy()
                lineup_names = set(l_df['Name'].tolist())
                
                if not any(lineup_names == set(f['Names']) for f in final_lineups):
                    # Sort for DK Upload Format
                    qb = l_df[l_df['is_QB']==1].iloc[0]
                    dst = l_df[l_df['is_DST']==1].iloc[0]
                    rbs = l_df[l_df['is_RB']==1].sort_values('Salary', ascending=False)
                    wrs = l_df[l_df['is_WR']==1].sort_values('Salary', ascending=False)
                    tes = l_df[l_df['is_TE']==1].sort_values('Salary', ascending=False)
                    
                    main_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], tes.iloc[0]['ID'], dst['ID']]
                    flex = l_df[~l_df['ID'].isin(main_ids)].iloc[0]

                    final_lineups.append({
                        'QB': f"{qb['Name']} ({qb['ID']})", 'RB1': f"{rbs.iloc[0]['Name']} ({rbs.iloc[0]['ID']})",
                        'RB2': f"{rbs.iloc[1]['Name']} ({rbs.iloc[1]['ID']})", 'WR1': f"{wrs.iloc[0]['Name']} ({wrs.iloc[0]['ID']})",
                        'WR2': f"{wrs.iloc[1]['Name']} ({wrs.iloc[1]['ID']})", 'WR3': f"{wrs.iloc[2]['Name']} ({wrs.iloc[2]['ID']})",
                        'TE': f"{tes.iloc[0]['Name']} ({tes.iloc[0]['ID']})", 'FLEX': f"{flex['Name']} ({flex['ID']})", 
                        'DST': f"{dst['Name']} ({dst['ID']})", 'Names': list(lineup_names), 'Score': round(sim[idx_list].sum(), 2)
                    })
                    for name in lineup_names: exposure_counts[name] += 1

        bar.empty()
        return final_lineups

# --- UI LAYER ---
st.title("🧪 VANTAGE 99 | QUANT LAB")
st.markdown("### `INDEPENDENT DFS PORTFOLIO REBALANCER`")

f = st.file_uploader("UPLOAD DK SALARY FILE", type="csv")
if f:
    raw_df = pd.read_csv(f)
    # Skip metadata rows if present
    if "Field" in str(raw_df.columns):
        raw_df = pd.read_csv(f, skiprows=7)
    
    engine = QuantNFLOptimizer(raw_df)
    
    st.sidebar.subheader("🕹️ CONTROL PANEL")
    stacks = st.sidebar.slider("QB Stack Size", 1, 2, 1)
    sims = st.sidebar.slider("Sim Volume", 100, 1000, 500)
    exposure = st.sidebar.slider("Max Exposure (%)", 10, 100, 50) / 100.0
    
    if st.button("🚀 EXECUTE QUANT STRATEGY"):
        start = time.time()
        portfolio = engine.generate_portfolio(n_sims=sims, portfolio_size=20, max_exposure=exposure, stack_size=stacks)
        
        if portfolio:
            st.success(f"PORTFOLIO GENERATED IN {round(time.time()-start, 2)}s")
            st.dataframe(pd.DataFrame(portfolio).drop(columns=['Names']))
            csv = pd.DataFrame(portfolio)[['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']].to_csv(index=False)
            st.download_button("📥 DOWNLOAD DK UPLOAD FILE", csv, "Vantage_Independent.csv")
