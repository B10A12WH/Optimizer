import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import time

# --- VANTAGE 99: NFL GPP ENGINE (V43.0 - HEADER AGNOSTIC) ---
st.set_page_config(page_title="VANTAGE 99 | NFL PORTFOLIO LAB", layout="wide", page_icon="🏈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] { background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d; border-radius: 12px; padding: 15px; }
    .roster-card { background: linear-gradient(145deg, #161b22, #0d1117); border: 1px solid #c9d1d9; border-radius: 12px; padding: 20px; margin: 10px 0px; border-left: 5px solid #00ffcc; }
    .pulse { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #00ffcc; animation: pulse 2s infinite; margin-right: 8px; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0, 255, 204, 0.7); } 70% { box-shadow: 0 0 0 8px rgba(0, 255, 204, 0); } 100% { box-shadow: 0 0 0 0 rgba(0, 255, 204, 0); } }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    # Deep scan to skip metadata rows (DK specific)
    for i, row in df.head(15).iterrows():
        vals = [str(v).lower() for v in row.values]
        if any(x in vals for x in ['name', 'player', 'salary', 'price']):
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            new_df.columns = [str(c).strip() for c in new_df.columns]
            return new_df.reset_index(drop=True)
    return df

class NFLIndustrialOptimizer:
    def __init__(self, df):
        # FUZZY HEADER MAPPING: Fixes the KeyError
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        
        proj_key = next((cols[k] for k in cols if k in ['projpts', 'proj', 'points', 'fp', 'fpts', 'projection']), None)
        sal_key = next((cols[k] for k in cols if k in ['salary', 'sal', 'price']), 'Salary')
        pos_key = next((cols[k] for k in cols if k in ['position', 'pos', 'rosterposition']), 'Position')
        name_key = next((cols[k] for k in cols if k in ['name', 'player']), 'Name')
        id_key = next((cols[k] for k in cols if k in ['id', 'playerid']), 'ID')
        team_key = next((cols[k] for k in cols if k in ['team', 'teamabbrev', 'tm']), 'TeamAbbrev')

        if not proj_key:
            st.error("❌ ERROR: No Projection column detected. Ensure your CSV has a column named 'ProjPts' or 'Points'.")
            st.stop()

        # Clean numeric data
        df['Proj'] = pd.to_numeric(df[proj_key], errors='coerce').fillna(0.0)
        df['Salary'] = pd.to_numeric(df[sal_key], errors='coerce').fillna(50000)
        df['Pos'] = df[pos_key].astype(str)
        df['Name'] = df[name_key].astype(str)
        df['ID'] = df[id_key].astype(str)
        df['Team'] = df[team_key].astype(str)
        
        # NFL Core Roster Logic
        for p in ['QB','RB','WR','TE','DST']: 
            df[f'is_{p}'] = (df['Pos'] == p).astype(int)
        
        # Tomorrow's "Out" List (Jan 18, 2026)
        scrubbed = ['Nico Collins', 'Justin Watson'] 
        self.df = df[~df['Name'].isin(scrubbed)].reset_index(drop=True)

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
            
            sim = np.random.normal(proj_vals, proj_vals * jitter).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # 9 Players
            A.append(sal_vals); bl.append(48800.0); bu.append(50000.0) # Salary Cap
            
            # Position Guards
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)
            
            # QB Stacking Constraint
            for qidx, row in self.df[self.df['is_QB'] == 1].iterrows():
                m = np.zeros(n_p)
                teammate_mask = (self.df['Team'] == row['Team']) & (self.df['Pos'].isin(['WR', 'TE']))
                m[teammate_mask] = 1
                m[qidx] = -stack_size 
                A.append(m); bl.append(0); bu.append(10)

            # Exposure Check
            for idx, name in enumerate(self.df['Name']):
                if exposure_counts[name] >= (portfolio_size * max_exposure):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx_list = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx_list].copy().reset_index(drop=True)
                
                # Dynamic Slotting
                qb = l_df[l_df['is_QB']==1].iloc[0]
                dst = l_df[l_df['is_DST']==1].iloc[0]
                rbs = l_df[l_df['is_RB']==1].sort_values('Salary', ascending=False)
                wrs = l_df[l_df['is_WR']==1].sort_values('Salary', ascending=False)
                tes = l_df[l_df['is_TE']==1].sort_values('Salary', ascending=False)
                
                # Logic to identify FLEX
                main_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], tes.iloc[0]['ID'], dst['ID']]
                flex = l_df[~l_df['ID'].isin(main_ids)].iloc[0]

                rost = {
                    'QB': f"{qb['Name']} ({qb['ID']})", 'RB1': f"{rbs.iloc[0]['Name']} ({rbs.iloc[0]['ID']})",
                    'RB2': f"{rbs.iloc[1]['Name']} ({rbs.iloc[1]['ID']})", 'WR1': f"{wrs.iloc[0]['Name']} ({wrs.iloc[0]['ID']})",
                    'WR2': f"{wrs.iloc[1]['Name']} ({wrs.iloc[1]['ID']})", 'WR3': f"{wrs.iloc[2]['Name']} ({wrs.iloc[2]['ID']})",
                    'TE': f"{tes.iloc[0]['Name']} ({tes.iloc[0]['ID']})", 'FLEX': f"{flex['Name']} ({flex['ID']})", 'DST': f"{dst['Name']} ({dst['ID']})"
                }

                if not any(set(l_df['Name'].tolist()) == set(f['Names']) for f in final_lineups):
                    for name in l_df['Name']: exposure_counts[name] += 1
                    final_lineups.append({**rost, 'Score': round(sim[idx_list].sum(), 2), 'Sal': int(l_df['Salary'].sum()), 'Names': l_df['Name'].tolist()})

        bar.empty()
        return final_lineups

# --- UI COMMAND CENTER ---
st.sidebar.markdown("### ⚙️ DIVISIONAL STRATEGY")
stack_val = st.sidebar.selectbox("QB Stack Size", options=[1, 2], index=0)
sim_target = st.sidebar.select_slider("Sim Volume", options=[100, 500, 1000, 2000], value=500)
port_size = st.sidebar.slider("Lineups", 1, 150, 20)
exp_limit = st.sidebar.slider("Max Exposure (%)", 10, 100, 60) / 100.0

st.title("🏈 VANTAGE 99 | DIVISIONAL LAB")
st.markdown(f"<span class='pulse'></span> **STATUS:** Institutional Optimizer Active for Jan 18 Slate", unsafe_allow_html=True)

f = st.file_uploader("LOAD PROJECTIONS / DK SALARIES", type="csv")
if f:
    df_raw = load_and_clean(f)
    try:
        engine = NFLIndustrialOptimizer(df_raw)
        if st.button("🚀 COOK DIVISIONAL PORTFOLIO"):
            start = time.time()
            portfolio = engine.generate_portfolio(n_sims=sim_target, portfolio_size=port_size, max_exposure=exp_limit, stack_size=stack_val)
            duration = time.time() - start
            
            if portfolio:
                st.success(f"🏆 PORTFOLIO VERIFIED: {len(portfolio)} Optimized Lineups Produced")
                top = portfolio[0]
                st.markdown("<div class='roster-card'>", unsafe_allow_html=True)
                st.subheader("🥇 PRIMARY ALPHA ASSEMBLY")
                c1, c2, c3 = st.columns(3)
                c1.metric("SIM SCORE", top['Score'])
                c2.metric("SALARY", f"${top['Sal']}")
                c3.metric("TIME", f"{round(duration, 2)}s")
                
                grid = st.columns(3)
                slots = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
                for i, s in enumerate(slots): grid[i % 3].markdown(f"**{s}** \n{top.get(s, 'Unassigned')}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                t1, t2 = st.tabs(["📊 EXPOSURE", "📥 EXPORT"])
                with t1:
                    all_names = [n for p in portfolio for n in p['Names']]
                    exp_df = pd.Series(all_names).value_counts().reset_index()
                    exp_df.columns = ['Player', 'Lineups']
                    exp_df['Exposure %'] = (exp_df['Lineups'] / len(portfolio)) * 100
                    st.dataframe(exp_df, use_container_width=True)
                with t2:
                    export_cols = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']
                    st.download_button("📥 Download DK CSV", pd.DataFrame(portfolio)[export_cols].to_csv(index=False), "Vantage_Jan18.csv")
    except Exception as e:
        st.error(f"ENGINE ERROR: {e}")
