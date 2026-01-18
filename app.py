import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime
import time

# --- VANTAGE 99: NFL GPP ENGINE (V42.0 - TOURNAMENT CORRELATION) ---
st.set_page_config(page_title="VANTAGE 99 | NFL PORTFOLIO LAB", layout="wide", page_icon="🏈")

# High-end NFL DFS aesthetics
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
    for i, row in df.head(15).iterrows():
        vals = [str(v).lower() for v in row.values]
        if 'name' in vals and 'salary' in vals:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            new_df.columns = [str(c).strip() for c in new_df.columns]
            return new_df.reset_index(drop=True)
    return df

class NFLIndustrialOptimizer:
    def __init__(self, df):
        # NFL Position Mapping (DK Standard: QB, RB, WR, TE, FLEX, DST)
        for p in ['QB','RB','WR','TE','DST']: 
            df[f'is_{p}'] = (df['Position'] == p).astype(int)
        
        # Flex Eligibility (RB/WR/TE)
        df['is_FLEX'] = df['Position'].isin(['RB','WR','TE']).astype(int)
        
        # Data Sanitization
        df['Proj'] = pd.to_numeric(df['ProjPts'], errors='coerce').fillna(0.0)
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(50000)
        
        # Game Theory: Extract Opponent for Correlation
        def get_opp(x):
            m = re.search(r'([A-Z]+)@([A-Z]+)', str(x))
            if m:
                teams = [m.group(1), m.group(2)]
                return teams
            return [None, None]
        
        df[['TeamA', 'TeamB']] = pd.DataFrame(df['Game Info'].apply(get_opp).tolist(), index=df.index)
        df['Opponent'] = df.apply(lambda x: x['TeamB'] if x['TeamAbbrev'] == x['TeamA'] else x['TeamA'], axis=1)

        # GPP SCRUB: Blacklist injured or low-ceiling players for tomorrow
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
            
            # SIMULATION NODE (Sabersim Style Range of Outcomes)
            sim = np.random.normal(proj_vals, proj_vals * jitter).clip(min=0)
            
            A, bl, bu = [], [], []
            # 1. Total Roster (DK = 9 spots)
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            # 2. Salary Cap ($50k)
            A.append(sal_vals); bl.append(48500.0); bu.append(50000.0)
            
            # 3. Positional Requirements
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3) # Min 2, Max 3 (Flex)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4) # Min 3, Max 4 (Flex)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(2) # Min 1, Max 2 (Flex)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)
            
            # 4. Exposure Governor
            for idx, name in enumerate(self.df['Name']):
                if exposure_counts[name] >= (portfolio_size * max_exposure):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            # 5. QB STACKING & GAME THEORY (Linear Constraints)
            # For every QB, if selected, must have 'stack_size' teammates (WR/TE)
            for qidx, row in self.df[self.df['is_QB'] == 1].iterrows():
                m = np.zeros(n_p)
                # Teammates (WR/TE)
                teammate_mask = (self.df['TeamAbbrev'] == row['TeamAbbrev']) & (self.df['Position'].isin(['WR', 'TE']))
                m[teammate_mask] = 1
                m[qidx] = -stack_size 
                A.append(m); bl.append(0); bu.append(9)

            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx_list = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx_list].copy().reset_index(drop=True)
                
                if not any(set(l_df['Name'].tolist()) == set(f['Names']) for f in final_lineups):
                    # SLOT ASSIGNMENT LOGIC
                    rost = {}
                    qb = l_df[l_df['is_QB']==1].iloc[0]
                    rbs = l_df[l_df['is_RB']==1]
                    wrs = l_df[l_df['is_WR']==1]
                    tes = l_df[l_df['is_TE']==1]
                    dst = l_df[l_df['is_DST']==1].iloc[0]
                    
                    rost['QB'] = f"{qb['Name']} ({qb['ID']})"
                    rost['RB1'] = f"{rbs.iloc[0]['Name']} ({rbs.iloc[0]['ID']})"
                    rost['RB2'] = f"{rbs.iloc[1]['Name']} ({rbs.iloc[1]['ID']})"
                    rost['WR1'] = f"{wrs.iloc[0]['Name']} ({wrs.iloc[0]['ID']})"
                    rost['WR2'] = f"{wrs.iloc[1]['Name']} ({wrs.iloc[1]['ID']})"
                    rost['WR3'] = f"{wrs.iloc[2]['Name']} ({wrs.iloc[2]['ID']})"
                    rost['TE'] = f"{tes.iloc[0]['Name']} ({tes.iloc[0]['ID']})"
                    rost['DST'] = f"{dst['Name']} ({dst['ID']})"
                    
                    # FLEX Logic: The player not in the primary 7 + DST
                    assigned_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], tes.iloc[0]['ID'], dst['ID']]
                    flex = l_df[~l_df['ID'].isin(assigned_ids)].iloc[0]
                    rost['FLEX'] = f"{flex['Name']} ({flex['ID']})"

                    for name in l_df['Name']: exposure_counts[name] += 1
                    final_lineups.append({**rost, 'Score': round(sim[idx_list].sum(), 2), 'Sal': int(l_df['Salary'].sum()), 'Names': l_df['Name'].tolist()})

        bar.empty()
        return final_lineups

# --- UI COMMAND CENTER ---
st.sidebar.markdown("### ⚙️ GPP STRATEGY")
stack_val = st.sidebar.selectbox("QB Stack Size (Teammates)", options=[1, 2], index=0)
sim_target = st.sidebar.select_slider("Sim Volume (SaberSim Level)", options=[100, 500, 1000, 5000], value=500)
port_size = st.sidebar.slider("Lineups to Generate", 1, 150, 20)
exp_limit = st.sidebar.slider("Max Player Exposure (%)", 10, 100, 50) / 100.0

st.title("🏈 VANTAGE 99 | NFL PORTFOLIO LAB")
st.markdown(f"<span class='pulse'></span> **STATUS:** GPP Correlation Engine Active (Vegas Targeting)", unsafe_allow_html=True)

f = st.file_uploader("LOAD DRAFTKINGS SALARIES / STOKASTIC PROJECTIONS", type="csv")
if f:
    df_raw = load_and_clean(f)
    engine = NFLIndustrialOptimizer(df_raw)
    if st.button("🚀 COOK TOURNAMENT PORTFOLIO"):
        start = time.time()
        portfolio = engine.generate_portfolio(n_sims=sim_target, portfolio_size=port_size, max_exposure=exp_limit, stack_size=stack_val)
        duration = time.time() - start
        
        if portfolio:
            st.success(f"🏆 PORTFOLIO VERIFIED: {len(portfolio)} Optimized GPP Lineups Produced")
            top = portfolio[0]
            st.markdown("<div class='roster-card'>", unsafe_allow_html=True)
            st.subheader("🥇 PRIMARY ALPHA ASSEMBLY (Lineup 1)")
            c1, c2, c3 = st.columns(3)
            c1.metric("PROJECTED SCORE", top['Score'])
            c2.metric("SALARY USED", f"${top['Sal']}")
            c3.metric("ENGINE LATENCY", f"{round(duration, 2)}s")
            
            grid = st.columns(3)
            slots = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
            for i, s in enumerate(slots): grid[i % 3].markdown(f"**{s}** \n{top.get(s, 'Unassigned')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            t1, t2 = st.tabs(["📊 EXPOSURE AUDIT", "📥 BATCH EXPORT"])
            with t1:
                all_names = [n for p in portfolio for n in p['Names']]
                exp_df = pd.Series(all_names).value_counts().reset_index()
                exp_df.columns = ['Player', 'Lineups']
                exp_df['Exposure %'] = (exp_df['Lineups'] / len(portfolio)) * 100
                st.bar_chart(exp_df.set_index('Player')['Exposure %'])
                st.dataframe(exp_df, use_container_width=True)
            with t2:
                export_cols = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']
                st.download_button("📥 Download DK Bulk Entry File", pd.DataFrame(portfolio)[export_cols].to_csv(index=False), "NFL_Vantage_Portfolio.csv")
