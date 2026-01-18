import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime
import time

# --- VANTAGE 99: INSTITUTIONAL PORTFOLIO ENGINE (V37.0 - MOLECULAR POSITION FIX) ---
st.set_page_config(page_title="VANTAGE 99 | PORTFOLIO LAB", layout="wide", page_icon="🏀")

# Custom CSS for high-end DFS aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] { background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d; border-radius: 12px; padding: 15px; }
    .roster-card { background: linear-gradient(145deg, #161b22, #0d1117); border: 1px solid #00ffcc; border-radius: 12px; padding: 20px; margin: 10px 0px; }
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

class IndustrialOptimizer:
    def __init__(self, df):
        # Industrial-grade position mapping
        for p in ['PG','SG','SF','PF','C']: 
            df[f'is_{p}'] = df['Position'].str.contains(p).astype(int)
        df['is_G'] = ((df['is_PG']==1)|(df['is_SG']==1)).astype(int)
        df['is_F'] = ((df['is_SF']==1)|(df['is_PF']==1)).astype(int)
        
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        df['Time'] = df['Game Info'].apply(get_time)
        df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        
        # Current Scrub (Jan 17 Official - Multi-node purge)
        scrubbed = ['Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green', 'Andrew Nembhard', 'Jonas Valanciunas', 'Isaiah Hartenstein', 'Jaime Jaquez Jr.', 'Devin Vassell', 'Moussa Diabate', 'Christian Braun', 'T.J. McConnell', 'Zaccharie Risacher', 'Davion Mitchell', 'Jamaree Bouyea', 'Jayson Tatum', 'Cameron Johnson', 'Payton Pritchard', 'Bennedict Mathurin', 'Dyson Daniels', 'Isaiah Jackson', 'Daniel Gafford', 'P.J. Washington', 'Obi Toppin']
        self.df = df[~df['Name'].isin(scrubbed)].reset_index(drop=True)

    def generate_portfolio(self, n_sims=500, portfolio_size=20, max_exposure=0.5, jitter=0.20):
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
            A.append(np.ones(n_p)); bl.append(8); bu.append(8)
            A.append(sal_vals); bl.append(49200.0); bu.append(50000.0)
            for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: A.append(self.df[c].values.astype(float)); bl.append(1.0); bu.append(8.0)
            A.append(self.df['is_G'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            A.append(self.df['is_F'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            
            # Exposure Governor
            for idx, name in enumerate(self.df['Name']):
                if exposure_counts[name] >= (portfolio_size * max_exposure):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx_list = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx_list].copy().reset_index(drop=True)
                if not any(set(l_df['Name'].tolist()) == set(f['Names']) for f in final_lineups):
                    # --- FIXED MOLECULAR SLOTTING (Linear Assignment Problem) ---
                    latest = l_df['Time'].max()
                    slots = ['PG','SG','SF','PF','C','G','F','UTIL']
                    conds = ['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']
                    
                    # Cost matrix C[player, slot]
                    C = np.zeros((8, 8))
                    for pi, p in l_df.iterrows():
                        for si, cond in enumerate(conds):
                            if getattr(p, cond) == 1: C[pi, si] = -10
                        C[pi, 7] = -1 # All fit in UTIL
                        if p.Time == latest: C[pi, 7] = -100 # Bias UTIL to latest

                    A_as, bl_as, bu_as = [], [], []
                    for r in range(8): m = np.zeros((8, 8)); m[r, :]=1; A_as.append(m.flatten()); bl_as.append(1); bu_as.append(1)
                    for c in range(8): m = np.zeros((8, 8)); m[:, c]=1; A_as.append(m.flatten()); bl_as.append(1); bu_as.append(1)
                    
                    res_as = milp(c=C.flatten(), constraints=LinearConstraint(A_as, bl_as, bu_as), integrality=np.ones(64), bounds=Bounds(0, 1))
                    
                    if res_as.success:
                        map_res = res_as.x.reshape((8, 8))
                        rost = {}
                        for pi in range(8):
                            for si in range(8):
                                if map_res[pi, si] > 0.5:
                                    p_match = l_df.iloc[pi]
                                    rost[slots[si]] = f"{p_match['Name']} ({p_match['ID']})"
                        
                        for name in l_df['Name']: exposure_counts[name] += 1
                        final_lineups.append({**rost, 'Score': round(sim[idx_list].sum(), 2), 'Sal': int(l_df['Salary'].sum()), 'Names': l_df['Name'].tolist()})

        bar.empty()
        return final_lineups

# --- UI COMMAND CENTER ---
st.sidebar.markdown("### ⚙️ PORTFOLIO STRATEGY")
sim_target = st.sidebar.select_slider("Sim Volume", options=[100, 500, 1000, 5000], value=500)
port_size = st.sidebar.slider("Lineups to Generate", 1, 150, 20)
exp_limit = st.sidebar.slider("Max Player Exposure (%)", 10, 100, 50) / 100.0

st.title("🏀 VANTAGE 99 | PORTFOLIO LAB")
st.markdown(f"<span class='pulse'></span> **STATUS:** Institutional Rebalancing Engine Active", unsafe_allow_html=True)

f = st.file_uploader("LOAD RAW DATA", type="csv")
if f:
    df_raw = load_and_clean(f)
    engine = IndustrialOptimizer(df_raw)
    if st.button("🚀 COOK INDUSTRIAL PORTFOLIO"):
        start = time.time()
        portfolio = engine.generate_portfolio(n_sims=sim_target, portfolio_size=port_size, max_exposure=exp_limit)
        duration = time.time() - start
        
        if portfolio:
            st.success(f"🏆 PORTFOLIO VERIFIED: {len(portfolio)} Optimized Lineups Produced")
            top = portfolio[0]
            st.markdown("<div class='roster-card'>", unsafe_allow_html=True)
            st.subheader("🥇 PRIMARY ALPHA ASSEMBLY")
            c1, c2, c3 = st.columns(3)
            c1.metric("SIM SCORE", top['Score'])
            c2.metric("SALARY", f"${top['Sal']}")
            c3.metric("SIM TIME", f"{round(duration, 2)}s")
            
            grid = st.columns(4)
            slots = ['PG','SG','SF','PF','C','G','F','UTIL']
            for i, s in enumerate(slots): grid[i % 4].markdown(f"**{s}** \n{top.get(s, 'Unassigned')}")
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
                st.download_button("📥 Download DK Bulk Entry File", pd.DataFrame(portfolio)[slots].to_csv(index=False), "Vantage_Portfolio.csv")
