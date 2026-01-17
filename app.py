import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime
import time

# --- VANTAGE 99: INDUSTRIAL NBA SIMULATOR (V30.0) ---
st.set_page_config(page_title="VANTAGE NBA LAB", layout="wide", page_icon="🏀")

# --- INDUSTRIAL UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0b0e11; color: #e6e6e6; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; border-bottom-color: #00ffcc !important; }
    .stButton>button { background-color: #00ffcc; color: black; font-weight: bold; width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

def deep_scan(df):
    for i, row in df.head(15).iterrows():
        vals = [str(v).lower() for v in row.values]
        if 'name' in vals and 'salary' in vals:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

class VantageSimulator:
    def __init__(self, s_df):
        s_df = deep_scan(s_df)
        s_df.columns = [str(c).strip() for c in s_df.columns]
        
        # 1. POSITION MAP
        for p in ['PG','SG','SF','PF','C']: s_df[f'is_{p}'] = s_df['Position'].str.contains(p).astype(int)
        s_df['is_G'] = ((s_df['is_PG']==1)|(s_df['is_SG']==1)).astype(int)
        s_df['is_F'] = ((s_df['is_SF']==1)|(s_df['is_PF']==1)).astype(int)
        
        # 2. TIME & PROJECTIONS
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        s_df['Time'] = s_df['Game Info'].apply(get_time)
        s_df['AvgPointsPerGame'] = pd.to_numeric(s_df['AvgPointsPerGame'], errors='coerce').fillna(10.0)
        s_df['Proj'] = s_df['AvgPointsPerGame'].clip(lower=5.0)
        
        # 3. INDUSTRIAL INJURY PURGE (Jan 17 Hardened)
        scrubbed = [
            'Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green', 
            'Andrew Nembhard', 'Jonas Valanciunas', 'Isaiah Hartenstein', 
            'Jaime Jaquez Jr.', 'Devin Vassell', 'Moussa Diabate', 
            'Christian Braun', 'T.J. McConnell', 'Zaccharie Risacher', 
            'Davion Mitchell', 'Jamaree Bouyea', 'Jayson Tatum', 'Cameron Johnson'
        ]
        self.df = s_df[~s_df['Name'].isin(scrubbed)].reset_index(drop=True)

    def run_sims(self, n_sims=1000, jitter=0.20, min_unique=3):
        n_p = len(self.df)
        proj_vals, sal_vals = self.df['Proj'].values.astype(float), self.df['Salary'].values.astype(float)
        
        # Main Solver Constraints
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8); bu.append(8)
        A.append(sal_vals); bl.append(49200.0); bu.append(50000.0)
        for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: A.append(self.df[c].values.astype(float)); bl.append(1.0); bu.append(8.0)
        A.append(self.df['is_G'].values.astype(float)); bl.append(3.0); bu.append(8.0)
        A.append(self.df['is_F'].values.astype(float)); bl.append(3.0); bu.append(8.0)

        raw_pool = []
        progress = st.progress(0)
        for i in range(n_sims):
            if i % (n_sims//20) == 0: progress.progress(i / n_sims)
            sim = np.random.normal(proj_vals, proj_vals * jitter).clip(min=0)
            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                raw_pool.append({'idx': set(idx), 'score': sim[idx].sum()})
        
        # PORTFOLIO SELECTION & MOLECULAR SLOTTING
        raw_pool = sorted(raw_pool, key=lambda x: x['score'], reverse=True)
        final = []
        slots = ['PG','SG','SF','PF','C','G','F','UTIL']
        
        for item in raw_pool:
            if len(final) >= 20: break
            if not any(len(item['idx'] & f['idx']) > (8 - min_unique) for f in final):
                l_df = self.df.iloc[list(item['idx'])].copy().reset_index(drop=True)
                latest = l_df['Time'].max()
                
                # --- ASSIGNMENT ENGINE (Fixes IndexError) ---
                M, cost = np.zeros((8, 8)), np.zeros((8, 8))
                for pi, p in l_df.iterrows():
                    for si, c in enumerate(['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']):
                        if p[c]: M[pi, si] = 1
                    M[pi, 7] = 1 # UTIL
                    if p['Time'] == latest: cost[pi, 7] = -1000 # Lock latest player to UTIL
                
                res_as = milp(c=cost.flatten(), constraints=LinearConstraint(np.zeros((1, 64)), [0], [0]), 
                             integrality=np.ones(64), bounds=Bounds(0, M.flatten()))
                
                if res_as.success:
                    mapping = res_as.x.reshape((8, 8))
                    rost = {slots[j]: f"{l_df.iloc[i]['Name']} ({l_df.iloc[i]['ID']})" for i in range(8) for j in range(8) if mapping[i,j]>0.5}
                    final.append({**rost, 'Score': item['score'], 'idx': item['idx'], 'Names': l_df['Name'].tolist()})
        
        progress.empty()
        return final

# --- UI COMMAND CENTER ---
st.title("🏀 VANTAGE 99 INDUSTRIAL SIMULATOR")
st.sidebar.header("⚙️ SIM SETTINGS")
n_sims = st.sidebar.select_slider("Simulations", options=[100, 500, 1000, 5000], value=1000)

f = st.file_uploader("Upload Salary CSV", type="csv")
if f:
    engine = VantageSimulator(pd.read_csv(f))
    if st.button(f"🚀 EXECUTE {n_sims} INDUSTRIAL SIMULATIONS"):
        results = engine.run_sims(n_sims=n_sims)
        if results:
            st.success(f"🏆 TOP ALPHA LINEUP IDENTIFIED (Winner of {n_sims} Simulations)")
            st.table(pd.DataFrame([results[0]])[['PG','SG','SF','PF','C','G','F','UTIL']])
            
            t1, t2 = st.tabs(["📊 Exposure Report", "📥 Download Batch"])
            with t1:
                exp = pd.Series([p for r in results for p in r['Names']]).value_counts().reset_index()
                st.bar_chart(exp.set_index('index')['count'])
            with t2:
                st.download_button("Download CSV", pd.DataFrame(results).drop(columns=['Score','idx','Names']).to_csv(index=False), "NBA_Alpha.csv")
