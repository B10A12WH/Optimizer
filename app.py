import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime

# --- VANTAGE 99: INDUSTRIAL NBA COMMAND CENTER (V26.0) ---
st.set_page_config(page_title="VANTAGE NBA LAB", layout="wide", page_icon="🏀")

# --- 1:1 UI PARITY CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e11; color: #e6e6e6; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0b0e11; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; border-bottom-color: #00ffcc !important; }
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

class VantageNBA:
    def __init__(self, s_df, blacklisted_names=[], jitter=0.15):
        s_df = deep_scan(s_df)
        s_df.columns = [str(c).strip() for c in s_df.columns]
        for p in ['PG','SG','SF','PF','C']: s_df[f'is_{p}'] = s_df['Position'].str.contains(p).astype(int)
        s_df['is_G'] = ((s_df['is_PG']==1)|(s_df['is_SG']==1)).astype(int)
        s_df['is_F'] = ((s_df['is_SF']==1)|(s_df['is_PF']==1)).astype(int)
        
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        s_df['Time'] = s_df['Game Info'].apply(get_time)
        s_df['Proj'] = pd.to_numeric(s_df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        
        # --- INDUSTRIAL INJURY PURGE ---
        scrubbed_out = ['Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green', 'Andrew Nembhard', 
                       'Jonas Valanciunas', 'Isaiah Hartenstein', 'Jaime Jaquez Jr.', 'Devin Vassell', 
                       'Moussa Diabate', 'Christian Braun', 'T.J. McConnell', 'Zaccharie Risacher', 
                       'Davion Mitchell', 'Jamaree Bouyea', 'Jayson Tatum']
        ruled_out = scrubbed_out + blacklisted_names
        self.df = s_df[~s_df['Name'].isin(ruled_out)].reset_index(drop=True)
        self.jitter = jitter

    def cook(self, n=50, min_unique=3):
        pool = []
        n_p = len(self.df)
        proj_vals, sal_vals = self.df['Proj'].values.astype(float), self.df['Salary'].values.astype(float)
        
        for _ in range(n):
            sim = np.random.normal(proj_vals, proj_vals * self.jitter).clip(min=0)
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(8); bu.append(8)
            A.append(sal_vals); bl.append(49200.0); bu.append(50000.0)
            for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: A.append(self.df[c].values.astype(float)); bl.append(1.0); bu.append(8.0)
            A.append(self.df['is_G'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            A.append(self.df['is_F'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            for prev in [p['idx'] for p in pool]:
                m = np.zeros(n_p); m[prev] = 1; A.append(m); bl.append(0); bu.append(float(8 - min_unique))

            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx].copy().reset_index(drop=True)
                
                # --- DOUBLE-PASS ASSIGNMENT ENGINE (Fixes KeyError) ---
                latest_time = l_df['Time'].max()
                M, cost = np.zeros((8, 8)), np.zeros((8, 8))
                for i, p in l_df.iterrows():
                    for j, c in enumerate(['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']): 
                        if p[c]: M[i, j] = 1
                    M[i, 7] = 1 
                    if p['Time'] == latest_time: cost[i, 7] = -10000
                
                A_as, bl_as, bu_as = [], [], []
                for i in range(8): r=np.zeros((8,8)); r[i,:]=1; A_as.append(r.flatten()); bl_as.append(1); bu_as.append(1)
                for j in range(8): c=np.zeros((8,8)); c[:,j]=1; A_as.append(c.flatten()); bl_as.append(1); bu_as.append(1)
                
                res_as = milp(c=cost.flatten(), constraints=LinearConstraint(A_as, bl_as, bu_as), integrality=np.ones(64), bounds=Bounds(0, M.flatten()))
                if res_as.success:
                    map_res = res_as.x.reshape((8, 8))
                    slots = ['PG','SG','SF','PF','C','G','F','UTIL']
                    rost = {slots[j]: f"{l_df.iloc[i]['Name']} ({l_df.iloc[i]['ID']})" for i in range(8) for j in range(8) if map_res[i,j]>0.5}
                    pool.append({'roster': rost, 'score': sim[idx].sum(), 'idx': idx, 'players': l_df['Name'].tolist()})
        return sorted(pool, key=lambda x: x['score'], reverse=True)

# --- SIDEBAR (SaberSim Parity) ---
st.sidebar.title("🧪 INDUSTRIAL FILTERS")
n_batch = st.sidebar.slider("Batch Size", 10, 500, 100)
jitter = st.sidebar.slider("Sim Variance (%)", 5, 40, 15) / 100.0

# --- MAIN TERMINAL ---
st.title("🏀 VANTAGE NBA COMMAND DECK")
f = st.file_uploader("Upload Salary CSV", type="csv")

if f:
    engine = VantageNBA(pd.read_csv(f), jitter=jitter)
    
    tab1, tab2, tab3 = st.tabs(["📊 PLAYER POOL", "🏆 LINEUPS", "🔍 AUDIT"])
    
    with tab1:
        st.subheader("Industrial Projections Editor")
        edited_df = st.data_editor(engine.df[['Name', 'Position', 'Salary', 'Proj']], num_rows="dynamic", use_container_width=True)
        engine.df['Proj'] = edited_df['Proj'] # Sync back

    with tab2:
        if st.button("🚀 EXECUTE BATCH"):
            results = engine.cook(n_batch)
            if results:
                top = results[0]
                st.success("TOP ALPHA BUILD IDENTIFIED")
                # FAULT TOLERANT TABLE: Uses only keys present in the roster
                st.table(pd.DataFrame([top['roster']])) 
                
                st.subheader("Batch Exposure Report")
                all_p = [p for r in results for p in r['players']]
                exp = pd.Series(all_p).value_counts().reset_index()
                exp.columns = ['Player', 'Count']
                exp['Exposure %'] = (exp['Count'] / len(results)) * 100
                st.bar_chart(exp.set_index('Player')['Exposure %'])
                
                st.download_button("Download CSV", pd.DataFrame([r['roster'] for r in results]).to_csv(index=False), "NBA_Batch.csv")
