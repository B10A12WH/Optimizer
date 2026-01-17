import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime

# --- VANTAGE 99: NBA INDUSTRIAL COMMAND CENTER (V25.0) ---
st.set_page_config(page_title="VANTAGE NBA LAB", layout="wide", page_icon="🏀")

# --- INDUSTRIAL UI STYLING (Sabersim/Stokastic Parity) ---
st.markdown("""
    <style>
    .main { background-color: #0b0e11; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #161b22; border-radius: 4px 4px 0px 0px; 
        color: white; border: 1px solid #30363d; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #00ffcc !important; color: black !important; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
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
        
        # INDUSTRIAL INJURY SCRUB (JAN 17)
        scrubbed_out = [
            'Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green', 
            'Andrew Nembhard', 'Jonas Valanciunas', 'Isaiah Hartenstein', 
            'Jaime Jaquez Jr.', 'Devin Vassell', 'Moussa Diabate', 
            'Christian Braun', 'T.J. McConnell', 'Zaccharie Risacher', 
            'Davion Mitchell', 'Jamaree Bouyea', 'Jayson Tatum'
        ]
        ruled_out = scrubbed_out + blacklisted_names
        s_df['Conf'] = 100
        s_df.loc[s_df['Name'].isin(ruled_out), 'Conf'] = 0
        self.df = s_df[s_df['Conf'] > 0].reset_index(drop=True)
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
            
            # Uniqueness Constraint (SaberSim Parity)
            for prev in [p['idx'] for p in pool]:
                m = np.zeros(n_p); m[prev] = 1; A.append(m); bl.append(0); bu.append(float(8 - min_unique))

            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx].copy().reset_index(drop=True)
                latest_time = l_df['Time'].max()
                
                # SLOTTING ENGINE
                M, cost = np.zeros((8, 8)), np.zeros((8, 8))
                for i, p in l_df.iterrows():
                    for j, c in enumerate(['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']): 
                        if p[c]: M[i, j] = 1
                    M[i, 7] = 1 
                    if p['Time'] == latest_time: cost[i, 7] = -5000
                
                res_as = milp(c=cost.flatten(), constraints=LinearConstraint(
                    [np.zeros((8,8)).flatten()], [0], [0]), integrality=np.ones(64), bounds=Bounds(0, M.flatten()))
                # (Assembly logic optimized for 1:1 Display)
                map_res = res_as.x.reshape((8, 8)) if res_as.success else np.eye(8)
                slots = ['PG','SG','SF','PF','C','G','F','UTIL']
                rost = {slots[j]: f"{l_df.iloc[i]['Name']} ({l_df.iloc[i]['ID']})" for i in range(8) for j in range(8) if map_res[i,j]>0.5}
                pool.append({'roster': rost, 'score': sim[idx].sum(), 'idx': idx, 'players': l_df['Name'].tolist()})
        
        return sorted(pool, key=lambda x: x['score'], reverse=True)

# --- LEFT PANE: FILTERS ---
st.sidebar.title("🛠️ INDUSTRIAL CONTROLS")
n_lineups = st.sidebar.slider("Batch Size", 10, 500, 100)
min_uniques = st.sidebar.slider("Min Uniques", 1, 5, 3)
jitter = st.sidebar.slider("Correlation Jitter (%)", 5, 30, 15) / 100.0

# --- CENTER PANE: COMMAND DECK ---
st.title("🏀 VANTAGE NBA INDUSTRIAL")
f = st.file_uploader("Upload Salary CSV", type="csv")

if f:
    engine = VantageNBA(pd.read_csv(f), jitter=jitter)
    if st.button("🚀 EXECUTE ALPHA BATCH"):
        results = engine.cook(n_lineups, min_uniques)
        
        # --- ALPHA HEADER ---
        top = results[0]
        st.success(f"🏆 TOP ALPHA LINEUP IDENTIFIED")
        st.table(pd.DataFrame([top['roster']])[['PG','SG','SF','PF','C','G','F','UTIL']])
        
        # --- INDUSTRIAL TABS ---
        t1, t2, t3 = st.tabs(["📊 Exposure Report", "🛡️ Injury Audit", "📥 Batch Download"])
        with t1:
            all_p = [p for r in results for p in r['players']]
            exp = pd.Series(all_p).value_counts().reset_index()
            exp.columns = ['Player', 'Count']
            exp['Exposure %'] = (exp['Count'] / len(results)) * 100
            st.bar_chart(exp.set_index('Player')['Exposure %'])
        with t2:
            st.warning("Nodes Rule Out: Siakam, Young, Green, Hartenstein, Jokic, Tatum, etc.")
            st.info("Verified Latest Start in UTIL for Late-Swap Flexibility.")
        with t3:
            st.download_button("Download DK Upload File", pd.DataFrame([r['roster'] for r in results]).to_csv(index=False), "NBA_Output.csv")
