import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime
import time

# --- VANTAGE 99: INDUSTRIAL NBA SIMULATOR (V29.0) ---
st.set_page_config(page_title="VANTAGE NBA LAB", layout="wide", page_icon="🏀")

# --- UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0b0e11; color: #e6e6e6; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
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

class VantageSimulator:
    def __init__(self, s_df):
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
        
        # INDUSTRIAL INJURY PURGE
        scrubbed = ['Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green', 'Andrew Nembhard', 
                   'Jonas Valanciunas', 'Isaiah Hartenstein', 'Jaime Jaquez Jr.', 'Devin Vassell', 
                   'Moussa Diabate', 'Christian Braun', 'T.J. McConnell', 'Zaccharie Risacher', 
                   'Davion Mitchell', 'Jamaree Bouyea', 'Jayson Tatum', 'Cameron Johnson']
        self.df = s_df[~s_df['Name'].isin(scrubbed)].reset_index(drop=True)

    def run_alpha_sims(self, n_sims=1000, jitter=0.20, min_unique=3):
        n_p = len(self.df)
        proj_vals, sal_vals = self.df['Proj'].values.astype(float), self.df['Salary'].values.astype(float)
        
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8); bu.append(8)
        A.append(sal_vals); bl.append(49200.0); bu.append(50000.0)
        for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: A.append(self.df[c].values.astype(float)); bl.append(1.0); bu.append(8.0)
        A.append(self.df['is_G'].values.astype(float)); bl.append(3.0); bu.append(8.0)
        A.append(self.df['is_F'].values.astype(float)); bl.append(3.0); bu.append(8.0)

        raw_pool = []
        progress = st.progress(0)
        status = st.empty()
        
        for i in range(n_sims):
            if i % (n_sims//10) == 0: 
                progress.progress(i / n_sims)
                status.text(f"Industrial Simulation in progress... {i}/{n_sims}")
            
            sim = np.random.normal(proj_vals, proj_vals * jitter).clip(min=0)
            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                raw_pool.append({'idx': set(idx), 'score': sim[idx].sum()})
        
        # DIVERSITY SORT (Fixed KeyError Logic)
        raw_pool = sorted(raw_pool, key=lambda x: x['score'], reverse=True)
        final_lineups = []
        
        for item in raw_pool:
            if len(final_lineups) >= 20: break
            # Logic: Ensure new lineup doesn't overlap too much with already selected ones
            if not any(len(item['idx'] & f['idx']) > (8 - min_unique) for f in final_lineups):
                l_df = self.df.iloc[list(item['idx'])]
                latest = l_df['Time'].max()
                
                # Assembly logic for PG,SG,SF,PF,C,G,F,UTIL
                rost = {}
                p_pool = l_df.copy()
                for slot, cond in [('PG','is_PG'),('SG','is_SG'),('SF','is_SF'),('PF','is_PF'),('C','is_C'),('G','is_G'),('F','is_F')]:
                    match = p_pool[(p_pool[cond]==1) & (p_pool['Time'] != latest)].sort_values('Proj', ascending=False).head(1)
                    if match.empty: match = p_pool[p_pool[cond]==1].sort_values('Proj', ascending=False).head(1)
                    rost[slot] = f"{match.iloc[0]['Name']} ({match.iloc[0]['ID']})"
                    p_pool = p_pool.drop(match.index)
                rost['UTIL'] = f"{p_pool.iloc[0]['Name']} ({p_pool.iloc[0]['ID']})"
                
                final_lineups.append({**rost, 'Score': round(item['score'], 2), 'Salary': int(l_df['Salary'].sum()), 'Players': l_df['Name'].tolist(), 'idx': item['idx']})
        
        progress.empty()
        status.empty()
        return final_lineups

# --- UI ---
st.title("🏀 VANTAGE 99 INDUSTRIAL SIMULATOR")
st.sidebar.header("⚙️ SIM SETTINGS")
n_sims = st.sidebar.select_slider("Simulations", options=[100, 500, 1000, 5000], value=1000)

f = st.file_uploader("Upload DraftKings Salary CSV", type="csv")
if f:
    engine = VantageSimulator(pd.read_csv(f))
    if st.button(f"🚀 EXECUTE {n_sims} SIMULATIONS"):
        results = engine.run_alpha_sims(n_sims=n_sims)
        if results:
            st.success(f"🏆 TOP ALPHA LINEUP IDENTIFIED")
            st.table(pd.DataFrame([results[0]])[['PG','SG','SF','PF','C','G','F','UTIL']])
            
            t1, t2 = st.tabs(["📊 Exposure Report", "📥 Download Batch"])
            with t1:
                all_p = [p for r in results for p in r['Players']]
                exp = pd.Series(all_p).value_counts().reset_index()
                exp.columns = ['Player', 'Count']
                st.bar_chart(exp.set_index('Player')['Count'])
            with t2:
                st.download_button("Download CSV", pd.DataFrame(results).drop(columns=['Score','Salary','Players','idx']).to_csv(index=False), "NBA_Alpha_Batch.csv")
