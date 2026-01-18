import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, linear_sum_assignment
import re
from datetime import datetime
import time

# --- VANTAGE 99 | V44.0: NBA ALPHA-MME (MOLECULAR FLEX FIX) ---
st.set_page_config(page_title="VANTAGE 99 | NBA ALPHA", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] { background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d; border-radius: 12px; padding: 15px; }
    .roster-card { background: linear-gradient(145deg, #161b22, #0d1117); border: 1px solid #00ffcc; border-radius: 12px; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean(file):
    try:
        df = pd.read_csv(file)
        for i, row in df.head(15).iterrows():
            vals = [str(v).lower() for v in row.values]
            if 'name' in vals and 'salary' in vals:
                new_df = df.iloc[i+1:].copy()
                new_df.columns = [str(c).strip() for c in df.iloc[i].values]
                return new_df.reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"File Load Error: {e}")
        return None

class AlphaOptimizer:
    def __init__(self, df):
        # 1. Position Mapping (NBA Specific)
        for p in ['PG','SG','SF','PF','C']: 
            df[f'is_{p}'] = df['Position'].str.contains(p).astype(int)
        df['is_G'] = ((df['is_PG']==1)|(df['is_SG']==1)).astype(int)
        df['is_F'] = ((df['is_SF']==1)|(df['is_PF']==1)).astype(int)
        
        # 2. Time Logic for UTIL Priority
        def get_time(x):
            m = re.search(r'(\d{1,2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        df['Time'] = df['Game Info'].apply(get_time)
        
        # 3. Clean Projections & Ownership
        df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(10.0)
        if 'Ownership' in df.columns:
            df['Own'] = pd.to_numeric(df['Ownership'], errors='coerce').fillna(15.0)
        else:
            df['Own'] = 15.0
        
        # 4. Scrub List (Update for 1/18 slate)
        scrubbed = ['Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green'] 
        self.df = df[~df['Name'].isin(scrubbed)].reset_index(drop=True)

    def run_alpha_sims(self, n_sims=5000, own_cap=125, jitter=0.20):
        n_p = len(self.df)
        proj_vals = self.df['Proj'].values.astype(float)
        sal_vals = self.df['Salary'].values.astype(float)
        own_vals = self.df['Own'].values.astype(float)
        
        # Pre-build MILP Constraints
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8); bu.append(8)
        A.append(sal_vals); bl.append(49700); bu.append(49900)
        A.append(own_vals); bl.append(0); bu.append(own_cap)
        for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: 
            A.append(self.df[c].values.astype(float)); bl.append(1); bu.append(8)
        A.append(self.df['is_G'].values.astype(float)); bl.append(3); bu.append(8)
        A.append(self.df['is_F'].values.astype(float)); bl.append(3); bu.append(8)
        A_stack = np.vstack(A)

        lineup_counts = {}
        status = st.empty()
        
        # High-Speed Simulation Loop
        for i in range(1, n_sims + 1):
            if i % 500 == 0: status.write(f"🔬 `SIMULATING SLATE:` {i}/{n_sims}...")
            
            # Stud Jitter (Higher variance for $9k+ players)
            dyn_jitter = np.where(sal_vals >= 9000, jitter * 1.5, jitter)
            sim = np.random.normal(proj_vals, proj_vals * dyn_jitter).clip(min=0)
            
            res = milp(c=-sim, constraints=LinearConstraint(A_stack, bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1), 
                       options={'mip_rel_gap': 0.05, 'presolve': True})
            
            if res.success:
                ids = tuple(sorted(np.where(res.x > 0.5)[0]))
                lineup_counts[ids] = lineup_counts.get(ids, 0) + 1

        status.empty()
        top_winners = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        final_pool = []
        for ids, freq in top_winners:
            ldf = self.df.iloc[list(ids)].copy()
            # --- THE FIX: MOLECULAR SLOT ASSIGNMENT ---
            slots = ['PG','SG','SF','PF','C','G','F','UTIL']
            conds = ['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']
            
            cost_matrix = np.full((8, 8), 1000.0) 
            latest_time = ldf['Time'].max()
            
            for pi, (_, p) in enumerate(ldf.iterrows()):
                for si, cond in enumerate(conds):
                    if p[cond] == 1: cost_matrix[pi, si] = 10
                cost_matrix[pi, 7] = 50 
                if p['Time'] == latest_time:
                    cost_matrix[pi, 7] = 0 # Priority for UTIL
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            rost = { "Win %": f"{round((freq/n_sims)*100, 2)}%", "Salary": ldf['Salary'].sum() }
            # Sort indices to match slot order
            for r, c in zip(row_ind, col_ind):
                rost[slots[c]] = ldf.iloc[r]['Name']
            final_pool.append(rost)
            
        return final_pool

# --- UI COMMAND CENTER ---
st.title("🏀 VANTAGE 99 | NBA ALPHA LAB")
st.sidebar.header("SIMULATION ENGINE")
sim_count = st.sidebar.select_slider("Sim Volume", options=[500, 1000, 5000, 10000], value=5000)
own_fade = st.sidebar.slider("Ownership Cap", 80, 180, 120)

f = st.file_uploader("LOAD DK SALARY CSV", type="csv")
if f:
    df_raw = load_and_clean(f)
    if st.button("🚀 EXECUTE ALPHA ANALYSIS"):
        engine = AlphaOptimizer(df_raw)
        results = engine.run_alpha_sims(n_sims=sim_count, own_cap=own_fade)
        if results:
            st.success(f"Top {len(results)} Optimals Identified")
            st.table(pd.DataFrame(results)[['Win %', 'Salary', 'PG','SG','SF','PF','C','G','F','UTIL']])
