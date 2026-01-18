import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime
import time

# --- VANTAGE 99: INSTITUTIONAL ALPHA-MME HYBRID (V41.0) ---
st.set_page_config(page_title="VANTAGE 99 | ALPHA LAB", layout="wide", page_icon="🏀")

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
    df = pd.read_csv(file)
    for i, row in df.head(15).iterrows():
        vals = [str(v).lower() for v in row.values]
        if 'name' in vals and 'salary' in vals:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = [str(c).strip() for c in df.iloc[i].values]
            return new_df.reset_index(drop=True)
    return df

class AlphaMMEEngine:
    def __init__(self, df):
        # 1. Position Logic
        for p in ['PG','SG','SF','PF','C']: df[f'is_{p}'] = df['Position'].str.contains(p).astype(int)
        df['is_G'] = ((df['is_PG']==1)|(df['is_SG']==1)).astype(int)
        df['is_F'] = ((df['is_SF']==1)|(df['is_PF']==1)).astype(int)
        
        # 2. Projections Cleaning
        df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        
        # 3. SAFE Ownership Logic: Fixes the traceback crash
        if 'Ownership' in df.columns:
            df['Own'] = pd.to_numeric(df['Ownership'], errors='coerce').fillna(15.0)
        else:
            df['Own'] = 15.0 # Institutional default if column is missing
        
        # 4. Time Parsing for UTIL Flexibility
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        df['Time'] = df['Game Info'].apply(get_time)
        
        # 5. Scrub List (Update based on the Jan 17 Injury Report)
        # Note: Jayson Tatum and Tyrese Haliburton are confirmed OUT today.
        scrubbed = ['Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green', 'Jayson Tatum', 'Tyrese Haliburton'] 
        self.df = df[~df['Name'].isin(scrubbed)].reset_index(drop=True)

    def run_alpha_sims(self, n_sims=5000, pool_size=10, jitter=0.20, own_cap=125, boost_games=[]):
        n_p = len(self.df)
        proj_vals = self.df['Proj'].values.astype(float).copy()
        
        # --- STEALTH BOOST LOGIC ---
        for game in boost_games:
            mask = self.df['Game Info'].str.contains(game, case=False)
            proj_vals[mask] *= 1.05 
            
        sal_vals = self.df['Salary'].values.astype(float)
        own_vals = self.df['Own'].values.astype(float)
        
        # Base Linear Constraints
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8); bu.append(8)
        A.append(sal_vals); bl.append(49700); bu.append(49900) # Dup Guard
        A.append(own_vals); bl.append(0); bu.append(own_cap)
        
        for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: A.append(self.df[c].values.astype(float)); bl.append(1); bu.append(8)
        A.append(self.df['is_G'].values.astype(float)); bl.append(3); bu.append(8)
        A.append(self.df['is_F'].values.astype(float)); bl.append(3); bu.append(8)
        
        lineup_counts = {}
        status = st.empty()
        
        for i in range(1, n_sims + 1):
            if i % 500 == 0: status.write(f"🔬 `SIMULATING SLATE:` {i}/{n_sims}...")
            
            # Upside Jitter with Stud Stress Test
            dyn_jitter = np.where(sal_vals >= 9000, jitter * 1.5, jitter)
            sim = np.random.normal(proj_vals, proj_vals * dyn_jitter).clip(min=0)
            
            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                ids = tuple(sorted(np.where(res.x > 0.5)[0]))
                lineup_counts[ids] = lineup_counts.get(ids, 0) + 1

        status.empty()
        top_winners = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:pool_size]
        
        pool = []
        for ids, freq in top_winners:
            ldf = self.df.iloc[list(ids)]
            pool.append({
                "Rank": len(pool)+1,
                "Sim_Win_%": f"{round((freq/n_sims)*100, 2)}%",
                "Roster": ", ".join(ldf['Name'].tolist()),
                "Salary": int(ldf['Salary'].sum()),
                "Avg_Own": round(ldf['Own'].mean(), 1)
            })
        return pool

# --- UI COMMAND CENTER ---
st.title("🏀 VANTAGE 99 | ALPHA-MME LAB")
st.sidebar.header("SIMULATION ENGINE")
sim_count = st.sidebar.select_slider("Sim Volume", options=[500, 1000, 5000, 10000], value=5000)
high_totals = st.sidebar.multiselect("Boost High-Total Games (+5%)", ["GSW@CHA", "BOS@ATL", "MIN@SAS", "OKC@MIA"])

f = st.file_uploader("LOAD SALARY CSV", type="csv")
if f:
    df_raw = load_and_clean(f)
    engine = AlphaMMEEngine(df_raw)
    if st.button("🚀 EXECUTE TOURNAMENT SIMULATION"):
        start = time.time()
        results = engine.run_alpha_sims(n_sims=sim_count, boost_games=high_totals)
        st.success(f"Simulation Complete in {round(time.time()-start, 2)}s")
        st.table(pd.DataFrame(results))
