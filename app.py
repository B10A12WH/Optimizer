import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime
import time

# --- VANTAGE 99 | BUG-FIXED TURBO EDITION (V43.0) ---
st.set_page_config(page_title="VANTAGE 99 | ALPHA LAB", layout="wide", page_icon="🏀")

# Global CSS for aesthetics
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
        st.error(f"Error loading CSV: {e}")
        return None

class AlphaOptimizer:
    def __init__(self, df):
        # 1. Position Logic
        for p in ['PG','SG','SF','PF','C']: df[f'is_{p}'] = df['Position'].str.contains(p).astype(int)
        df['is_G'] = ((df['is_PG']==1)|(df['is_SG']==1)).astype(int)
        df['is_F'] = ((df['is_SF']==1)|(df['is_PF']==1)).astype(int)
        
        # 2. Projections & Ownership
        df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        df['Own'] = pd.to_numeric(df['Ownership'], errors='coerce') if 'Ownership' in df.columns else 15.0
        df['Own'] = df['Own'].fillna(15.0)

        # 3. Time Parsing
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        df['Time'] = df['Game Info'].apply(get_time)
        
        # 4. Scrub List
        scrubbed = ['Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green'] 
        self.df = df[~df['Name'].isin(scrubbed)].reset_index(drop=True)

    def run_fast_sims(self, n_sims=2000, own_cap=125, jitter=0.20):
        n_p = len(self.df)
        proj_vals = self.df['Proj'].values.astype(float)
        sal_vals = self.df['Salary'].values.astype(float)
        own_vals = self.df['Own'].values.astype(float)
        
        # Pre-generate scenarios as a matrix (Fastest)
        dyn_jitter = np.where(sal_vals >= 9000, jitter * 1.5, jitter)
        sim_matrix = np.random.normal(proj_vals, proj_vals * dyn_jitter, size=(n_sims, n_p)).clip(min=0)
        
        lineup_counts = {}
        
        # Pre-build Constraints
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8); bu.append(8)
        A.append(sal_vals); bl.append(49600); bu.append(49900)
        A.append(own_vals); bl.append(0); bu.append(own_cap)
        for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: A.append(self.df[c].values.astype(float)); bl.append(1); bu.append(8)
        A.append(self.df['is_G'].values.astype(float)); bl.append(3); bu.append(8)
        A.append(self.df['is_F'].values.astype(float)); bl.append(3); bu.append(8)
        A_stack = np.vstack(A)

        with st.status("🔬 ANALYZING SLATE...", expanded=True) as status:
            for i in range(n_sims):
                # Using 'Turbo' settings: rel_gap=0.01 (1% optimality is fine for DFS)
                res = milp(c=-sim_matrix[i], constraints=LinearConstraint(A_stack, bl, bu), 
                           integrality=np.ones(n_p), bounds=Bounds(0, 1),
                           options={'mip_rel_gap': 0.01, 'presolve': True})
                
                if res.success:
                    ids = tuple(sorted(np.where(res.x > 0.5)[0]))
                    lineup_counts[ids] = lineup_counts.get(ids, 0) + 1
                
                if i % 500 == 0:
                    status.update(label=f"Simulated {i}/{n_sims} game scripts...")

        sorted_lineups = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return sorted_lineups

# --- APP UI ---
st.title("🏀 VANTAGE 99 | ALPHA LAB")
f = st.file_uploader("LOAD SALARY CSV", type="csv")

if f:
    df_raw = load_and_clean(f)
    if df_raw is not None:
        engine = AlphaOptimizer(df_raw)
        if st.button("🚀 RUN ALPHA SIMULATION"):
            results = engine.run_fast_sims(n_sims=2000) # 2k is safe & fast
            if results:
                st.subheader("🥇 TOP WINNING LINEUPS")
                # Format for display
                display_list = []
                for ids, freq in results:
                    ldf = engine.df.iloc[list(ids)]
                    display_list.append({
                        "Win %": f"{round((freq/2000)*100, 2)}%",
                        "Roster": ", ".join(ldf['Name'].tolist()),
                        "Salary": ldf['Salary'].sum()
                    })
                st.table(pd.DataFrame(display_list))
