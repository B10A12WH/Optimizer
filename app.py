import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re

# --- UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | 10K ALPHA", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .main { background: radial-gradient(circle at top right, #1a1f2e, #0d1117); color: #c9d1d9; }
    .card-elite { border: 2px solid #238636 !important; background: rgba(35, 134, 54, 0.15); box-shadow: 0 0 20px rgba(35, 134, 54, 0.4); }
    .card-strong { border: 2px solid #d29922 !important; background: rgba(210, 153, 34, 0.1); }
    .card-standard { border: 1px solid #30363d !important; background: rgba(22, 27, 34, 0.5); }
    .lineup-card { border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .badge-label { padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: bold; color: white; margin-left: 5px; }
    .bg-win { background: #238636; }
    .bg-proj { background: #1f6feb; }
    </style>
    """, unsafe_allow_html=True)

# IRON-CLAD INJURY LIST
# We force these players out by name and ID substring
FORCED_OUT = ["Obi Toppin", "Haliburton", "Darius Garland", "Jalen Brunson", "Kyrie Irving"]

@st.cache_data
def process_data(file_content, manual_scratches):
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    
    # Standardize
    df['Proj'] = pd.to_numeric(df[cols.get('proj', df.columns[0])], errors='coerce').fillna(0.0)
    df['Sal'] = pd.to_numeric(df[cols.get('salary', df.columns[0])], errors='coerce').fillna(50000)
    df['Name'] = df[cols.get('name', df.columns[0])].astype(str)
    
    # THE TOPPIN KILLER: Multi-layer filtering
    scratch_list = [s.strip().lower() for s in manual_scratches + FORCED_OUT]
    df = df[~df['Name'].str.lower().str.contains('|'.join(scratch_list), na=False)]
    return df[df['Proj'] > 0.1].reset_index(drop=True)

class VantageOptimizer:
    def __init__(self, df):
        self.df = df
        self.n_p = len(df)

    def run_sims(self, n_sims=10000):
        # Optimization: Move constant matrices outside the loop
        A = np.vstack([np.ones(self.n_p), self.df['Sal'].values])
        constraints = LinearConstraint(A, [8, 45000], [8, 50000])
        integrality = np.ones(self.n_p)
        bounds = Bounds(0, 1)
        
        lineup_counts = {}
        progress_bar = st.progress(0)
        
        # Batching for performance
        for i in range(n_sims):
            # Simulation logic
            sim_p = self.df['Proj'].values * np.random.normal(1.0, 0.15, self.n_p)
            res = milp(c=-sim_p, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.success:
                idx = tuple(sorted(np.where(res.x > 0.5)[0]))
                lineup_counts[idx] = lineup_counts.get(idx, 0) + 1
            
            if i % 500 == 0:
                progress_bar.progress(i / n_sims)
        
        return lineup_counts

# --- UI ---
st.sidebar.title("🕹️ COMMAND")
f = st.sidebar.file_uploader("UPLOAD CSV", type="csv")
scratches = st.sidebar.text_area("🚑 ADD SCRATCHES").split('\n')

if f:
    data = process_data(f.getvalue(), scratches)
    optimizer = VantageOptimizer(data)
    
    if st.button("🚀 GENERATE 10,000 SIMS"):
        counts = optimizer.run_sims()
        # Sort and Display Logic...
        st.success("Simulations Complete!")
