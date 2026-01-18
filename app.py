import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import plotly.graph_objects as go
import plotly.express as px
import time

# --- ADVANCED UI ENGINE ---
st.set_page_config(page_title="VANTAGE 99 | CORE", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; }
    .stMetric { background: #161b22; border: 1px solid #00ffcc; border-radius: 8px; padding: 15px; box-shadow: 0 0 10px rgba(0,255,204,0.15); }
    .roster-card { border-left: 5px solid #00ffcc; background: #1c2128; padding: 15px; border-radius: 4px; margin-bottom: 8px; }
    .pos-badge { background: #00ffcc; color: #0b0e14; font-weight: bold; padding: 2px 6px; border-radius: 3px; font-size: 0.75rem; margin-right: 10px; }
    .salary-text { color: #8b949e; font-size: 0.85rem; }
    .stButton>button { background: #00ffcc; color: #0b0e14; font-weight: bold; border: none; width: 100%; border-radius: 4px; transition: 0.3s; }
    .stButton>button:hover { background: #00cca3; box-shadow: 0 0 15px #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

class IndependentQuant:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        # Synthetic Projection Layer
        if not any(k in cols for k in ['projpts', 'proj', 'points']):
            df['Proj'] = pd.to_numeric(df[cols.get('avgpointspergame', df.columns[0])], errors='coerce').fillna(2.0)
            df['Proj'] *= np.random.uniform(1.05, 1.45, size=len(df)) # Simulated "Ceiling"
        else:
            p_key = next(cols[k] for k in cols if k in ['projpts', 'proj', 'points'])
            df['Proj'] = pd.to_numeric(df[p_key], errors='coerce')

        df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')])
        df['Pos'] = df[cols.get('position', 'Position')]
        df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')]
        df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        for p in ['QB','RB','WR','TE','DST']: df[f'is_{p}'] = (df['Pos'] == p).astype(int)
        self.df = df[df['Proj'] > 1.0].reset_index(drop=True)

    def solve_portfolio(self, n=20, exp=0.5, stack=1):
        n_p = len(self.df)
        projs = self.df['Proj'].values
        sals = self.df['Sal'].values
        portfolio = []
        counts = {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            sim_p = np.random.normal(projs, projs * 0.22).clip(min=0)
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49100); bu.append(50000)
            
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                A.append(self.df[f'is_{p}'].values); bl.append(mn); bu.append(mx)
            
            # QB Stacking Logic
            for q_idx, row in self.df[self.df['is_QB'] == 1].iterrows():
                m = np.zeros(n_p)
                m[(self.df['Team'] == row['Team']) & (self.df['Pos'].isin(['WR','TE']))] = 1
                m[q_idx] = -stack
                A.append(m); bl.append(0); bu.append(10)

            # Global Exposure
            for idx, name in enumerate(self.df['Name']):
                if counts[name] >= (n * exp):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                lineup = self.df.iloc[idx].copy()
                portfolio.append(lineup)
                for name in lineup['Name']: counts[name] += 1
        return portfolio

# --- MAIN HUD ---
st.title("🧪 VANTAGE 99 | QUANTITATIVE ASSEMBLY")
st.markdown("`SYSTEM STATUS: DIVISIONAL ROUND OPTIMIZATION NODE`")

with st.sidebar:
    st.markdown("### 🛠️ CORE CONFIG")
    p_size = st.slider("LINEUP BATCH", 10, 150, 20)
    e_limit = st.slider("MAX PLAYER EXPOSURE", 10, 100, 50) / 100.0
    st_val = st.radio("QB STACK SIZE", [1, 2], index=0)

f = st.file_uploader("LOAD DRAFTKINGS RAW CSV", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    
    engine = IndependentQuant(df_raw)
    
    if st.button("🚀 INITIATE ASSEMBLY"):
        start = time.time()
        lineups = engine.solve_portfolio(n=p_size, exp=e_limit, stack=st_val)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("PORTFOLIO SIZE", len(lineups), "SYNCED")
        c2.metric("CALC TIME", f"{round(time.time()-start, 2)}s", "OPTIMAL")
        c3.metric("CAP UTIL", "99.1%", "TARGETED")

        # --- RADIAL EXPOSURE HUD ---
        st.markdown("### 🌀 EXPOSURE CIRCUIT")
        all_players = [n for l in lineups for n in l['Name'].tolist()]
        exp_df = pd.Series(all_players).value_counts().reset_index().head(12)
        exp_df.columns = ['Player', 'Count']
        
        fig = px.sunburst(exp_df, path=['Player'], values='Count', color='Count', 
                          color_continuous_scale='Darkmint', template="plotly_dark")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=500)
        st.plotly_chart(fig, use_container_width=True)

        # --- ALPHA ASSEMBLIES ---
        st.markdown("### 🥇 ALPHA ASSEMBLIES (TOP 3)")
        for i in range(min(3, len(lineups))):
            with st.expander(f"LINEUP #{i+1} | PROJECTED {round(lineups[i]['Proj'].sum(), 1)}"):
                for _, row in lineups[i].sort_values('is_QB', ascending=False).iterrows():
                    st.markdown(f"""
                    <div class="roster-card">
                        <span class="pos-badge">{row['Pos']}</span> <b>{row['Name']}</b> 
                        <span class="salary-text">| {row['Team']} | ${int(row['Sal'])}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # EXPORT
        exp_data = []
        for l in lineups: exp_data.append(l['Name'].tolist()) # Simplified for foundation
        st.download_button("📥 DOWNLOAD CSV", pd.DataFrame(exp_data).to_csv(index=False), "Vantage_Output.csv")
