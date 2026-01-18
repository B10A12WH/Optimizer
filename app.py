import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import plotly.graph_objects as go
import time

# --- COMMAND CENTER UI ---
st.set_page_config(page_title="VANTAGE 99 | ELITE", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* Professional Lineup Card */
    .lineup-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0;
        margin-bottom: 30px;
        overflow: hidden;
    }
    .lineup-header {
        background: #21262d;
        padding: 10px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #30363d;
    }
    .roster-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1px;
        background: #30363d; /* Creates the grid line effect */
    }
    .slot-box {
        background: #161b22;
        padding: 15px;
        display: flex;
        flex-direction: column;
    }
    .slot-label {
        color: #00ffcc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: bold;
        margin-bottom: 4px;
    }
    .player-name { font-weight: 700; font-size: 0.9rem; margin-bottom: 2px; }
    .player-meta { color: #8b949e; font-size: 0.75rem; }
    
    /* Audit Footer */
    .audit-footer {
        background: #0d1117;
        padding: 12px 20px;
        border-top: 1px solid #30363d;
        font-size: 0.8rem;
        color: #00ffcc;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

class EliteOptimizer:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        # Header Mapping
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        for p in ['QB','RB','WR','TE','DST']: self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.5, jitter=0.22):
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        scale = np.clip(raw_p * jitter, 0.01, None)
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            sim_p = np.random.normal(raw_p, scale).clip(min=0)
            A, bl, bu = [], [], []
            A.append(np.ones(len(self.df))); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49200); bu.append(50000)
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                A.append(self.df[f'is_{p}'].values); bl.append(mn); bu.append(mx)

            # Max Exposure
            for idx, name in enumerate(self.df['Name']):
                if counts[name] >= (n * exp):
                    m = np.zeros(len(self.df)); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(len(self.df)), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                lineup = self.df.iloc[idx].copy()
                portfolio.append(lineup)
                for name in lineup['Name']: counts[name] += 1
        return portfolio

# --- MAIN RENDERER ---
st.title("🧪 VANTAGE 99 | STRATEGIC HUD")
f = st.file_uploader("UPLOAD DATASET", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteOptimizer(df_raw)
    
    if st.button("🚀 EXECUTE ALPHA ASSEMBLY"):
        lineups = engine.assemble(n=20)
        
        for i, l in enumerate(lineups):
            # 1. ORGANIZE BY OFFICIAL DK ORDER
            qb = l[l['Pos'] == 'QB'].iloc[0]
            rbs = l[l['Pos'] == 'RB'].sort_values('Sal', ascending=False)
            wrs = l[l['Pos'] == 'WR'].sort_values('Sal', ascending=False)
            te = l[l['Pos'] == 'TE'].iloc[0]
            dst = l[l['Pos'] == 'DST'].iloc[0]
            
            # Find Flex
            all_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
            flex = l[~l['ID'].isin(all_ids)].iloc[0]
            
            roster = [
                ("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]), 
                ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]), 
                ("TE", te), ("FLEX", flex), ("DST", dst)
            ]
            
            # 2. RENDER THE HUD CARD
            st.markdown(f"""
            <div class="lineup-card">
                <div class="lineup-header">
                    <span style="color:#00ffcc; font-weight:bold;">LINEUP #{i+1}</span>
                    <span style="font-family:'JetBrains Mono'; font-size:0.9rem;">PROJECTED: {round(l['Proj'].sum(), 1)} | SALARY: ${int(l['Sal'].sum())}</span>
                </div>
                <div class="roster-grid">
            """, unsafe_allow_html=True)
            
            for label, p in roster:
                st.markdown(f"""
                    <div class="slot-box">
                        <span class="slot-label">{label}</span>
                        <span class="player-name">{p['Name']}</span>
                        <span class="player-meta">{p['Team']} • ${int(p['Sal'])}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                </div>
                <div class="audit-footer">
                    ⚡ TOURNAMENT DNA: {"STOCKED" if len(l[l['Team'] == qb['Team']]) > 1 else "BALANCED"} • 
                    {"VALUE FLEX" if flex['Sal'] < 5000 else "POWER FLEX"} • 
                    MATCHUP CORRELATION: {len(l['Team'].unique())}/9
                </div>
            </div>
            """, unsafe_allow_html=True)
