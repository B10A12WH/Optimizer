import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import plotly.graph_objects as go
import time

# --- COMPACT UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | COMPACT", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* Condensed Row Container */
    .compact-row {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 8px 15px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.85rem;
    }
    .compact-row:hover { border-color: #00ffcc; background: #1c2128; }
    
    /* Player Tag Styling */
    .player-tag {
        background: #21262d;
        border: 1px solid #30363d;
        padding: 2px 8px;
        border-radius: 3px;
        margin-right: 4px;
        display: inline-block;
        font-size: 0.75rem;
    }
    .pos-label { color: #00ffcc; font-weight: bold; margin-right: 4px; font-family: 'JetBrains Mono'; }
    
    /* Tiny Badges */
    .dna-badge { font-size: 0.65rem; padding: 1px 5px; border-radius: 10px; border: 1px solid #00ffcc; color: #00ffcc; }
    .score-text { font-weight: bold; color: #00ffcc; width: 60px; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

class CompactOptimizer:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        for p in ['QB','RB','WR','TE','DST']: self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
        # Divisional Blacklist
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.5):
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        scale = np.clip(raw_p * 0.22, 0.01, None)
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            sim_p = np.random.normal(raw_p, scale).clip(min=0)
            A, bl, bu = [], [], []
            A.append(np.ones(len(self.df))); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49200); bu.append(50000)
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                A.append(self.df[f'is_{p}'].values); bl.append(mn); bu.append(mx)

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

# --- HUD RENDERER ---
st.title("🧪 VANTAGE 99 | COMPACT HUD")
f = st.file_uploader("LOAD DK SALARIES", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = CompactOptimizer(df_raw)
    
    if st.button("🚀 ASSEMBLE PORTFOLIO"):
        lineups = engine.assemble(n=20)
        
        st.markdown("### 📋 PORTFOLIO OVERVIEW")
        for i, l in enumerate(lineups):
            # Sort players into DraftKings Order
            qb = l[l['Pos'] == 'QB'].iloc[0]
            rbs = l[l['Pos'] == 'RB'].sort_values('Sal', ascending=False)
            wrs = l[l['Pos'] == 'WR'].sort_values('Sal', ascending=False)
            te = l[l['Pos'] == 'TE'].iloc[0]
            dst = l[l['Pos'] == 'DST'].iloc[0]
            
            # FLEX Logic
            all_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
            flex = l[~l['ID'].isin(all_ids)].iloc[0]
            
            # Tournament DNA logic (Condensed)
            is_stacked = "STACKED" if len(l[l['Team'] == qb['Team']]) > 1 else "SOLO"
            
            # Roster Tag Display
            tags_html = ""
            roster = [("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]), ("WR", wrs.iloc[0]), 
                      ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]), ("TE", te), ("F", flex), ("DST", dst)]
            
            for label, p in roster:
                tags_html += f'<div class="player-tag"><span class="pos-label">{label}</span> {p["Name"]}</div>'

            st.markdown(f"""
            <div class="compact-row">
                <div style="display:flex; align-items:center;">
                    <span style="margin-right:15px; color:#8b949e; width:30px;">#{i+1}</span>
                    <div style="display:flex; flex-wrap:wrap;">{tags_html}</div>
                </div>
                <div style="display:flex; align-items:center;">
                    <span class="dna-badge" style="margin-right:10px;">{is_stacked}</span>
                    <span class="score-text">{round(l['Proj'].sum(), 1)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
