import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import plotly.graph_objects as go
import plotly.express as px
import time

# --- INDUSTRIAL UI ENGINE ---
st.set_page_config(page_title="VANTAGE 99 | COMMAND", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; }
    .lineup-container { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 25px; border-top: 4px solid #00ffcc; }
    .player-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #21262d; font-size: 0.95rem; }
    .pos-label { color: #0b0e14; background: #00ffcc; font-weight: bold; width: 42px; display: inline-block; text-align: center; border-radius: 3px; font-size: 0.75rem; margin-right: 10px; }
    .audit-reasoning { font-style: italic; color: #8b949e; font-size: 0.85rem; margin-top: 15px; padding: 10px; background: rgba(0, 255, 204, 0.05); border-radius: 6px; border: 1px dashed #30363d; }
    .header-score { float: right; color: #00ffcc; font-size: 1.1rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

class TacticalOptimizer:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        
        # 1. BROADCAST-SAFE PROJECTION ENGINE
        if not any(k in cols for k in ['projpts', 'proj', 'points']):
            avg_key = cols.get('avgpointspergame', df.columns[0])
            self.df['Proj'] = pd.to_numeric(df[avg_key], errors='coerce').fillna(2.0)
        else:
            p_key = next(cols[k] for k in cols if k in ['projpts', 'proj', 'points'])
            self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)

        # 2. CORE SANITIZATION
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
        
        # 3. JAN 18 DIVISIONAL BLACKLIST
        scrubbed = ['Nico Collins', 'Justin Watson'] # Concussion Protocol
        self.df = self.df[~self.df['Name'].isin(scrubbed)].reset_index(drop=True)

    def audit_lineup(self, lineup):
        reasons = []
        qb_row = lineup[lineup['Pos'] == 'QB']
        if not qb_row.empty:
            qb = qb_row.iloc[0]
            stack = lineup[(lineup['Team'] == qb['Team']) & (lineup['Pos'].isin(['WR','TE']))]
            if len(stack) >= 2: reasons.append("🔥 TRIPLE THREAT: High-ceiling game stack.")
            elif len(stack) == 1: reasons.append("🎯 CORE PAIR: Classic QB/WR1 correlation.")
        
        if lineup['Sal'].sum() >= 49800: reasons.append("💰 MAX CAP: Industrial talent density.")
        if lineup[lineup['Pos'] == 'RB']['Sal'].min() < 5200: reasons.append("💎 VALUE BACK: Strategic salary relief.")
        return " | ".join(reasons) if reasons else "Balanced Tournament Entry"

    def assemble(self, n=10, exp=0.5, jitter=0.22):
        # --- FIXED BROADCASTING LOGIC ---
        # Explicitly flatten and cast to float64 to prevent NumPy broadcasting errors
        raw_p = self.df['Proj'].values.flatten().astype(np.float64)
        sals = self.df['Sal'].values.flatten().astype(np.float64)
        n_p = len(self.df)
        
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        bar = st.progress(0)
        
        for i in range(n):
            # Safe Normal Draw: projs * jitter must result in matching array shape
            sim_p = np.random.normal(raw_p, raw_p * jitter).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49000); bu.append(50000)
            
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                A.append(self.df[f'is_{p}'].values); bl.append(mn); bu.append(mx)

            for idx, name in enumerate(self.df['Name']):
                if counts[name] >= (n * exp):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                lineup = self.df.iloc[idx].copy()
                portfolio.append(lineup)
                for name in lineup['Name']: counts[name] += 1
            bar.progress((i + 1) / n)
        return portfolio

# --- HUD COMMANDS ---
st.title("🧪 VANTAGE 99 | STRATEGIC COMMAND")
st.markdown("`SYSTEM STATUS: DIVISIONAL SUNDAY ACTIVE` | HOU@NE • LAR@CHI")

with st.sidebar:
    st.header("⚙️ PORTFOLIO")
    batch_size = st.slider("BATCH", 10, 50, 20)
    exposure = st.slider("MAX EXPOSURE", 0.1, 1.0, 0.5)
    variance = st.slider("SIM JITTER", 0.1, 0.4, 0.22)

f = st.file_uploader("LOAD DK SALARY CSV", type="csv")

if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = TacticalOptimizer(df_raw)
    
    if st.button("🚀 EXECUTE SCOUTING BATCH"):
        lineups = engine.assemble(n=batch_size, exp=exposure, jitter=variance)
        
        # 📊 SLATE RADAR (Metric Row)
        st.subheader("📡 SLATE RADAR")
        all_teams = [p['Team'] for l in lineups for _, p in l.iterrows()]
        team_exposure = pd.Series(all_teams).value_counts()
        cols = st.columns(min(len(team_exposure), 4))
        for i, (team, count) in enumerate(team_exposure.head(4).items()):
            cols[i].metric(f"TARGET: {team}", f"{int((count/len(all_teams))*100)}%", "EXPOSURE")

        # 🌀 EXPOSURE CIRCUIT (Sunburst)
        exp_df = pd.Series([n for l in lineups for n in l['Name'].tolist()]).value_counts().reset_index().head(12)
        exp_df.columns = ['Player', 'Count']
        fig = px.sunburst(exp_df, path=['Player'], values='Count', color='Count', color_continuous_scale='Darkmint', template="plotly_dark")
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 🥇 SCOUTING REPORTS
        st.markdown("---")
        st.subheader("🥇 PRIMARY SCOUTING REPORTS")
        c1, c2 = st.columns(2)
        for i, l in enumerate(lineups):
            target = c1 if i % 2 == 0 else c2
            with target:
                st.markdown(f"""
                <div class="lineup-container">
                    <span class="header-score">{round(l['Proj'].sum(), 1)} PTS</span>
                    <h4 style="margin:0; color:#00ffcc;">REPORT #{i+1}</h4>
                    <div style="margin: 15px 0;">
                """, unsafe_allow_html=True)
                for _, p in l.sort_values('is_QB', ascending=False).iterrows():
                    st.markdown(f"""
                    <div class="player-row">
                        <span><span class="pos-label">{p['Pos']}</span> <b>{p['Name']}</b></span>
                        <span style="color:#8b949e;">{p['Team']} • ${int(p['Sal'])}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="audit-reasoning">
                        <strong>💡 THE ANGLE:</strong> {engine.audit_lineup(l)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
