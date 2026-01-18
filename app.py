import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | ELITE", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* Master-Detail Split */
    .lineup-row {
        background: #161b22; border: 1px solid #30363d; border-radius: 6px;
        padding: 12px; margin-bottom: 8px; cursor: pointer; transition: 0.2s;
        display: flex; justify-content: space-between; align-items: center;
    }
    .lineup-row:hover { border-color: #00ffcc; background: #1c2128; }
    
    /* Scouting Report Styles */
    .scouting-card { background: #0d1117; border: 1px solid #00ffcc; border-radius: 12px; padding: 25px; position: sticky; top: 20px; }
    .slot-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; }
    .pos-label { color: #00ffcc; font-family: 'JetBrains Mono'; font-weight: bold; width: 50px; }
    </style>
    """, unsafe_allow_html=True)

class EliteOptimizer:
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
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.5):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        scale = np.clip(raw_p * 0.22, 0.01, None)
        
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            sim_p = np.random.normal(raw_p, scale).clip(min=0)
            A, bl, bu = [], [], []
            
            # --- THE HARDSTOP CONSTRAINTS ---
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # 9 total players
            A.append(sals); bl.append(49200); bu.append(50000) # Salary Cap
            
            # Position | Min | Max (including FLEX)
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3) # Max 3 RBs
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4) # Max 4 WRs
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(2) # Max 2 TEs
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            # Global Exposure Governor
            for idx, name in enumerate(self.df['Name']):
                if counts[name] >= (n * exp):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                portfolio.append(self.df.iloc[idx].copy())
                for name in self.df.iloc[idx]['Name']: counts[name] += 1
        return portfolio

# --- DASHBOARD RENDERER ---
st.title("🧪 VANTAGE 99 | ELITE COMMAND")

f = st.file_uploader("LOAD DK DATASET", type="csv")
if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteOptimizer(df_raw)
    
    if 'portfolio' not in st.session_state:
        if st.button("🚀 ASSEMBLE PORTFOLIO"):
            st.session_state.portfolio = engine.assemble(n=20)
            st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state:
        col_list, col_scout = st.columns([1, 1.5])

        with col_list:
            st.markdown("### 📋 PORTFOLIO INDEX")
            for i, l in enumerate(st.session_state.portfolio):
                score = round(l['Proj'].sum(), 1)
                if st.button(f"L{i+1} | {score} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            # --- DRAFTKINGS OFFICIAL ORDERING ---
            qb = l[l['Pos'] == 'QB'].iloc[0]
            rbs = l[l['Pos'] == 'RB'].sort_values('Sal', ascending=False)
            wrs = l[l['Pos'] == 'WR'].sort_values('Sal', ascending=False)
            te = l[l['Pos'] == 'TE'].sort_values('Sal', ascending=False).iloc[0]
            dst = l[l['Pos'] == 'DST'].iloc[0]
            
            # Find Flex (The remaining RB/WR/TE)
            core_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
            flex = l[~l['ID'].isin(core_ids)].iloc[0]
            
            roster = [("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]), ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]), ("TE", te), ("FLEX", flex), ("DST", dst)]

            st.markdown(f"""
            <div class="scouting-card">
                <h3 style="margin-top:0; color:#00ffcc;">SCOUTING REPORT: LINEUP #{st.session_state.sel_idx+1}</h3>
                <p style="color:#8b949e; font-size:0.9rem;">TOTAL SALARY: ${int(l['Sal'].sum())} / $50,000</p>
                <div style="margin-bottom:20px;"></div>
            """, unsafe_allow_html=True)
            
            for label, p in roster:
                st.markdown(f"""
                <div class="slot-row">
                    <span class="pos-label">{label}</span>
                    <span style="font-weight:bold;">{p['Name']}</span>
                    <span style="color:#8b949e;">{p['Team']} • ${int(p['Sal'])}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
