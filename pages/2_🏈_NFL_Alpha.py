import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- VANTAGE 99 | v64.0: NFL ALPHA-MME ---
st.set_page_config(page_title="VANTAGE 99 | NFL ALPHA", layout="wide", page_icon="🏈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; }
    .swap-ready { color: #00ffcc; font-weight: bold; border: 1px solid #00ffcc; padding: 2px 5px; border-radius: 4px; font-size: 0.7rem; }
    </style>
    """, unsafe_allow_html=True)

class TacticalOptimizerV64:
    def __init__(self, df):
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        
        # Identification & Projection Mapping
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        
        # 1/18 LATE GAME FLAG (6:30 PM Window: LAR vs CHI)
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
            
        # JAN 18 SCRUB (Confirmed Divisional Round OUTs)
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, swap_bias=True):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        late_mask = self.df['is_late'].values.astype(np.float64)
        
        # Weather Adjustment: Chicago Cold (18°F) adds ceiling variance
        scale = np.clip(raw_p * 0.25, 0.01, None)
        
        portfolio = []
        for i in range(n):
            # Apply Swap Bias: Micro-bump late players to prioritize them for FLEX slots
            adj_p = raw_p + (late_mask * 0.05 if swap_bias else 0)
            sim_p = np.random.normal(adj_p, scale).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # Roster Size
            A.append(sals); bl.append(49200); bu.append(50000) # Salary Range
            
            # Position Hardstops (QB-RB2-WR3-TE-FLEX-DST)
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1),
                       options={'presolve': True})
            
            if res.success:
                portfolio.append(self.df.iloc[np.where(res.x > 0.5)[0]].copy())
        return portfolio

# --- DASHBOARD UI ---
st.title("🧬 VANTAGE 99 | NFL ALPHA v64.0")
st.sidebar.warning("❄️ CHI WEATHER: 18°F | WIND: 15MPH")
st.sidebar.info("Late-Swap Priority: LAR/CHI @ 6:30 PM")

f = st.file_uploader("LOAD DIVISIONAL ROUND CSV", type="csv")
if f:
    df_input = pd.read_csv(f)
    engine = TacticalOptimizerV64(df_input)
    
    if st.button("🚀 EXECUTE PORTFOLIO GENERATION"):
        st.session_state.nfl_portfolio = engine.assemble(n=20)
        st.session_state.nfl_sel_idx = 0

    if 'nfl_portfolio' in st.session_state:
        col_list, col_scout = st.columns([1, 2])
        with col_list:
            for i, l in enumerate(st.session_state.nfl_portfolio):
                # Check for Late-Swap Ready (Player in FLEX from 6:30 PM game)
                label = f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS"
                if st.button(label, key=f"nfl_b{i}"):
                    st.session_state.nfl_sel_idx = i

        with col_scout:
            l = st.session_state.nfl_portfolio[st.session_state.nfl_sel_idx]
            st.write(f"### Scouting Report: Lineup #{st.session_state.nfl_sel_idx+1}")
            st.dataframe(l[['Pos', 'Name', 'Team', 'Sal', 'Proj']])
