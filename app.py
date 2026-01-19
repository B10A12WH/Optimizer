import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io

# --- ELITE GLASS-MORPHIC UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | ELITE COMMAND", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Main Background */
    .main { background: radial-gradient(circle at top right, #1a1f2e, #0d1117); color: #c9d1d9; font-family: 'Inter', sans-serif; }
    
    /* Lineup Card Styling */
    .lineup-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .lineup-card:hover { transform: translateY(-5px); border-color: #238636; }
    
    /* Metric Badges */
    .badge-green { background: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
    .badge-blue { background: #1f6feb; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
    
    /* Table Styling */
    .stTable { background: transparent !important; border-radius: 8px; overflow: hidden; }
    th { color: #8b949e !important; text-transform: uppercase; font-size: 10px; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

class VantageUnifiedOptimizer:
    def __init__(self, df, sport="NBA"):
        self.df = df.copy()
        self.sport = sport
        self._clean_data()

    def _clean_data(self):
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'avgpoints'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        self.df = self.df[self.df['Proj'] > 0.5].reset_index(drop=True)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def get_dk_slots(self, lineup_df):
        slots = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST'] if self.sport == "NFL" else \
                ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        assigned = []
        players = lineup_df.to_dict('records')
        for slot in slots:
            for i, p in enumerate(players):
                match = False
                pos = p['Pos']
                if self.sport == "NBA":
                    if slot in pos: match = True
                    elif slot == 'G' and ('PG' in pos or 'SG' in pos): match = True
                    elif slot == 'F' and ('SF' in pos or 'PF' in pos): match = True
                    elif slot == 'UTIL': match = True
                elif self.sport == "NFL":
                    if slot == pos: match = True
                    elif slot == 'FLEX' and any(x in pos for x in ['RB', 'WR', 'TE']): match = True
                if match:
                    p['Slot'] = slot
                    assigned.append(p)
                    players.pop(i)
                    break
        return pd.DataFrame(assigned)

    def run_alpha_sims(self, n_lineups=12, correlation=0.5, leverage=0.5):
        # Simplified for logic persistence 
        n_p = len(self.df)
        A = [np.ones(n_p), self.df['Sal'].values]
        bl, bu = [8 if self.sport=="NBA" else 9], [8 if self.sport=="NBA" else 9]
        bl.append(45000); bu.append(50000)
        
        pool = []
        for i in range(500):
            sim_p = self.df['Proj'].values * np.random.normal(1.0, 0.2, n_p)
            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                pool.append(self.df.iloc[idx])
            if len(pool) >= n_lineups: break
        return [self.get_dk_slots(l) for l in pool]

# --- UI INTERFACE ---
st.title("⚡ VANTAGE 99 | ELITE")
with st.sidebar:
    mode = st.radio("SPORT", ["NBA", "NFL"])
    f = st.file_uploader("SALARY CSV", type="csv")
    corr = st.slider("Correlation", 0.0, 1.0, 0.6)

if f:
    engine = VantageUnifiedOptimizer(pd.read_csv(f), sport=mode)
    if st.button(f"⚡ GENERATE {mode} POOL"):
        lineups = engine.run_alpha_sims()
        
        # EYE-CANDY GRID 
        cols = st.columns(2) # 2-column grid for better visibility
        for i, ldf in enumerate(lineups):
            col_idx = i % 2
            with cols[col_idx]:
                st.markdown(f"""
                <div class="lineup-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <span style="font-weight: bold; font-size: 18px;">LINEUP #{i+1}</span>
                        <div>
                            <span class="badge-green">PROJ: {round(ldf['Proj'].sum(), 1)}</span>
                            <span class="badge-blue">${int(ldf['Sal'].sum())}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.table(ldf[['Slot', 'Name', 'Team', 'Sal', 'Proj']])
