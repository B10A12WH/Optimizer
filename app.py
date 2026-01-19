import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io

# --- ELITE UI & MULTI-SPORT CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | DK LEGAL ALPHA", layout="wide", page_icon="⚡")

# Institutional Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] { background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d; border-radius: 12px; padding: 15px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 10px 20px; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #238636; color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

class VantageUnifiedOptimizer:
    def __init__(self, df, sport="NBA"):
        self.df = df.copy()
        self.sport = sport
        self._clean_data()

    def _clean_data(self):
        # Auto-detect headers and strictly purge "Out" or Zero-Proj players
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
        """
        SLOT ASSIGNMENT ENGINE: Hard-orders players into DK entry slots
        NBA: PG, SG, SF, PF, C, G, F, UTIL
        NFL: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
        """
        if self.sport == "NFL":
            slots = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
        else:
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
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

    def get_hard_lock_constraints(self):
        """ Hard-Locked Positional Rules """
        n_p = len(self.df)
        A, bl, bu = [], [], []
        # Total Size & Salary Cap
        total = 9 if self.sport == "NFL" else 8
        A.append(np.ones(n_p)); bl.append(total); bu.append(total)
        A.append(self.df['Sal'].values); bl.append(45000); bu.append(50000)

        if self.sport == "NFL":
            for p in ['QB', 'RB', 'WR', 'TE', 'DST']:
                A.append((self.df['Pos'] == p).astype(int).values)
                if p == 'QB': bl.append(1); bu.append(1)
                elif p == 'RB': bl.append(2); bu.append(3)
                elif p == 'WR': bl.append(3); bu.append(4)
                elif p == 'TE': bl.append(1); bu.append(2)
                else: bl.append(1); bu.append(1)
        else: # NBA Multi-Pos Coverage
            for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                A.append(self.df['Pos'].str.contains(p).astype(int).values); bl.append(1); bu.append(5)

        return np.vstack(A), bl, bu

    def run_alpha_sims(self, n_lineups=20, correlation=0.5, leverage=0.5):
        n_p = len(self.df)
        teams = self.df['Team'].unique()
        A, bl, bu = self.get_hard_lock_constraints()
        pool = []
        
        for i in range(1000):
            t_shift = {t: np.random.normal(1.0, 0.15 * correlation) for t in teams}
            sim_p = np.array([row['Proj'] * t_shift[row['Team']] * np.random.normal(1.0, 0.1) for _, row in self.df.iterrows()])
            sim_p *= (1 - (self.df['Own'].values * (leverage / 150)))
            
            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                pool.append({'idx': tuple(idx), 'sim_score': sim_p[idx].sum()})
            if len(pool) >= n_lineups * 2: break
        
        unique = {e['idx']: e for e in pool}.values()
        sorted_pool = sorted(unique, key=lambda x: x['sim_score'], reverse=True)[:n_lineups]
        return [self.get_dk_slots(self.df.iloc[list(e['idx'])]) for e in sorted_pool]

# --- UI INTERFACE ---
st.title("⚡ VANTAGE 99 | LEGAL ALPHA")
with st.sidebar:
    mode = st.radio("SPORT", ["NBA", "NFL"])
    uploaded_file = st.file_uploader("SALARY CSV", type="csv")
    corr = st.slider("Correlation", 0.0, 1.0, 0.6)
    lev = st.slider("Leverage", 0.0, 1.0, 0.4)

if uploaded_file:
    engine = VantageUnifiedOptimizer(pd.read_csv(uploaded_file), sport=mode)
    if st.button(f"⚡ GENERATE ORDERED {mode} LINEUPS"):
        st.session_state.results = engine.run_alpha_sims(correlation=corr, leverage=lev)
        for i, ldf in enumerate(st.session_state.results):
            with st.expander(f"LINEUP #{i+1} | Proj: {round(ldf['Proj'].sum(), 1)}"):
                st.table(ldf[['Slot', 'Name', 'Team', 'Sal', 'Proj']])
