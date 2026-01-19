import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re

# --- PLATFORM CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | DFS COMMAND", layout="wide", page_icon="⚡")

# Institutional Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 10px 20px; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #238636; color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

class VantageEngine:
    def __init__(self, df, sport="NFL"):
        self.df = df.copy()
        self.sport = sport
        self._prepare_data()

    def _prepare_data(self):
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        
        # STRICTURE: REMOVE "OUT" AND ZERO-PROJ PUNTS
        self.df = self.df[self.df['Proj'] > 0.5].reset_index(drop=True)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def get_constraints(self):
        n_p = len(self.df)
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8 if self.sport=="NBA" else 9); bu.append(8 if self.sport=="NBA" else 9)
        A.append(self.df['Sal'].values); bl.append(45000); bu.append(50000)
        if self.sport == "NFL":
            A.append((self.df['Pos'] == 'QB').astype(int).values); bl.append(1); bu.append(1)
            A.append((self.df['Pos'] == 'RB').astype(int).values); bl.append(2); bu.append(3)
            A.append((self.df['Pos'] == 'WR').astype(int).values); bl.append(3); bu.append(4)
            A.append((self.df['Pos'] == 'TE').astype(int).values); bl.append(1); bu.append(2)
            A.append((self.df['Pos'] == 'DST').astype(int).values); bl.append(1); bu.append(1)
        else:
            for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                A.append(self.df['Pos'].str.contains(p).astype(int).values); bl.append(1); bu.append(4)
        return np.vstack(A), bl, bu

    def run_sim_to_opto(self, n_lineups=20, correlation=0.5, leverage=0.5):
        n_p = len(self.df)
        teams = self.df['Team'].unique()
        A, bl, bu = self.get_constraints()
        pool = []
        for i in range(1000):
            team_shift = {t: np.random.normal(1.0, 0.15 * correlation) for t in teams}
            sim_projs = np.array([row['Proj'] * team_shift[row['Team']] * np.random.normal(1.0, 0.1) for _, row in self.df.iterrows()])
            sim_projs = sim_projs * (1 - (self.df['Own'].values * (leverage / 150)))
            res = milp(c=-sim_projs, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = tuple(np.where(res.x > 0.5)[0])
                pool.append({'idx': idx, 'sim_score': sim_projs[list(idx)].sum()})
            if len(pool) >= n_lineups * 2: break
        unique_lineups = {entry['idx']: entry for entry in pool}.values()
        sorted_pool = sorted(unique_lineups, key=lambda x: x['sim_score'], reverse=True)[:n_lineups]
        return [self.df.iloc[list(entry['idx'])] for entry in sorted_pool]

    def run_diagnostic(self, entries_df):
        # 1. Cleaner: Strip "(ID)" from DK Entry Names
        def clean_name(val):
            return re.sub(r'\s*\(\d+\)', '', str(val)).strip()
        
        roster_cols = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST'] if self.sport == "NFL" else \
                      ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        # Flatten and Clean Entry Names
        user_lineups = entries_df[roster_cols].map(clean_name)
        all_players = user_lineups.values.flatten()
        counts = pd.Series(all_players).value_counts()
        
        # 2. Match to Projections
        diag_df = self.df.copy()
        total_entries = len(entries_df)
        diag_df['My Exp'] = diag_df['Name'].map(counts).fillna(0)
        diag_df['My %'] = (diag_df['My Exp'] / total_entries) * 100
        diag_df['Leverage'] = diag_df['My %'] - diag_df['Own']
        
        return diag_df[diag_df['My Exp'] > 0].sort_values(by='My %', ascending=False)

# --- UI INTERFACE ---
st.title("VANTAGE ⚡ ZERO")
with st.sidebar:
    st.header("🕹️ COMMAND")
    sport = st.radio("SPORT", ["NFL", "NBA"])
    uploaded_file = st.file_uploader("1. Projections / Salary CSV", type="csv")
    entry_file = st.file_uploader("2. Entries CSV (Diagnostic)", type="csv")
    st.divider()
    corr_val = st.slider("Correlation", 0.0, 1.0, 0.6)
    own_val = st.slider("Leverage", 0.0, 1.0, 0.4)

if uploaded_file:
    engine = VantageEngine(pd.read_csv(uploaded_file), sport=sport)
    tab1, tab2, tab3 = st.tabs(["📊 LAB", "🏆 POOL", "🔍 DIAGNOSTIC"])
    
    with tab1:
        st.dataframe(engine.df[['Pos', 'Name', 'Team', 'Sal', 'Proj', 'Own']], use_container_width=True)
        if st.button("⚡ GENERATE"):
            st.session_state.results = engine.run_sim_to_opto(n_lineups=20, correlation=corr_val, leverage=own_val)
    
    with tab2:
        if 'results' in st.session_state:
            for i, l_df in enumerate(st.session_state.results):
                with st.expander(f"LINEUP #{i+1} | Proj: {round(l_df['Proj'].sum(), 1)}"):
                    st.table(l_df[['Pos', 'Name', 'Sal', 'Own']])
    
    with tab3:
        if entry_file:
            diag_df = engine.run_diagnostic(pd.read_csv(entry_file))
            st.subheader("Field Leverage Report")
            st.dataframe(diag_df[['Name', 'Team', 'Own', 'My %', 'Leverage']], use_container_width=True)
            
            # KPI Cards
            c1, c2 = st.columns(2)
            c1.metric("Highest Exposure", f"{diag_df.iloc[0]['Name']} ({round(diag_df.iloc[0]['My %'])}%)")
            c2.metric("Max Leverage", f"{diag_df.sort_values(by='Leverage', ascending=False).iloc[0]['Name']}")
        else:
            st.info("Upload your DraftKings 'Entries.csv' to see your exposure vs the field.")
