import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

# --- PLATFORM CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | QUANTITATIVE DFS", layout="wide", page_icon="⚡")

# Custom CSS for "SaberSim" Dark Aesthetics
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #238636; color: white; border: none; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

class VantageEngine:
    def __init__(self, df, sport="NFL"):
        self.df = df.copy()
        self.sport = sport
        self._prepare_data()

    def _prepare_data(self):
        # 1. Standardize Columns
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'points'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        
        # 2. STRICTURE: REMOVE "OUT" AND ZERO-VALUE PUNTS
        # This fixes the issue where 3k players with no projection were being picked
        self.df = self.df[self.df['Proj'] > 0.5].reset_index(drop=True)
        if 'status' in cols:
            status_col = cols['status']
            self.df = self.df[~self.df[status_col].str.contains('OUT|O|Irrelevant', case=False, na=False)]

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def get_constraints(self):
        n_p = len(self.df)
        A, bl, bu = [], [], []
        
        # Total Players & Salary
        A.append(np.ones(n_p)); bl.append(8 if self.sport=="NBA" else 9); bu.append(8 if self.sport=="NBA" else 9)
        A.append(self.df['Sal'].values); bl.append(45000); bu.append(50000)

        if self.sport == "NFL":
            # DraftKings NFL: QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST
            A.append((self.df['Pos'] == 'QB').astype(int).values); bl.append(1); bu.append(1)
            A.append((self.df['Pos'] == 'RB').astype(int).values); bl.append(2); bu.append(3)
            A.append((self.df['Pos'] == 'WR').astype(int).values); bl.append(3); bu.append(4)
            A.append((self.df['Pos'] == 'TE').astype(int).values); bl.append(1); bu.append(2)
            A.append((self.df['Pos'] == 'DST').astype(int).values); bl.append(1); bu.append(1)
        else:
            # DraftKings NBA: 1 PG, 1 SG, 1 SF, 1 PF, 1 C, 1 G, 1 F, 1 UTIL
            for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                A.append(self.df['Pos'].str.contains(p).astype(int).values); bl.append(1); bu.append(4)
        
        return np.vstack(A), bl, bu

    def run_sim_to_opto(self, n_lineups=20, iterations=1000, correlation=0.5, leverage=0.5):
        n_p = len(self.df)
        teams = self.df['Team'].unique()
        A, bl, bu = self.get_constraints()
        
        pool = []
        with st.status("🚀 EXECUTING GAME SCRIPTS...", expanded=False) as status:
            for i in range(iterations):
                # TEAM-BASED COVARIANCE (SaberSim Mimic)
                # If a team "succeeds" in a sim, all players on that team get a boost
                team_shift = {t: np.random.normal(1.0, 0.15 * correlation) for t in teams}
                sim_projs = np.array([
                    row['Proj'] * team_shift[row['Team']] * np.random.normal(1.0, 0.1) 
                    for _, row in self.df.iterrows()
                ])
                
                # OWNERSHIP LEVERAGE (Stokastic Mimic)
                # Penalize high ownership to find low-owned ceiling outcomes
                sim_projs = sim_projs * (1 - (self.df['Own'].values * (leverage / 150)))

                res = milp(c=-sim_projs, constraints=LinearConstraint(A, bl, bu), 
                           integrality=np.ones(n_p), bounds=Bounds(0, 1))
                
                if res.success:
                    idx = tuple(np.where(res.x > 0.5)[0])
                    pool.append({'idx': idx, 'sim_score': sim_projs[list(idx)].sum(), 'real_proj': self.df.iloc[list(idx)]['Proj'].sum()})
                
                if len(pool) >= n_lineups * 3: break # Optimize for speed

        # Sort by Simulated Score but return top N unique lineups
        unique_lineups = {entry['idx']: entry for entry in pool}.values()
        sorted_pool = sorted(unique_lineups, key=lambda x: x['sim_score'], reverse=True)[:n_lineups]
        return [self.df.iloc[list(entry['idx'])] for entry in sorted_pool]

# --- UI HEADER ---
col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.title("VANTAGE ⚡ ZERO")
    st.caption("ULTRA-HIGH VARIANCE SIMULATOR | SABER-LOGIC V2")

with col_status:
    st.metric("LATENCY", "9ms", "OPTIMAL")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🕹️ COMMAND CENTER")
    sport = st.radio("ACTIVE THEATRE", ["NFL", "NBA"])
    
    st.subheader("SABER SLIDERS")
    corr_val = st.slider("Correlation (Stacking)", 0.0, 1.0, 0.6, help="Higher = tighter team stacks")
    own_val = st.slider("Leverage (Chalk Tax)", 0.0, 1.0, 0.4, help="Higher = avoid popular players")
    
    lineup_count = st.number_input("LINEUPS TO GENERATE", 1, 150, 20)
    
    st.divider()
    uploaded_file = st.file_uploader("UPLOAD SALARY.CSV (DK FORMAT)", type="csv")

# --- MAIN INTERFACE ---
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    engine = VantageEngine(data, sport=sport)
    
    tab1, tab2 = st.tabs(["📊 PROJECTION LAB", "🏆 OPTIMIZED POOL"])
    
    with tab1:
        st.subheader(f"Filtered {sport} Field")
        st.caption("Note: Players with 0 projection or 'OUT' status have been automatically purged.")
        st.dataframe(engine.df[['Pos', 'Name', 'Team', 'Sal', 'Proj', 'Own']], use_container_width=True, height=400)
        
        if st.button("⚡ EXECUTE SIM-TO-OPTO ENGINE"):
            st.session_state.results = engine.run_sim_to_opto(
                n_lineups=lineup_count, 
                correlation=corr_val, 
                leverage=own_val
            )
            st.success(f"Generated {len(st.session_state.results)} GPP-Ready Scenarios")

    with tab2:
        if 'results' in st.session_state:
            for i, l_df in enumerate(st.session_state.results):
                with st.expander(f"LINEUP #{i+1} | Proj: {round(l_df['Proj'].sum(), 1)} | Sal: ${l_df['Sal'].sum()}"):
                    st.table(l_df[['Pos', 'Name', 'Team', 'Sal', 'Proj', 'Own']])
        else:
            st.info("Run the engine in the Projections Lab to view lineups.")
else:
    st.warning("Please upload a DraftKings Salary CSV to begin.")
