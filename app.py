import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io

# --- ELITE UI & MULTI-SPORT CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | DFS COMMAND", layout="wide", page_icon="⚡")

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
        # Auto-detect common CSV headers using your 'hunt' logic
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        
        # RECOMMENDED FIX: Changed .fillna(5.0) to .fillna(0.0) to stop 'ghost' punts
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'avgpoints'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        
        # STRICTURE: Purge anyone with negligible projections to ensure stability
        self.df = self.df[self.df['Proj'] > 0.5].reset_index(drop=True)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def run_alpha_sims(self, n_lineups=20, correlation=0.5, leverage=0.5):
        n_p = len(self.df)
        teams = self.df['Team'].unique()
        
        # Define DraftKings Position Constraints
        A, bl, bu = [], [], []
        A.append(np.ones(n_p)); bl.append(8 if self.sport=="NBA" else 9); bu.append(8 if self.sport=="NBA" else 9)
        A.append(self.df['Sal'].values); bl.append(45000); bu.append(50000)
        
        if self.sport == "NFL":
            for p in ['QB','RB','WR','TE','DST']:
                A.append((self.df['Pos'] == p).astype(int).values)
                if p == 'QB': bl.append(1); bu.append(1)
                elif p == 'RB': bl.append(2); bu.append(3)
                elif p == 'WR': bl.append(3); bu.append(4)
                elif p == 'TE': bl.append(1); bu.append(2)
                else: bl.append(1); bu.append(1)
        else: # NBA Position logic
            for p in ['PG','SG','SF','PF','C']:
                A.append(self.df['Pos'].str.contains(p).astype(int).values)
                bl.append(1); bu.append(5)

        lineup_pool = []
        with st.status(f"🚀 EXECUTING {self.sport} GAME SCRIPTS...", expanded=False):
            for i in range(1000):
                # RECOMMENDED FIX: CORRELATED SIMULATION (SaberSim Mimic)
                # Shift whole teams together to capture game scripts
                team_shift = {t: np.random.normal(1.0, 0.15 * correlation) for t in teams}
                sim_projs = np.array([
                    row['Proj'] * team_shift[row['Team']] * np.random.normal(1.0, 0.1) 
                    for _, row in self.df.iterrows()
                ])
                
                # Ownership Leverage (Stokastic Mimic 'Chalk Tax')
                sim_projs = sim_projs * (1 - (self.df['Own'].values * (leverage / 150)))

                res = milp(c=-sim_projs, constraints=LinearConstraint(np.vstack(A), bl, bu),
                           integrality=np.ones(n_p), bounds=Bounds(0, 1))
                
                if res.success:
                    idx = tuple(np.where(res.x > 0.5)[0])
                    lineup_pool.append({'idx': idx, 'sim_score': sim_projs[list(idx)].sum()})
                if len(lineup_pool) >= n_lineups * 3: break
        
        unique_lineups = {entry['idx']: entry for entry in lineup_pool}.values()
        sorted_pool = sorted(unique_lineups, key=lambda x: x['sim_score'], reverse=True)[:n_lineups]
        return [self.df.iloc[list(entry['idx'])] for entry in sorted_pool]

    def run_diagnostic(self, entries_df):
        # RECOMMENDED FIX: REGEX CLEANER (Strip DK Player IDs)
        def clean_name(val):
            return re.sub(r'\s*\(\d+\)', '', str(val)).strip()
        
        roster_cols = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST'] if self.sport == "NFL" else \
                      ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        available_cols = [c for c in roster_cols if c in entries_df.columns]
        user_lineups = entries_df[available_cols].applymap(clean_name)
        all_players = user_lineups.values.flatten()
        counts = pd.Series(all_players).value_counts()
        
        diag_df = self.df.copy()
        diag_df['My %'] = (diag_df['Name'].map(counts).fillna(0) / len(entries_df)) * 100
        diag_df['Leverage'] = diag_df['My %'] - diag_df['Own']
        return diag_df[diag_df['My %'] > 0].sort_values(by='My %', ascending=False)

# --- CSV ROBUST PARSER ---
def robust_read_csv(file):
    try:
        content = file.getvalue().decode('utf-8')
        # Skip DraftKings instructional metadata
        clean_content = content.split('\n\n')[0] 
        return pd.read_csv(io.StringIO(clean_content), on_bad_lines='skip')
    except Exception as e:
        st.error(f"Entry File Error: {e}")
        return None

# --- MAIN APP ---
st.sidebar.title("🕹️ CONTROL CENTER")
mode = st.sidebar.radio("THEATRE", ["NBA", "NFL"])
uploaded_file = st.sidebar.file_uploader("1. PROJECTIONS CSV", type="csv")
entry_file = st.sidebar.file_uploader("2. ENTRIES CSV (Optional)", type="csv")
st.sidebar.divider()
corr_val = st.sidebar.slider("CORRELATION (Stacking)", 0.0, 1.0, 0.6)
own_val = st.sidebar.slider("LEVERAGE (Chalk Tax)", 0.0, 1.0, 0.4)

if uploaded_file:
    engine = VantageUnifiedOptimizer(pd.read_csv(uploaded_file), sport=mode)
    tab1, tab2, tab3 = st.tabs(["📊 LAB", "🏆 POOL", "🔍 DIAGNOSTIC"])
    
    with tab1:
        st.dataframe(engine.df[['Pos', 'Name', 'Team', 'Sal', 'Proj', 'Own']], use_container_width=True)
        if st.button(f"⚡ RUN {mode} ALPHA"):
            st.session_state.results = engine.run_alpha_sims(correlation=corr_val, leverage=own_val)
    
    with tab2:
        if 'results' in st.session_state:
            for i, ldf in enumerate(st.session_state.results):
                with st.expander(f"LINEUP #{i+1} | Proj: {round(ldf['Proj'].sum(), 1)}"):
                    st.table(ldf[['Pos', 'Name', 'Team', 'Sal', 'Proj', 'Own']])
    
    with tab3:
        if entry_file:
            entries_df = robust_read_csv(entry_file)
            if entries_df is not None:
                diag_results = engine.run_diagnostic(entries_df)
                st.subheader("Field Leverage Report")
                st.dataframe(diag_results[['Name', 'Team', 'Own', 'My %', 'Leverage']], use_container_width=True)
        else:
            st.info("Upload your 'Entries.csv' to analyze your exposure vs. the field.")
