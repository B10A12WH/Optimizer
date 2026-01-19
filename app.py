import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io

# --- RESTORED CLASSIC UI CONFIG ---
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

# 3:30 PM Official MLK Day Scratches
OFFICIAL_330_SCRATCHES = [
    "Obi Toppin", "Tyrese Haliburton", "Bennedict Mathurin", "Isaiah Jackson",
    "Darius Garland", "Max Strus", "Sam Merrill", "Dean Wade",
    "Isaiah Hartenstein", "Jalen Williams", "Brooks Barnhizer",
    "Jalen Brunson", "Daniel Gafford", "Dereck Lively II", "Kyrie Irving",
    "Day'Ron Sharpe", "Cam Thomas", "Egor Demin", "Kristaps Porzingis", 
    "Zaccharie Risacher", "Bradley Beal", "Kawhi Leonard", "Bilal Coulibaly"
]

class VantageUnifiedOptimizer:
    def __init__(self, df, sport="NBA", manual_scratches=[]):
        self.df = df.copy()
        self.sport = sport
        self.excluded_names = list(set(OFFICIAL_330_SCRATCHES + manual_scratches))
        self._clean_data()

    def _clean_data(self):
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'avgpoints'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        self.df['Name+ID'] = self.df[self._hunt(['name+id'], cols)].astype(str)
        
        # Injury Filtering
        clean_excludes = [p.strip().lower() for p in self.excluded_names if p.strip()]
        self.df = self.df[~self.df['Name'].str.lower().isin(clean_excludes)]
        self.df = self.df[~self.df['Name+ID'].str.contains(r'\(OUT\)', flags=re.IGNORECASE, na=False)]
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

    def run_alpha_sims(self, n_lineups=20, n_sims=5000, correlation=0.6, leverage=0.4):
        n_p = len(self.df)
        team_list = self.df['Team'].values
        unique_teams = list(set(team_list))
        team_to_idx = {team: i for i, team in enumerate(unique_teams)}
        team_indices = np.array([team_to_idx[t] for t in team_list])
        
        proj_base = self.df['Proj'].values
        own_tax = (1 - (self.df['Own'].values * (leverage / 150)))
        A = [np.ones(n_p), self.df['Sal'].values]
        bl, bu = [8 if self.sport=="NBA" else 9], [8 if self.sport=="NBA" else 9]
        bl.append(45000); bu.append(50000)
        
        lineup_counts = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_sims):
            shifts = np.random.normal(1.0, 0.15 * correlation, len(unique_teams))
            sim_p = proj_base * shifts[team_indices] * np.random.normal(1.0, 0.1, n_p)
            sim_p *= own_tax
            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = tuple(sorted(np.where(res.x > 0.5)[0]))
                lineup_counts[idx] = lineup_counts.get(idx, 0) + 1
            if i % 500 == 0:
                progress_bar.progress((i + 1) / n_sims)
                status_text.text(f"Processed {i}/{n_sims} simulations...")

        status_text.empty()
        if not lineup_counts: return []
        sorted_lineups = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:n_lineups]
        max_freq = sorted_lineups[0][1]
        
        final_pool = []
        for idx, count in sorted_lineups:
            ldf = self.get_dk_slots(self.df.iloc[list(idx)])
            final_pool.append({
                'df': ldf, 
                'win_pct': (count/n_sims)*100, 
                'rel_score': (count/max_freq)*100,
                'proj': ldf['Proj'].sum()
            })
        return final_pool

# --- UI INTERFACE ---
st.sidebar.title("🕹️ COMMAND")
sport_mode = st.sidebar.radio("MODE", ["NBA", "NFL"])
sim_count = st.sidebar.select_slider("SIMULATIONS", options=[1000, 3000, 5000, 10000], value=5000)

# Injury Report Link & Scratch List
st.sidebar.markdown("[NBA Official Injury Report (PDF)](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-01-19_03_30PM.pdf)")
manual_input = st.sidebar.text_area("🚑 EXTRA SCRATCHES", placeholder="One name per line...")
manual_list = manual_input.split('\n') if manual_input else []

uploaded_file = st.sidebar.file_uploader("SALARY CSV", type="csv")

if uploaded_file:
    engine = VantageUnifiedOptimizer(pd.read_csv(uploaded_file), sport=sport_mode, manual_scratches=manual_list)
    tab1, tab2 = st.tabs(["🏆 POOL", "📊 EXPOSURE"])
    
    with tab1:
        if st.button(f"⚡ RUN {sim_count} ALPHA SCRIPTS"):
            st.session_state.results = engine.run_alpha_sims(n_sims=sim_count)
            
        if 'results' in st.session_state:
            for i, res in enumerate(st.session_state.results):
                # Grading labels for the restored UI
                score = res['rel_score']
                grade = "ELITE" if score > 85 else "STRONG" if score > 50 else "STANDARD"
                
                with st.expander(f"LINEUP #{i+1} | {grade} | Win: {round(res['win_pct'], 2)}% | Proj: {round(res['proj'], 1)}"):
                    st.table(res['df'][['Slot', 'Name', 'Team', 'Sal', 'Proj']])
                    
    with tab2:
        if 'results' in st.session_state:
            st.subheader("Global Player Exposure")
            all_players = pd.concat([res['df'] for res in st.session_state.results])
            exp_df = all_players['Name'].value_counts().reset_index()
            exp_df.columns = ['Player', 'Lineups']
            exp_df['Exposure %'] = (exp_df['Lineups'] / len(st.session_state.results)) * 100
            st.dataframe(exp_df, use_container_width=True)
