import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io

# --- ELITE DYNAMIC UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | DYNAMIC ALPHA", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background: radial-gradient(circle at top right, #1a1f2e, #0d1117); color: #c9d1d9; font-family: 'Inter', sans-serif; }
    
    /* DYNAMIC COLOR CLASSES */
    .card-elite { border: 2px solid #238636 !important; background: rgba(35, 134, 54, 0.15); box-shadow: 0 0 20px rgba(35, 134, 54, 0.4); }
    .card-strong { border: 2px solid #d29922 !important; background: rgba(210, 153, 34, 0.1); }
    .card-standard { border: 1px solid #30363d !important; background: rgba(22, 27, 34, 0.5); }

    .lineup-card { border-radius: 12px; padding: 20px; margin-bottom: 20px; transition: 0.3s; }
    .badge-label { padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: bold; color: white; margin-left: 5px; }
    .bg-win { background: #238636; }
    .bg-proj { background: #1f6feb; }
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
        sorted_lineups = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:n_lineups]
        
        # DYNAMIC SCORING: Calculate confidence based on the top lineup's frequency
        max_freq = sorted_lineups[0][1] if sorted_lineups else 1
        
        final_pool = []
        for idx, count in sorted_lineups:
            ldf = self.get_dk_slots(self.df.iloc[list(idx)])
            # Relative confidence score (0 to 100)
            rel_confidence = (count / max_freq) * 100
            final_pool.append({
                'df': ldf, 
                'win_pct': (count/n_sims)*100, 
                'rel_score': rel_confidence,
                'proj': ldf['Proj'].sum()
            })
        return final_pool

# --- UI INTERFACE ---
st.sidebar.title("🕹️ COMMAND")
sport_mode = st.sidebar.radio("MODE", ["NBA", "NFL"])
sim_count = st.sidebar.select_slider("SIMULATIONS", options=[1000, 3000, 5000, 10000], value=5000)
uploaded_file = st.sidebar.file_uploader("SALARY CSV", type="csv")

if uploaded_file:
    engine = VantageUnifiedOptimizer(pd.read_csv(uploaded_file), sport=sport_mode)
    if st.button(f"⚡ RUN {sim_count} ALPHA SCRIPTS"):
        st.session_state.results = engine.run_alpha_sims(n_sims=sim_count)
        
if 'results' in st.session_state:
    cols = st.columns(2)
    for i, res in enumerate(st.session_state.results):
        # DYNAMIC GRADING: Based on relative performance to the #1 lineup
        score = res['rel_score']
        card_class = "card-elite" if score > 85 else "card-strong" if score > 50 else "card-standard"
        
        with cols[i % 2]:
            st.markdown(f"""
            <div class="lineup-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-weight: bold; font-size: 16px;">LINEUP #{i+1}</span>
                    <div>
                        <span class="badge-label bg-win">WIN: {round(res['win_pct'], 2)}%</span>
                        <span class="badge-label bg-proj">PROJ: {round(res['proj'], 1)}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.table(res['df'][['Slot', 'Name', 'Team', 'Sal', 'Proj']])
