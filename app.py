import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io
import time

st.set_page_config(page_title="VANTAGE-V11.1 IRONCLAD", layout="wide", page_icon="🏎️")

class VantageProV11:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        mapping = {'Name':'Name', 'Salary':'Sal', 'dk_points':'Market_Proj', 'Adj Own':'Own', 'Pos':'Pos', 'Team':'Team'}
        self.raw_df = self.raw_df.rename(columns=mapping)
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)
        self.df = self.raw_df.copy()

    def simulate_win_pct_institutional(self, lineup_players, num_sims):
        projs = np.array([p['Final_Proj'] for p in lineup_players])
        teams = np.array([p['Team'] for p in lineup_players])
        player_variance = np.random.normal(1.0, 0.20, (num_sims, 8))
        
        # Usage Cannibalism Logic
        for team in np.unique(teams):
            teammate_indices = np.where(teams == team)[0]
            if len(teammate_indices) > 1:
                boom_mask = player_variance[:, teammate_indices[0]] > 1.25
                for other_idx in teammate_indices[1:]:
                    player_variance[boom_mask, other_idx] *= 0.88 

        sim_results = np.sum(projs * player_variance, axis=1)
        target = 295 
        wins = np.sum(sim_results >= target)
        return (wins / num_sims) * 100

    def build_pool(self, num_lineups, exp_limit, late_teams, team_limit, leverage_weight, sim_strength):
        LATE_TEAMS = late_teams if late_teams else ['CHI', 'BKN', 'LAC', 'TOR']
        self.df['Is_Late'] = self.df['Team'].apply(lambda x: 1 if x in LATE_TEAMS else 0)
        
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = self.df.copy()
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.12, len(sim_df))
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / (1 + ((sim_df['Own'] / 100) * leverage_weight))
            
            prob = pulp.LpProblem(f"V11_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= 49000
            
            # Late-Swap Logic
            if any(sim_df['Is_Late'] == 1):
                prob += pulp.lpSum([choices[i]['UTIL'] for i in sim_df.index if sim_df.loc[i, 'Is_Late'] == 1]) == 1
            
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            for t in sim_df['Team'].unique():
                prob += pulp.lpSum([choices[i][s] for i in sim_df.index if sim_df.loc[i, 'Team'] == t for s in slots]) <= team_limit
            for prev in indices_store: prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or (s == 'PG' and 'PG' in p_pos) or (s == 'SG' and 'SG' in p_pos) or (s == 'SF' and 'SF' in p_pos) or (s == 'PF' and 'PF' in p_pos) or (s == 'C' and 'C' in p_pos) or (s == 'G' and ('PG' in p_pos or 'SG' in p_pos)) or (s == 'F' and ('SF' in p_pos or 'PF' in p_pos))
                    if not eligible: prob += choices[i][s] == 0

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
            if pulp.LpStatus[prob.status] == 'Optimal':
                l_list = [sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1]
                win_pct = self.simulate_win_pct_institutional(l_list, sim_strength)
                final_pool.append({'players': l_list, 'metrics': {'Win': round(win_pct, 2), 'Own': sum([p['Own'] for p in l_list]), 'Sal': sum([p['Sal'] for p in l_list])}})
                indices_store.append([i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)])
                for p in l_list: player_counts[p['Name']] = player_counts.get(p['Name'], 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool

# --- UI SECTION ---
st.title("🏎️ VANTAGE-V11.1 IRONCLAD")
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    st.info(f"📋 Data Loaded: {len(raw_data)} players found.")
    engine = VantageProV11(raw_data)
    
    st.sidebar.header("🕹️ Parameters")
    late_teams = st.sidebar.multiselect("Late Games", ['CLE', 'PHI', 'IND', 'NOP', 'CHI', 'BKN', 'LAC', 'TOR'], default=['CHI', 'BKN', 'LAC', 'TOR'])
    team_limit = st.sidebar.slider("Max Per Team", 1, 4, 3)
    exp_limit = st.sidebar.slider("Exp Cap", 0.1, 1.0, 0.6)
    
    if st.button("🔥 GENERATE PORTFOLIO"):
        with st.spinner("Analyzing Slate Correlation & Simulations..."):
            engine.df['Final_Proj'] = engine.df['Market_Proj']
            # Manual Alpha Boosts
            engine.df.loc[engine.df['Name'] == 'Donovan Mitchell', 'Final_Proj'] *= 1.22
            engine.df.loc[engine.df['Name'] == 'Scottie Barnes', 'Final_Proj'] *= 1.28
            
            pool = engine.build_pool(15, exp_limit, late_teams, team_limit, 1.2, 5000)
            
            if not pool:
                st.error("❌ No lineups found. Try lowering 'Max Per Team' or increasing 'Exp Cap'.")
            else:
                for i, l in enumerate(pool):
                    st.write(f"Lineup #{i+1} | Win%: {l['metrics']['Win']}% | Sal: ${l['metrics']['Sal']}")
                    st.table(pd.DataFrame(l['players'])[['Name', 'Team', 'Sal', 'Own']])
