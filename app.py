import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io
import time

st.set_page_config(page_title="V11.2 LOCKDOWN PRO", layout="wide", page_icon="🏎️")

class VantageProV11:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        # Professional Mapping for SaberSim CSV
        mapping = {'Name':'Name', 'Salary':'Sal', 'dk_points':'Market_Proj', 'Adj Own':'Own', 'Pos':'Pos', 'Team':'Team'}
        self.raw_df = self.raw_df.rename(columns=mapping)
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)
        self.df = self.raw_df.copy()

    def simulate_win_pct_institutional(self, lineup_players, num_sims):
        """Usage Cannibalism Logic (Inverse Correlation)"""
        projs = np.array([p['Final_Proj'] for p in lineup_players])
        teams = np.array([p['Team'] for p in lineup_players])
        player_variance = np.random.normal(1.0, 0.20, (num_sims, 8))
        
        # Inverse Correlation Logic: Boom for one = Penalty for others on same team
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
            # Stochastic Game Sample
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.12, len(sim_df))
            # The Shark Leverage Formula
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / (1 + ((sim_df['Own'] / 100) * leverage_weight))
            
            prob = pulp.LpProblem(f"V11_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= 49200
            
            # Late-Swap Logic: Mandate Late UTIL
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
            
            # Positional Logic
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or (s == 'PG' and 'PG' in p_pos) or (s == 'SG' and 'SG' in p_pos) or (s == 'SF' and 'SF' in p_pos) or (s == 'PF' and 'PF' in p_pos) or (s == 'C' and 'C' in p_pos) or (s == 'G' and ('PG' in p_pos or 'SG' in p_pos)) or (s == 'F' and ('SF' in p_pos or 'PF' in p_pos))
                    if not eligible: prob += choices[i][s] == 0

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
            if pulp.LpStatus[prob.status] == 'Optimal':
                l_list = [sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1]
                win_pct = self.simulate_win_pct_institutional(l_list, sim_strength)
                final_pool.append({'players': {slots[k]: l_list[k] for k in range(8)}, 'metrics': {'Win': round(win_pct, 2), 'Own': sum([p['Own'] for p in l_list]), 'Sal': sum([p['Sal'] for p in l_list])}})
                indices_store.append([i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)])
                for p in l_list: player_counts[p['Name']] = player_counts.get(p['Name'], 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool

# --- UI SECTION ---
st.title("🏎️ VANTAGE-V11.2 LOCKDOWN PRO")
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    st.info(f"📋 System Online. {len(raw_data)} assets in pool.")
    engine = VantageProV11(raw_data)
    
    st.sidebar.header("📊 Game State")
    num_lineups = st.sidebar.slider("Portfolio Size", 1, 50, 15)
    late_teams = st.sidebar.multiselect("Late Games (7:30 PM)", ['CLE', 'PHI', 'IND', 'NOP', 'CHI', 'BKN', 'LAC', 'TOR'], default=['CHI', 'BKN', 'LAC', 'TOR'])
    
    with st.sidebar.expander("🛠️ Advanced Quant Knobs"):
        sim_strength = st.select_slider("Sim Strength", options=[1000, 5000, 10000, 40000], value=10000)
        leverage_weight = st.slider("Leverage (Shark) Strength", 0.0, 2.0, 1.2)
        alpha_weight = st.slider("Alpha Blend Weight", 0.0, 1.0, 0.75)
        team_limit = st.slider("Max Per Team", 1, 4, 3)
        exp_limit = st.slider("Exposure Cap", 0.1, 1.0, 0.6)

    st.sidebar.header("🔭 Alpha Vision")
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.22)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.28)
    
    if st.button("🔥 GENERATE HIGH-FIDELITY PORTFOLIO"):
        with st.spinner("Crunching Usage Cannibalism & Monte Carlo..."):
            # Blend Projections
            engine.df['Final_Proj'] = (engine.df['Market_Proj'] * alpha_weight) + (engine.df['Market_Proj'] * (1-alpha_weight))
            # Apply Vision Boosts
            engine.df.loc[engine.df['Name'] == 'Donovan Mitchell', 'Final_Proj'] *= mitchell_b
            engine.df.loc[engine.df['Name'] == 'Scottie Barnes', 'Final_Proj'] *= barnes_b
            
            pool = engine.build_pool(num_lineups, exp_limit, late_teams, team_limit, leverage_weight, sim_strength)
            
            if not pool:
                st.error("❌ Optimization failed. Loosen constraints (Exposure/Team Limit).")
            else:
                for i, l in enumerate(pool):
                    m = l['metrics']
                    with st.expander(f"Lineup #{i+1} | Win%: {m['Win']}% | Sal: ${m['Sal']}"):
                        st.table(pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']])
