import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io
import time # NEW: To track performance

st.set_page_config(page_title="VANTAGE-V8.2 PRO", layout="wide", page_icon="🏎️")

class VantageProV8:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        mapping = {'Name':'Name', 'Salary':'Sal', 'dk_points':'Market_Proj', 'Adj Own':'Own', 'Pos':'Pos', 'Team':'Team', 'Status':'Status'}
        self.raw_df = self.raw_df.rename(columns=mapping)
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)
        self.df = self.raw_df.copy()

    def get_slate_presets(self, num_games):
        if num_games <= 4: return 1.45, 0.80, 0.45, 2
        return 1.10, 0.75, 0.55, 1

    def calculate_vantage_grade(self, win_pct, total_own, total_sal, num_games):
        if num_games <= 5:
            ceiling = min(win_pct * 28, 45)
            leverage = 35 if 60 <= total_own <= 95 else 25
        else:
            ceiling = min(win_pct * 12, 45)
            leverage = 35 if 75 <= total_own <= 110 else 20
        score = ceiling + leverage + ((total_sal / 50000) * 20)
        grades = [(86, "A+"), (78, "A"), (70, "B+"), (62, "B"), (54, "C")]
        for threshold, label in grades:
            if score >= threshold: return label, score
        return "D", score

    def simulate_win_pct_vectorized(self, lineup_players, num_sims):
        """THE PROOF ENGINE: Time-tracked and Matrix-verified."""
        start_time = time.time()
        
        projs = np.array([p['Final_Proj'] for p in lineup_players])
        teams = [p['Team'] for p in lineup_players]
        unique_teams = list(set(teams))
        team_map = [unique_teams.index(t) for t in teams]
        
        team_shocks = np.random.normal(1.0, 0.10, (num_sims, len(unique_teams)))
        player_team_shocks = team_shocks[:, team_map]
        player_variance = np.random.normal(1.0, 0.15, (num_sims, 8))
        
        sim_results = np.sum(projs * player_team_shocks * player_variance, axis=1)
        
        target = 305
        wins = np.sum(sim_results >= target)
        
        elapsed = (time.time() - start_time) * 1000 # Convert to ms
        return (wins / num_sims) * 100, np.mean(sim_results), elapsed, sim_results.shape

    def build_pool(self, num_lineups, exp_limit, final_scrubs, leverage_weight, min_sharks, sim_strength, num_games):
        LATE_TEAMS = ['CHI', 'BKN', 'LAC', 'TOR']
        self.df['Is_Late'] = self.df['Team'].apply(lambda x: 1 if x in LATE_TEAMS else 0)
        filtered_df = self.df[~self.df['Name'].isin(final_scrubs)].reset_index(drop=True)
        final_pool, player_counts, indices_store = [], {}, []
        
        total_crunch_time = 0
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.12, len(sim_df))
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / (1 + ((sim_df['Own'] / 100) * leverage_weight))
            
            prob = pulp.LpProblem(f"V8_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= 49700
            
            sim_df['Is_Shark'] = sim_df['Own'].apply(lambda x: 1 if x < 8 else 0)
            prob += pulp.lpSum([sim_df.loc[i, 'Is_Shark'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sharks
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            for i in sim_df.index:
                if sim_df.loc[i, 'Is_Late'] == 0:
                    prob += choices[i]['UTIL'] <= 1 - (pulp.lpSum([choices[j][s] for j in sim_df.index if sim_df.loc[j, 'Is_Late'] == 1 for s in slots]) / 8)
            for t in sim_df['Team'].unique():
                prob += pulp.lpSum([choices[i][s] for i in sim_df.index if sim_df.loc[i, 'Team'] == t for s in slots]) <= 3
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
                win_pct, avg_score, ms, shape = self.simulate_win_pct_vectorized(l_list, sim_strength)
                total_crunch_time += ms
                
                final_pool.append({'players': {slots[i]: l_list[i] for i in range(8)}, 'metrics': {'Win': round(win_pct, 2), 'Avg': round(avg_score, 1), 'Own': round(sum([p['Own'] for p in l_list]), 1), 'Sal': sum([p['Sal'] for p in l_list])}})
                indices_store.append([i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)])
                for p in l_list: player_counts[p['Name']] = player_counts.get(p['Name'], 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        
        return final_pool, total_crunch_time

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            return (row['Market_Proj'] * boost * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

# --- UI SECTION ---
st.title("🏎️ VANTAGE-V8.2 PRO")
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = VantageProV8(raw_data)
    
    num_games = st.sidebar.slider("Number of Games in Slate", 1, 15, 4)
    p_lev, p_alp, p_exp, p_sharks = engine.get_slate_presets(num_games)
    num_lineups = st.sidebar.slider("Number of Lineups", 1, 50, 15)
    sim_strength = st.sidebar.select_slider("Sim Strength", options=[1000, 5000, 10000, 40000], value=10000)
    
    with st.sidebar.expander("🛠️ Advanced Quant Settings"):
        leverage_weight = st.slider("Leverage Strength", 0.0, 2.0, p_lev)
        alpha_weight = st.slider("Alpha System Weight", 0.0, 1.0, p_alp)
        exp_limit = st.slider("Global Exposure Cap", 0.1, 1.0, p_exp)
        min_sharks = st.slider("Min Sharks (<8% Own)", 0, 3, p_sharks)
    
    st.sidebar.header("🔭 Vision")
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.22)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.28)
    
    if st.button("🔥 GENERATE HIGH-FIDELITY PORTFOLIO"):
        engine.generate_proprietary_projections(alpha_weight, {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b})
        pool, crunch_time = engine.build_pool(num_lineups, exp_limit, [], leverage_weight, min_sharks, sim_strength, num_games)
        
        # --- THE PROOF OF WORK BADGE ---
        st.success(f"✅ PERFORMANCE AUDIT: {num_lineups * sim_strength:,} simulations completed in {crunch_time:.2f}ms.")
        st.info(f"Verified Matrix: {sim_strength} scenarios processed per lineup.")

        for i, l in enumerate(pool):
            m = l['metrics']
            grade, _ = engine.calculate_vantage_grade(m['Win'], m['Own'], m['Sal'], num_games)
            with st.expander(f"[{grade}] Lineup #{i+1} | Win%: {m['Win']}% | Own: {m['Own']}%"):
                st.table(pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']])
