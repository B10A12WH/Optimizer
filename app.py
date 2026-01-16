import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

st.set_page_config(page_title="VANTAGE-V7.5 PRO", layout="wide", page_icon="📈")

class VantageProV7:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        mapping = {'Name':'Name', 'Salary':'Sal', 'dk_points':'Market_Proj', 'Adj Own':'Own', 'Pos':'Pos', 'Team':'Team', 'Status':'Status'}
        self.raw_df = self.raw_df.rename(columns=mapping)
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)

    def calculate_vantage_grade(self, win_pct, total_own, total_sal):
        """SLATE-AWARE GRADER: Adjusts expectations for small slates."""
        num_teams = self.raw_df['Team'].nunique()
        
        # --- DYNAMIC SCALING ---
        if num_teams <= 4: # Small Slate Calibration
            # We value lower Win% higher because it's harder to get separation
            ceiling = min(win_pct * 28, 45) 
            # Ownership Sweet Spot is shifted lower for elite leverage
            if 60 <= total_own <= 105: leverage = 35 
            elif total_own < 60: leverage = 30 # Elite leverage badge
            else: leverage = 15
        else: # Large Slate Calibration
            ceiling = min(win_pct * 12, 45)
            leverage = 35 if 75 <= total_own <= 105 else 20

        efficiency = (total_sal / 50000) * 20
        score = ceiling + leverage + efficiency
        
        # Adjusted Thresholds for the "A+"
        grades = [(86, "A+"), (78, "A"), (70, "B+"), (62, "B"), (54, "C")]
        for threshold, label in grades:
            if score >= threshold: return label, score
        return "D", score

    def build_pool(self, num_lineups, exp_limit, final_scrubs, leverage_weight, min_sal, min_sharks):
        LATE_TEAMS = ['CHI', 'BKN', 'LAC', 'TOR']
        self.df['Is_Late'] = self.df['Team'].apply(lambda x: 1 if x in LATE_TEAMS else 0)
        filtered_df = self.df[~self.df['Name'].isin(final_scrubs)].reset_index(drop=True)
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            team_shocks = {t: np.random.normal(1, 0.08) for t in sim_df['Team'].unique()}
            sim_df['Sim'] = sim_df['Final_Proj'] * sim_df['Team'].map(team_shocks) * np.random.normal(1, 0.12, len(sim_df))
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / (1 + ((sim_df['Own'] / 100) * leverage_weight))
            
            prob = pulp.LpProblem(f"V7_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sal
            
            sim_df['Is_Shark'] = sim_df['Own'].apply(lambda x: 1 if x < 8 else 0)
            prob += pulp.lpSum([sim_df.loc[i, 'Is_Shark'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sharks
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            
            # Late-Flex 
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
                lineup = {s: sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1}
                l_list = list(lineup.values())
                win_pct, avg_score = self.simulate_win_pct(l_list)
                total_own = sum([p['Own'] for p in l_list])
                grade, score = self.calculate_vantage_grade(win_pct, total_own, sum([p['Sal'] for p in l_list]))
                final_pool.append({'players': lineup, 'metrics': {'Sal': sum([p['Sal'] for p in l_list]), 'Own': round(total_own, 1), 'Win': win_pct, 'Avg': round(avg_score, 1), 'Grade': grade, 'Score': score}})
                curr_idx = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr_idx)
                for i in curr_idx:
                    n_key = sim_df.loc[i, 'Name']
                    player_counts[n_key] = player_counts.get(n_key, 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool

    # ... (Keep generate_proprietary_projections and simulate_win_pct same as V7.4) ...
