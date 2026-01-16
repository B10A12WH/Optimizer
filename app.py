import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

# --- FULL SYSTEM CONFIG ---
st.set_page_config(page_title="VANTAGE-V7.5 PRO", layout="wide", page_icon="📈")

class VantageProV7:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        
        mapping = {
            'Name': 'Name', 'Salary': 'Sal', 'dk_points': 'Market_Proj', 
            'Adj Own': 'Own', 'Pos': 'Pos', 'Team': 'Team', 'Status': 'Status'
        }
        self.raw_df = self.raw_df.rename(columns=mapping)
        
        # Guard rails for data types
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)
        
        self.df = self.raw_df.copy()

    def auto_calibrate_settings(self):
        """THE BRAIN: Detects slate context and recommends settings."""
        num_teams = self.raw_df['Team'].nunique()
        # Small slate (4 games) needs aggressive leverage
        if num_teams <= 4:
            return 1.45, 0.80, 0.45, 2 # Lev, Alpha, Exp, Sharks
        return 1.10, 0.75, 0.55, 1

    def calculate_vantage_grade(self, win_pct, total_own, total_sal):
        """SLATE-AWARE GRADER: Adjusts expectations for small slates."""
        num_teams = self.raw_df['Team'].nunique()
        
        if num_teams <= 4: # Small Slate Calibration
            ceiling = min(win_pct * 28, 45) 
            if 60 <= total_own <= 105: leverage = 35 
            elif total_own < 60: leverage = 30 
            else: leverage = 15
        else: # Large Slate Calibration
            ceiling = min(win_pct * 12, 45)
            leverage = 35 if 75 <= total_own <= 105 else 20

        efficiency = (total_sal / 50000) * 20
        score = ceiling + leverage + efficiency
        
        grades = [(86, "A+"), (78, "A"), (70, "B+"), (62, "B"), (54, "C")]
        for threshold, label in grades:
            if score >= threshold: return label, score
        return "D", score

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            return (row['Market_Proj'] * boost * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

    def simulate_win_pct(self, lineup_players):
        sims = 200
        teams = list(set([p['Team'] for p in lineup_players]))
        team_variance = {t: np.random.normal(1.0, 0.10) for t in teams}
        scores = [sum([p['Final_Proj'] * team_variance.get(p['Team'], 1.0) * np.random.normal(1.0, 0.15) for p in lineup_players]) for _ in range(sims)]
        target = 305 
        wins = sum(1 for s in scores if s >= target)
        if wins > 0:
            return (wins / sims) * 100, np.mean(scores)
        return round((max(scores) / target) * 0.45, 2), np.mean(scores)

    def build_pool(self, num_lineups, exp_limit, final_scrubs, leverage_weight, min_sal, min_sharks):
        # Late-Game Teams for UTIL anchoring
        LATE_TEAMS = ['CHI', 'BKN', 'LAC', 'TOR']
        self.df['Is_Late'] = self.df['Team'].apply(lambda x: 1 if x in LATE_TEAMS else 0)
        
        filtered_df = self.df[~self.df['Name'].isin(final_scrubs)].reset_index(drop=True)
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            # Correlation shocks
            team_shocks = {t: np.random.normal(1, 0.08) for t in sim_df['Team'].unique()}
            sim_df['Sim'] = sim_df['Final_Proj'] * sim_df['Team'].map(team_shocks) * np.random.normal(1, 0.12, len(sim_df))
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / (1 + ((sim_df['Own'] / 100) * leverage_weight))
            
            prob = pulp.LpProblem(f"V7_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            # Objective
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            
            # Constraints
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sal
            
            # Shark Rule
            sim_df['Is_Shark'] = sim_df['Own'].apply(lambda x: 1 if x < 8 else 0)
            prob += pulp.lpSum([sim_df.loc[i, 'Is_Shark'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sharks

            # Position/Slot Logic
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            
            # Late-Flex Anchor Rule
            for i in sim_df.index:
                if sim_df.loc[i, 'Is_Late'] == 0:
                    prob += choices[i]['UTIL'] <= 1 - (pulp.lpSum([choices[j][s] for j in sim_df.index if sim_df.loc[j, 'Is_Late'] == 1 for s in slots]) / 8)

            # Max 3 per team
            for t in sim_df['Team'].unique():
                prob += pulp.lpSum([choices[i][s] for i in sim_df.index if sim_df.loc[i, 'Team'] == t for s in slots]) <= 3

            # Diversity & Exposure
            for prev in indices_store: prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            
            # Pos Eligibility
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or \
                               (s == 'PG' and 'PG' in p_pos) or (s == '
