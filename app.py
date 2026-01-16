import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

st.set_page_config(page_title="VANTAGE-V7.2 PRO", layout="wide", page_icon="🛡️")

class VantageProV7:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        mapping = {
            'Name': 'Name', 'Salary': 'Sal', 'dk_points': 'Market_Proj', 
            'Adj Own': 'Own', 'Pos': 'Pos', 'Team': 'Team', 'Status': 'Status'
        }
        self.raw_df = self.raw_df.rename(columns=mapping)
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)

    def auto_calibrate_settings(self):
        num_teams = self.raw_df['Team'].nunique()
        if num_teams <= 4:
            return 1.35, 0.80, 0.45, 1 
        elif num_teams <= 10:
            return 1.10, 0.75, 0.55, 1
        else:
            return 0.85, 0.70, 0.60, 0

    def calculate_vantage_grade(self, win_pct, total_own, total_sal):
        ceiling = min(win_pct * 12, 45) 
        if 70 <= total_own <= 105: leverage = 35 
        elif 105 < total_own <= 130: leverage = 25 
        else: leverage = 15 
        efficiency = (total_sal / 50000) * 20
        score = ceiling + leverage + efficiency
        if score >= 90: return "A+", score
        elif score >= 82: return "A", score
        elif score >= 75: return "B+", score
        elif score >= 68: return "B", score
        else: return "C", score

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        self.df = self.raw_df.copy()
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
        return (wins / sims) * 100 if wins > 0 else round((max(scores) / target) * 0.45, 2), np.mean(scores)

    def build_pool(self, num_lineups, exp_limit, final_scrubs, leverage_weight, min_sal, min_sharks):
        filtered_df = self.df[~self.df['Name'].isin(final_scrubs)].reset_index(drop=True)
        
        # --- LATE FLEX IDENTIFICATION ---
        # Teams playing at 7:30 PM
        LATE_TEAMS = ['CHI', 'BKN', 'LAC', 'TOR']
        filtered_df['Is_Late'] = filtered_df['Team'].apply(lambda x: 1 if x in LATE_TEAMS else 0)
        
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            team_shocks = {t: np.random.normal(1, 0.08) for t in sim_df['Team'].unique()}
            sim_df['Sim'] = sim_df['Final_Proj'] * sim_df['Team'].map(team_shocks) * np.random.normal(1, 0.12, len(sim_df))
            sim_df['Lev_Fact'] = 1 + ((sim_df['Own'] / 100) * leverage_weight)
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / sim_df['Lev_Fact']
            
            prob = pulp.LpProblem(f"V7_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            
            # --- CONSTRAINTS ---
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sal
            
            # 🛡️ THE LATE-FLEX ANCHOR RULE
            # Logic: If a lineup contains ANY late-game players, one MUST be in the UTIL slot.
            for i in sim_df.index:
                if sim_df.loc[i, 'Is_Late'] == 0:
                    # If you aren't a late player, you can't be in UTIL if a late player is available in the lineup.
                    # We implement this by saying: lpSum(Late Players in UTIL) >= lpSum(Late Player i in Any Slot)
                    pass 

            # Simpler implementation: UTIL can ONLY be a Late Player if at least one is picked.
            # We use a helper variable 'has_late'
            has_late = pulp.LpVariable(f"has_late_{n}", 0, 1, cat='Binary')
            prob += pulp.lpSum([choices[i][s] for i in sim_df.index if sim_df.loc[i, 'Is_Late'] == 1 for s in slots]) >= has_late
            prob += pulp.lpSum([choices[i]['UTIL'] for i in sim_df.index if sim_df.loc[i, 'Is_Late'] == 1]) >= has_late

            # Standard Position Rules
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
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
                total_sal = sum([p['Sal'] for p in l_list])
                grade, score = self.calculate_vantage_grade(win_pct, total_own, total_sal)
                final_pool.append({'players': lineup, 'metrics': {'Sal': total_sal, 'Own': round(total_own, 1), 'Win': win_pct, 'Avg': round(avg_score, 1), 'Grade': grade, 'Score': score}})
                curr_idx = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr_idx)
                for i in curr_idx:
                    name = sim_df.loc[i, 'Name']
                    player_counts[name] = player_counts.get(name, 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool

# --- UI SECTION ---
# (Keep same as V7.1)
