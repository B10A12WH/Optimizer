import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io
import time

st.set_page_config(page_title="VANTAGE V12.2 TOTAL PACKAGE", layout="wide", page_icon="🏎️")

class VantageMaster:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        mapping = {'Name':'Name', 'Salary':'Sal', 'dk_points':'Market_Proj', 'Adj Own':'Own', 'Pos':'Pos', 'Team':'Team'}
        self.raw_df = self.raw_df.rename(columns=mapping)
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)
        self.df = self.raw_df.copy()

    # --- RESTORED PROPRIETARY METHOD ---
    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            return (row['Market_Proj'] * boost * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

    def calculate_vantage_grade(self, win_pct, total_own, total_sal, num_games):
        ceiling = min(win_pct * 28, 45) if num_games <= 5 else min(win_pct * 12, 45)
        leverage = 35 if (num_games <= 5 and 60 <= total_own <= 95) or (num_games > 5 and 75 <= total_own <= 110) else 25
        score = ceiling + leverage + ((total_sal / 50000) * 20)
        grades = [(86, "A+"), (78, "A"), (70, "B+"), (62, "B"), (54, "C")]
        for threshold, label in grades:
            if score >= threshold: return label, score
        return "D", score

    def simulate_win_pct_vectorized(self, lineup_players, num_sims):
        start_time = time.time()
        projs = np.array([p['Final_Proj'] for p in lineup_players])
        teams = np.array([p['Team'] for p in lineup_players])
        player_variance = np.random.normal(1.0, 0.20, (num_sims, 8))
        for team in np.unique(teams):
            teammate_indices = np.where(teams == team)[0]
            if len(teammate_indices) > 1:
                boom_mask = player_variance[:, teammate_indices[0]] > 1.25
                for other_idx in teammate_indices[1:]:
                    player_variance[boom_mask, other_idx] *= 0.88 
        sim_results = np.sum(projs * player_variance, axis=1)
        target = 295 
        wins = np.sum(sim_results >= target)
        elapsed = (time.time() - start_time) * 1000 
        return (wins / num_sims) * 100, np.mean(sim_results), elapsed

    def build_pool(self, num_lineups, exp_limit, late_teams, team_limit, leverage_weight, sim_strength):
        LATE_TEAMS = late_teams if late_teams else ['CHI', 'BKN', 'LAC', 'TOR']
        self.df['Is_Late'] = self.df['Team'].apply(lambda x: 1 if x in LATE_TEAMS else 0)
        final_pool, player_counts, indices_store, total_crunch_time = [], {}, [], 0
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = self.df.copy()
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.12, len(sim_df))
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / (1 + ((sim_df['Own'] / 100) * leverage_weight))
            prob = pulp.LpProblem(f"V12_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= 49500
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
                win_pct, avg, ms = self.simulate_win_pct_vectorized(l_list, sim_strength)
                total_crunch_time += ms
                final_pool.append({'players': {slots[k]: l_list[k] for k in range(8)}, 'metrics': {'Win': round(win_pct, 2), 'Own': round(sum([p['Own'] for p in l_list]), 1), 'Sal': sum([p['Sal'] for p in l_list])}})
                indices_store.append([i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)])
                for p in l_list: player_counts[p['Name']] = player_counts.get(p['Name'], 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool, total_crunch_time

# --- UI LAYER ---
st.title("📈 VANTAGE MASTER V12.2")
f1 = st.file_uploader("1. SaberSim CSV", type="csv")
f2 = st.file_uploader("2. DK Contest CSV", type="csv")

if f1:
    raw = pd.read_csv(f1)
    engine = VantageMaster(raw)
    num_games = st.sidebar.slider("Number of Games", 1, 15, 4)
    num_lineups = st.sidebar.slider("Portfolio Size", 1, 50, 15)
    
    if st.button("🔥 EXECUTE FULL STACK AUDIT"):
        # Correctly calling the restored method
        engine.generate_proprietary_projections(0.75, {'Donovan Mitchell': 1.22, 'Scottie Barnes': 1.28})
        pool, crunch = engine.build_pool(num_lineups, 0.45, [], 3, 1.45, 40000)
        
        st.success(f"✅ AUDIT: {len(pool)*40000:,} Matrix Iterations verified in {crunch:.0f}ms.")
        
        if f2:
            try:
                content = f2.getvalue().decode('utf-8').splitlines()
                header_row = next(i for i, line in enumerate(content) if 'Entry ID' in line)
                f2.seek(0)
                contest_df = pd.read_csv(f2, skiprows=header_row)
                for i in range(min(len(pool), len(contest_df))):
                    for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']:
                        contest_df.at[i, s] = pool[i]['players'][s]['Name']
                st.download_button("💾 DOWNLOAD DK BULK EDIT", data=contest_df.to_csv(index=False), file_name="dk_edit.csv")
            except: st.error("Contest CSV format error.")

        for i, l in enumerate(pool):
            m = l['metrics']
            grade, _ = engine.calculate_vantage_grade(m['Win'], m['Own'], m['Sal'], num_games)
            badge = "🔥 TAKEDOWN CAPABLE" if m['Win'] > 5.0 and m['Own'] < 120 else ""
            with st.expander(f"{badge} [{grade}] Lineup #{i+1} | Win: {m['Win']}% | Own: {m['Own']}% | Sal: ${m['Sal']}"):
                st.table(pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']])
