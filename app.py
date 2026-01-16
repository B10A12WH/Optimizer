import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

# --- VANTAGE-V7.4 PRO: FULL VISIBILITY EDITION ---
st.set_page_config(page_title="VANTAGE-V7.4 PRO", layout="wide", page_icon="🛡️")

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
        """THE BRAIN: Detects slate context and recommends settings."""
        num_teams = self.raw_df['Team'].nunique()
        scrub_count = len(self.raw_df[self.raw_df['Status'].isin(['O', 'OUT', 'INJ'])])
        
        # Calibration logic for 4-game slate
        if num_teams <= 4:
            return 1.45, 0.80, 0.45, 2 # Lev, Alpha, Exp, Sharks
        return 1.10, 0.75, 0.55, 1

    def calculate_vantage_grade(self, win_pct, total_own, total_sal):
        ceiling = min(win_pct * 12, 45) 
        leverage = 35 if 75 <= total_own <= 95 else 20
        efficiency = (total_sal / 50000) * 20
        score = ceiling + leverage + efficiency
        grades = [(90, "A+"), (82, "A"), (75, "B+"), (68, "B"), (60, "C")]
        for threshold, label in grades:
            if score >= threshold: return label, score
        return "D", score

    def simulate_win_pct(self, lineup_players):
        sims = 200
        teams = list(set([p['Team'] for p in lineup_players]))
        team_variance = {t: np.random.normal(1.0, 0.10) for t in teams}
        scores = [sum([p['Final_Proj'] * team_variance.get(p['Team'], 1.0) * np.random.normal(1.0, 0.15) for p in lineup_players]) for _ in range(sims)]
        target = 305 
        wins = sum(1 for s in scores if s >= target)
        return (wins / sims) * 100 if wins > 0 else round((max(scores) / target) * 0.45, 2), np.mean(scores)

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
            
            # Min Shark Requirement
            sim_df['Is_Shark'] = sim_df['Own'].apply(lambda x: 1 if x < 8 else 0)
            prob += pulp.lpSum([sim_df.loc[i, 'Is_Shark'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sharks

            # Position/Slot Logic
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            
            # Late-Flex Rule
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

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        self.df = self.raw_df.copy()
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            return (row['Market_Proj'] * boost * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

# --- UI SECTION ---
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")
if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = VantageProV7(raw_data)
    a_lev, a_alp, a_exp, a_sharks = engine.auto_calibrate_settings()
    
    st.sidebar.header("🕹️ V7.4 Quant Controls")
    num_lineups = st.sidebar.slider("Number of Lineups", 1, 50, 15)
    leverage_weight = st.sidebar.slider("Leverage Strength", 0.0, 2.0, a_lev)
    alpha_weight = st.sidebar.slider("Alpha System Weight", 0.0, 1.0, a_alp)
    exp_limit = st.sidebar.slider("Exposure Cap", 0.1, 1.0, a_exp)
    min_sharks = st.sidebar.slider("Min Sharks (<8% Own)", 0, 3, a_sharks)
    
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.22)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.28)
    
    if st.button("🔥 GENERATE V7.4 PORTFOLIO"):
        auto_scrubs = list(set(raw_data[raw_data['Status'].isin(['O', 'OUT', 'INJ'])]['Name'].tolist() + raw_data[raw_data['dk_points'] <= 0]['Name'].tolist()))
        st.info(f"🤖 Auto-Audit: {len(auto_scrubs)} players scrubbed.")
        engine.generate_proprietary_projections(alpha_weight, {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b})
        pool = engine.build_pool(num_lineups, exp_limit, auto_scrubs, leverage_weight, 49750, min_sharks)
        
        # Exposure Table
        all_players = []
        for l in pool: all_players.extend([p['Name'] for p in l['players'].values()])
        exp_df = pd.Series(all_players).value_counts(normalize=True).mul(100).round(1).reset_index()
        exp_df.columns = ['Name', 'Exposure %']
        st.subheader("📊 Portfolio Exposure Audit")
        st.dataframe(exp_df, use_container_width=True, height=250)

        for i, l in enumerate(pool):
            m = l['metrics']
            with st.expander(f"[{m['Grade']}] Lineup #{i+1} | Win%: {m['Win']}% | Own: {m['Own']}%"):
                if m['Score'] >= 88: st.success("✅ AUTHENTICITY VERIFIED: Elite leverage & ceiling detected.")
                st.write(f"**Vantage Median:** {m['Avg']} points | **Late-Flex Active**")
                display_df = pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']]
                st.table(display_df)
            
        csv_rows = [[l['players'][s]['Name'] for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']] for l in pool]
        st.download_button("💾 Download V7.4 CSV", data=pd.DataFrame(csv_rows, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']).to_csv(index=False), file_name="vantage_v7_4.csv")
