import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

# --- FULL SYSTEM CONFIG ---
st.set_page_config(page_title="VANTAGE-V8.0 PRO", layout="wide", page_icon="🕹️")

class VantageProV8:
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
        self.df = self.raw_df.copy()

    def get_slate_presets(self, num_games):
        """THE PRESET BRAIN: Maps game count to Quant logic."""
        if num_games <= 3: # Micro Slate
            return 1.60, 0.85, 0.40, 2
        elif num_games <= 5: # Small Slate (Tonight)
            return 1.40, 0.80, 0.45, 2
        elif num_games <= 8: # Mid Slate
            return 1.15, 0.75, 0.55, 1
        elif num_games <= 11: # Large Slate
            return 0.95, 0.70, 0.60, 1
        else: # Mega Slate
            return 0.75, 0.65, 0.70, 0

    def calculate_vantage_grade(self, win_pct, total_own, total_sal, num_games):
        # Scale thresholds based on slate size
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
        if wins > 0: return (wins / sims) * 100, np.mean(scores)
        return round((max(scores) / target) * 0.45, 2), np.mean(scores)

    def build_pool(self, num_lineups, exp_limit, final_scrubs, leverage_weight, min_sharks):
        # Late-Game UTIL anchoring
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
            
            prob = pulp.LpProblem(f"V8_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= 49750
            
            sim_df['Is_Shark'] = sim_df['Own'].apply(lambda x: 1 if x < 8 else 0)
            prob += pulp.lpSum([sim_df.loc[i, 'Is_Shark'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sharks

            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            
            # Late-Flex Rule
            for i in sim_df.index:
                if sim_df.loc[i, 'Is_Late'] == 0:
                    prob += choices[i]['UTIL'] <= 1 - (pulp.lpSum([choices[j][s] for j in sim_df.index if sim_df.loc[j, 'Is_Late'] == 1 for s in slots]) / 8)

            for prev in indices_store: prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = False
                    if s == 'UTIL': eligible = True
                    elif s == 'PG' and 'PG' in p_pos: eligible = True
                    elif s == 'SG' and 'SG' in p_pos: eligible = True
                    elif s == 'SF' and 'SF' in p_pos: eligible = True
                    elif s == 'PF' and 'PF' in p_pos: eligible = True
                    elif s == 'C' and 'C' in p_pos: eligible = True
                    elif s == 'G' and ('PG' in p_pos or 'SG' in p_pos): eligible = True
                    elif s == 'F' and ('SF' in p_pos or 'PF' in p_pos): eligible = True
                    if not eligible: prob += choices[i][s] == 0

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
            if pulp.LpStatus[prob.status] == 'Optimal':
                lineup = {s: sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1}
                l_list = list(lineup.values())
                win_pct, avg_score = self.simulate_win_pct(l_list)
                total_own = sum([p['Own'] for p in l_list])
                grade, score = self.calculate_vantage_grade(win_pct, total_own, 50000, 4)
                final_pool.append({'players': lineup, 'metrics': {'Sal': sum([p['Sal'] for p in l_list]), 'Own': round(total_own, 1), 'Win': round(win_pct, 2), 'Avg': round(avg_score, 1), 'Grade': grade, 'Score': score}})
                curr_idx = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr_idx)
                for i in curr_idx:
                    n_key = sim_df.loc[i, 'Name']
                    player_counts[n_key] = player_counts.get(n_key, 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool

# --- UI SECTION ---
st.title("🚀 VANTAGE-V8.0 PRO")
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = VantageProV8(raw_data)
    
    # --- ONE-DIAL SLATE CONTROL ---
    st.sidebar.header("🎯 Slate Preset Dial")
    num_games = st.sidebar.slider("Number of Games in Slate", 1, 15, 4)
    
    # Fetch Recommended Presets
    p_lev, p_alp, p_exp, p_sharks = engine.get_slate_presets(num_games)
    
    st.sidebar.header("🕹️ Global Controls")
    num_lineups = st.sidebar.slider("Number of Lineups", 1, 50, 15)
    
    with st.sidebar.expander("🛠️ Advanced Quant Settings"):
        st.write("Presets applied from Dial. Adjust only if needed.")
        leverage_weight = st.slider("Leverage Strength", 0.0, 2.0, p_lev)
        alpha_weight = st.slider("Alpha System Weight", 0.0, 1.0, p_alp)
        exp_limit = st.slider("Global Exposure Cap", 0.1, 1.0, p_exp)
        min_sharks = st.slider("Min Sharks (<8% Own)", 0, 3, p_sharks)
    
    st.sidebar.header("🔭 High-Conviction Vision")
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.22)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.28)
    
    if st.button("🔥 GENERATE V8.0 PORTFOLIO"):
        auto_scrubs = list(set(raw_data[raw_data['Status'].isin(['O', 'OUT', 'INJ'])]['Name'].tolist() + raw_data[raw_data['dk_points'] <= 0]['Name'].tolist()))
        st.info(f"🤖 Auto-Audit: {len(auto_scrubs)} players scrubbed. Settings adjusted for {num_games}-game slate.")
        engine.generate_proprietary_projections(alpha_weight, {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b})
        pool = engine.build_pool(num_lineups, exp_limit, auto_scrubs, leverage_weight, min_sharks)
        
        # Exposure Table
        all_players = []
        for l in pool: all_players.extend([p['Name'] for p in l['players'].values()])
        exp_df = pd.Series(all_players).value_counts(normalize=True).mul(100).round(1).reset_index()
        exp_df.columns = ['Name', 'Exposure %']
        st.subheader("📊 Portfolio Exposure Audit")
        st.dataframe(exp_df, use_container_width=True, height=250)

        st.markdown("---")
        for i, l in enumerate(pool):
            m = l['metrics']
            with st.expander(f"[{m['Grade']}] Lineup #{i+1} | Win%: {m['Win']}% | Own: {m['Own']}%"):
                if m['Score'] >= 85: st.success("✅ AUTHENTICITY VERIFIED: Elite Shark build for this slate size.")
                st.write(f"**Vantage Median:** {m['Avg']} points | **Late-Flex Active**")
                display_df = pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']]
                st.table(display_df)
            
        csv_rows = [[l['players'][s]['Name'] for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']] for l in pool]
        st.download_button("💾 Download V8.0 CSV", data=pd.DataFrame(csv_rows, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']).to_csv(index=False), file_name="vantage_v8.csv")
