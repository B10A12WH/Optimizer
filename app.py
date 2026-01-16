import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

st.set_page_config(page_title="VANTAGE-V6.0 PRO", layout="wide", page_icon="📈")

class VantageProV6:
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

    def get_auto_scrubs(self):
        csv_outs = self.raw_df[self.raw_df['Status'].isin(['O', 'OUT', 'INJ'])]['Name'].tolist()
        zero_proj = self.raw_df[self.raw_df['Market_Proj'] <= 0]['Name'].tolist()
        return list(set(csv_outs + zero_proj))

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        self.df = self.raw_df.copy()
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            my_alpha = row['Market_Proj'] * boost
            return (my_alpha * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

    def simulate_win_pct(self, lineup_players):
        sims = 200
        # --- NEW: CORRELATED SIMULATION ---
        # We simulate the team's "hotness" first
        teams = list(set([p['Team'] for p in lineup_players]))
        team_variance = {t: np.random.normal(1.0, 0.10) for t in teams}
        
        scores = []
        for _ in range(sims):
            l_score = sum([p['Final_Proj'] * team_variance.get(p['Team'], 1.0) * np.random.normal(1.0, 0.15) for p in lineup_players])
            scores.append(l_score)
            
        target_score = 305 
        wins = sum(1 for s in scores if s >= target_score)
        return (wins / sims) * 100 if wins > 0 else round((max(scores) / target_score) * 0.45, 2), np.mean(scores)

    def build_pool(self, num_lineups, exp_limit, final_scrubs, leverage_weight, min_sal, min_sharks):
        filtered_df = self.df[~self.df['Name'].isin(final_scrubs)].reset_index(drop=True)
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            
            # --- CORRELATED SIMULATION MATH ---
            # If a team gets hot, everyone on the team gets a boost
            unique_teams = sim_df['Team'].unique()
            team_shocks = {t: np.random.normal(1, 0.08) for t in unique_teams}
            sim_df['Team_Shock'] = sim_df['Team'].map(team_shocks)
            sim_df['Sim'] = sim_df['Final_Proj'] * sim_df['Team_Shock'] * np.random.normal(1, 0.12, len(sim_df))
            
            sim_df['Lev_Fact'] = 1 + ((sim_df['Own'] / 100) * leverage_weight)
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / sim_df['Lev_Fact']
            
            prob = pulp.LpProblem(f"V6_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            
            # --- BASIC CONSTRAINTS ---
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sal
            
            # --- NEW: OWNERSHIP BRACKET (Shark Requirement) ---
            # Force at least X players under 8% ownership
            sim_df['Is_Shark'] = sim_df['Own'].apply(lambda x: 1 if x < 8 else 0)
            prob += pulp.lpSum([sim_df.loc[i, 'Is_Shark'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sharks

            # --- NEW: TEAM LIMITS (Blowout Protection) ---
            for t in unique_teams:
                prob += pulp.lpSum([choices[i][s] for i in sim_df.index if sim_df.loc[i, 'Team'] == t for s in slots]) <= 3

            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            for prev in indices_store:
                prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            
            # Position Rules (Standard DK)
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or \
                               (s == 'PG' and 'PG' in p_pos) or (s == 'SG' and 'SG' in p_pos) or \
                               (s == 'SF' and 'SF' in p_pos) or (s == 'PF' and 'PF' in p_pos) or \
                               (s == 'C' and 'C' in p_pos) or \
                               (s == 'G' and ('PG' in p_pos or 'SG' in p_pos)) or \
                               (s == 'F' and ('SF' in p_pos or 'PF' in p_pos))
                    if not eligible: prob += choices[i][s] == 0

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
            if pulp.LpStatus[prob.status] == 'Optimal':
                lineup = {s: sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1}
                l_list = list(lineup.values())
                win_pct, avg_score = self.simulate_win_pct(l_list)
                total_own = sum([p['Own'] for p in l_list])
                final_pool.append({'players': lineup, 'metrics': {'Sal': sum([p['Sal'] for p in l_list]), 'Own': round(total_own, 1), 'Win': win_pct, 'Avg': round(avg_score, 1)}})
                curr_idx = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr_idx)
                for i in curr_idx:
                    name = sim_df.loc[i, 'Name']
                    player_counts[name] = player_counts.get(name, 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool

# --- UI SECTION ---
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = VantageProV6(raw_data)
    
    st.sidebar.header("🕹️ V6.0 Quant Controls")
    num_lineups = st.sidebar.slider("Number of Lineups", 1, 50, 15)
    min_sharks = st.sidebar.slider("Min 'Shark' Players (<8% Own)", 0, 3, 1)
    
    alpha_weight = st.sidebar.slider("Alpha System Weight", 0.0, 1.0, 0.75)
    leverage_weight = st.sidebar.slider("Leverage Strength", 0.0, 2.0, 1.15)
    exp_limit = st.sidebar.slider("Exposure Cap", 0.1, 1.0, 0.5)
    
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.22)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.28)
    
    if st.button("🔥 GENERATE V6.0 PORTFOLIO"):
        auto_scrubs = engine.get_auto_scrubs()
        st.info(f"🤖 Auto-Audit: {len(auto_scrubs)} players scrubbed.")
        
        with st.status("Running Correlated Simulations...", expanded=True) as status:
            engine.generate_proprietary_projections(alpha_weight, {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b})
            pool = engine.build_pool(num_lineups, exp_limit, auto_scrubs, leverage_weight, 49750, min_sharks)
            status.update(label="V6.0 Portfolio Generated!", state="complete", expanded=False)
        
        # Exposure Table
        all_players = []
        for l in pool: all_players.extend([p['Name'] for p in l['players'].values()])
        exp_df = pd.Series(all_players).value_counts(normalize=True).mul(100).round(1).reset_index()
        exp_df.columns = ['Name', 'Exposure %']
        st.subheader("📊 Quant Portfolio Audit")
        st.dataframe(exp_df, use_container_width=True, height=250)

        st.markdown("---")
        for i, l in enumerate(pool):
            m = l['metrics']
            with st.expander(f"Lineup #{i+1} | Win%: {m['Win']}% | Own: {m['Own']}% | Median: {m['Avg']}"):
                display_df = pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']]
                display_df.columns = ['Player', 'Team', 'Salary', 'Ownership %']
                st.table(display_df)
            
        # Export logic
        export_rows = [[l['players'][s]['Name'] for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']] for l in pool]
        export_df = pd.DataFrame(export_rows, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button("💾 Download V6.0 DK Upload", data=csv_buffer.getvalue(), file_name="vantage_v6.csv", mime="text/csv")
