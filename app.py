import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

st.set_page_config(page_title="VANTAGE-V5.2 PRO", layout="wide", page_icon="🚀")

class VantageProV5:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        
        # --- ROBUST MAPPING ---
        mapping = {
            'Name': 'Name', 
            'Salary': 'Sal', 
            'dk_points': 'Market_Proj', 
            'Adj Own': 'Own', 
            'Pos': 'Pos', 
            'Team': 'Team'
        }
        self.raw_df = self.raw_df.rename(columns=mapping)
        
        # Guard against missing data
        required = ['Sal', 'Market_Proj', 'Own']
        for col in required:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)
            else:
                st.error(f"Missing required column: {col}. Check your CSV headers.")

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        self.df = self.raw_df.copy()
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            my_alpha = row['Market_Proj'] * boost
            return (my_alpha * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

    def simulate_win_pct(self, lineup_players):
        sims = 250 # Increased precision
        scores = [sum([p['Final_Proj'] * np.random.normal(1.0, 0.22) for p in lineup_players]) for _ in range(sims)]
        target_score = 305 
        wins = sum(1 for s in scores if s >= target_score)
        if wins == 0: return round((max(scores) / target_score) * 0.45, 2)
        return (wins / sims) * 100

    def build_pool(self, num_lineups, exp_limit, scrub_list, leverage_weight, min_sal):
        filtered_df = self.df[~self.df['Name'].isin(scrub_list)].reset_index(drop=True)
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            # Simulate Slate Variance
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.18, len(sim_df))
            
            # Calibrated Leverage Factor
            sim_df['Leverage_Factor'] = 1 + ((sim_df['Own'] / 100) * leverage_weight)
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / sim_df['Leverage_Factor']
            
            prob = pulp.LpProblem(f"V5_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            
            # SALARY CONSTRAINTS
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sal
            
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            
            # DIVERSITY CONSTRAINT (Pro Tip: 4-game slates need tighter diversity)
            for prev in indices_store:
                prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5

            # EXPOSURE LIMITS
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            
            # POS ELIGIBILITY
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or (s == 'PG' and 'PG' in p_pos) or (s == 'SG' and 'SG' in p_pos) or (s == 'SF' and 'SF' in p_pos) or (s == 'PF' and 'PF' in p_pos) or (s == 'C' and 'C' in p_pos) or (s == 'G' and ('PG' in p_pos or 'SG' in p_pos)) or (s == 'F' and ('SF' in p_pos or 'PF' in p_pos))
                    if not eligible: prob += choices[i][s] == 0

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
            if pulp.LpStatus[prob.status] == 'Optimal':
                lineup = {s: sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1}
                l_list = list(lineup.values())
                final_pool.append({
                    'players': lineup,
                    'metrics': {'Sal': sum([p['Sal'] for p in l_list]), 'Own': sum([p['Own'] for p in l_list]), 'Win': self.simulate_win_pct(l_list)}
                })
                curr_idx = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr_idx)
                for i in curr_idx: player_counts[sim_df.loc[i, 'Name']] = player_counts.get(sim_df.loc[i, 'Name'], 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
        return final_pool

# --- UI ---
st.title("🚀 VANTAGE-V5.2 PRO: THE PORTFOLIO MANAGER")
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = VantageProV5(raw_data)
    
    st.sidebar.header("🕹️ Portfolio Controls")
    num_lineups = st.sidebar.slider("Number of Lineups", 1, 50, 15)
    min_sal = st.sidebar.slider("Min Salary Target", 48000, 50000, 49700)
    
    alpha_weight = st.sidebar.slider("Alpha System Weight", 0.0, 1.0, 0.75)
    leverage_weight = st.sidebar.slider("Leverage Strength", 0.0, 2.0, 0.95)
    exp_limit = st.sidebar.slider("Global Exposure Cap", 0.1, 1.0, 0.5)
    
    scrub_list = st.sidebar.multiselect("🚫 Scrub OUT", sorted(raw_data['Name'].unique().tolist()))
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.22)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.28)
    
    if st.button("🔥 GENERATE & AUDIT PORTFOLIO"):
        with st.status("Simulating Slate Scenarios...", expanded=True) as status:
            engine.generate_proprietary_projections(alpha_weight, {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b})
            pool = engine.build_pool(num_lineups, exp_limit, scrub_list, leverage_weight, min_sal)
            status.update(label="Portfolio Ready for Review!", state="complete", expanded=False)
        
        # --- PORTFOLIO DASHBOARD ---
        st.subheader("📊 Portfolio Exposure Dashboard")
        all_players = []
        for l in pool: all_players.extend([p['Name'] for p in l['players'].values()])
        exp_df = pd.Series(all_players).value_counts(normalize=True).mul(100).round(1).reset_index()
        exp_df.columns = ['Name', 'Exposure %']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Avg Portfolio Own", f"{round(np.mean([l['metrics']['Own'] for l in pool]), 1)}%")
            st.metric("Unique Players", len(exp_df))
        with col2:
            st.dataframe(exp_df.style.background_gradient(cmap='Greens'), use_container_width=True)

        # --- LINEUP VIEWER ---
        st.markdown("---")
        for i, l in enumerate(pool):
            m = l['metrics']
            with st.expander(f"💎 Lineup #{i+1} | Win%: {m['Win']}% | Sal: ${m['Sal']} | Own: {m['Own']}%"):
                st.table(pd.DataFrame(l['players']).T[['Name', 'Sal', 'Own', 'Team']])
            
        # CSV Export
        export_rows = [[l['players'][s]['Name'] for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']] for l in pool]
        export_df = pd.DataFrame(export_rows, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button("💾 Download DK Portfolio File", data=csv_buffer.getvalue(), file_name="vantage_lock.csv", mime="text/csv")
