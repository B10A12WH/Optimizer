import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

st.set_page_config(page_title="VANTAGE-V5 PRO", layout="wide", page_icon="🚀")

class VantageProV5:
    def __init__(self, df):
        self.raw_df = df.copy()
        # Clean up column headers for exact matching
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        
        # --- STRATEGIC MAPPING: Connecting to Adj Ownership ---
        mapping = {
            'Name': 'Name', 
            'Salary': 'Sal', 
            'dk_points': 'Market_Proj', 
            'Adj Own': 'Own', 
            'Pos': 'Pos', 
            'Team': 'Team'
        }
        self.raw_df = self.raw_df.rename(columns=mapping)
        self.raw_df['Market_Proj'] = pd.to_numeric(self.raw_df['Market_Proj'], errors='coerce').fillna(0)
        self.raw_df['Sal'] = pd.to_numeric(self.raw_df['Sal'], errors='coerce').fillna(50000)
        self.raw_df['Own'] = pd.to_numeric(self.raw_df['Own'], errors='coerce').fillna(5)

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        """Creates the custom 'Vantage Alpha' projection blend."""
        self.df = self.raw_df.copy()
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            my_alpha = row['Market_Proj'] * boost
            # Blend your Alpha math with the market (e.g. 70/30)
            return (my_alpha * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

    def simulate_win_pct(self, lineup_players):
        """Monte Carlo Simulation to determine GPP takedown probability."""
        simulations = 100
        lineup_scores = [sum([p['Final_Proj'] * np.random.normal(1.0, 0.22) for p in lineup_players]) for _ in range(simulations)]
        target_score = 265 # Competitive threshold for a 4-game slate
        wins = sum(1 for s in lineup_scores if s >= target_score)
        if wins == 0:
            return round(max(lineup_scores) / 3, 2)
        return (wins / simulations) * 100

    def build_pool(self, num_lineups, exp_limit, scrub_list):
        """Solves for optimal lineups using simulated variance and Shark Scoring."""
        # 1. Apply the Scrub Filter
        filtered_df = self.df[~self.df['Name'].isin(scrub_list)].reset_index(drop=True)
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            # 2. Simulate game-slate variance
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.18, len(sim_df))
            # 3. Apply the Shark Leverage Formula (Proj^3 / Ownership)
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / (sim_df['Own'] + 1)
            
            prob = pulp.LpProblem(f"V5_Run_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            
            # Position, Diversity, and Exposure Constraints
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            for prev in indices_store: prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            
            # DraftKings Position Eligibility
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or \
                               (s == 'PG' and 'PG' in p_pos) or \
                               (s == 'SG' and 'SG' in p_pos) or \
                               (s == 'SF' and 'SF' in p_pos) or \
                               (s == 'PF' and 'PF' in p_pos) or \
                               (s == 'C' and 'C' in p_pos) or \
                               (s == 'G' and ('PG' in p_pos or 'SG' in p_pos)) or \
                               (s == 'F' and ('SF' in p_pos or 'PF' in p_pos))
                    if not eligible: prob += choices[i][s] == 0

            # Solve with a 5-second time limit per lineup for cloud stability
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
                for i in curr_idx:
                    name = sim_df.loc[i, 'Name']
                    player_counts[name] = player_counts.get(name, 0) + 1
            
            progress_bar.progress((n + 1) / num_lineups)
            
        return final_pool

# --- USER INTERFACE ---
st.title("🚀 VANTAGE-V5: PROPRIETARY SIMULATOR")
st.markdown("---")

uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = VantageProV5(raw_data)
    
    st.sidebar.header("🕹️ Proprietary Controls")
    alpha_weight = st.sidebar.slider("Alpha System Weight (vs Market)", 0.0, 1.0, 0.7)
    scrub_list = st.sidebar.multiselect("🚫 Scrub OUT Players", sorted(raw_data['Name'].unique().tolist()))
    
    # Contextual Boosts (Mitchell & Barnes)
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.18)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.25)
    exp_limit = st.sidebar.slider("Global Exposure Cap", 0.1, 1.0, 0.6)
    
    if st.button("🔥 RUN PROPRIETARY SIMULATION"):
        with st.status("Solving and Simulating...", expanded=True) as status:
            engine.generate_proprietary_projections(alpha_weight, {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b})
            pool = engine.build_pool(15, exp_limit, scrub_list)
            status.update(label="Simulation Complete!", state="complete", expanded=False)
        
        # Display Lineups and Metrics
        for i, l in enumerate(pool):
            m = l['metrics']
            st.write(f"### Lineup #{i+1} | Win%: {m['Win']}% | Sal: ${m['Sal']} | Own: {m['Own']}%")
            st.table(pd.DataFrame(l['players']).T[['Name', 'Sal', 'Own']])
            
        # CSV EXPORT
        st.markdown("---")
        st.subheader("📤 Export to DraftKings")
        export_rows = [[l['players'][s]['Name'] for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']] for l in pool]
        export_df = pd.DataFrame(export_rows, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button("💾 Download DK Upload File", data=csv_buffer.getvalue(), file_name="vantage_lock.csv", mime="text/csv")
