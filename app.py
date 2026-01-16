import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

# Set up Page Configuration
st.set_page_config(page_title="VANTAGE-V4 ALPHA", layout="wide", page_icon="🏆")

# --- ENGINE CLASS: ALPHA V4 ---
class AlphaVantageV4:
    def __init__(self, df):
        # Initial cleanup and mapping
        self.raw_df = df.copy()
        
        # Map logical tiers for the UTIL late-swap insurance
        # (Prioritizing late games for the UTIL slot)
        time_map = {'CLE': 1, 'PHI': 1, 'NOP': 1, 'IND': 1, 'CHI': 2, 'BKN': 2, 'LAC': 2, 'TOR': 2}
        self.raw_df['Time_Tier'] = self.raw_df['Team'].map(time_map).fillna(1)

        # Standardizing Column Names from SaberSim/Stokastic
        mapping = {
            'Name': 'Name', 
            'Salary': 'Sal', 
            'dk_points': 'Base', 
            'My Own': 'Own', 
            'Pos': 'Pos', 
            'Team': 'Team', 
            'Opp': 'Opp'
        }
        self.raw_df = self.raw_df.rename(columns=mapping)
        
        # Convert numeric columns safely
        self.raw_df['Base'] = pd.to_numeric(self.raw_df['Base'], errors='coerce').fillna(0)
        self.raw_df['Sal'] = pd.to_numeric(self.raw_df['Sal'], errors='coerce').fillna(50000)
        self.raw_df['Own'] = pd.to_numeric(self.raw_df['Own'], errors='coerce').fillna(5)

    def scrub_injuries(self, out_list):
        self.raw_df = self.raw_df[~self.raw_df['Name'].isin(out_list)]

    def apply_alpha(self, usage_boosts, pace_map, dvp_map):
        self.df = self.raw_df.copy()
        
        def calc_custom_proj(row):
            score = row['Base']
            # Apply Pace Boost
            score *= pace_map.get(row['Team'], 1.0)
            # Apply Usage/Alpha Boost
            score *= usage_boosts.get(row['Name'], 1.0)
            # Apply DvP Boost (Defense vs Position)
            p_pos = str(row['Pos']).split('/')[0]
            dvp_mult = dvp_map.get(row['Opp'], {}).get(p_pos, 1.0)
            return score * dvp_mult

        self.df['Alpha_Proj'] = self.df.apply(calc_custom_proj, axis=1)
        # Blend Alpha Proj (80%) with Market Base (20%)
        self.df['Final_Proj'] = (self.df['Alpha_Proj'] * 0.8) + (self.df['Base'] * 0.2)
        
        # Filter out low-value players to speed up solver
        self.df = self.df[self.df['Final_Proj'] > 8].reset_index(drop=True)

    def simulate_win_pct(self, lineup_players, simulations=400):
        """Calculates % of time lineup exceeds a GPP-winning score."""
        scores = []
        for _ in range(simulations):
            # 0.22 std dev is standard for NBA player variance
            sim_score = sum([p['Final_Proj'] * np.random.normal(1.0, 0.22) for p in lineup_players])
            scores.append(sim_score)
        
        takedown_threshold = 285 # Typical GPP winning score for a mid-sized slate
        wins = sum(1 for s in scores if s >= takedown_threshold)
        return (wins / simulations) * 100

    def build_pool(self, num_lineups, exp_limit):
        indices_store, final_pool, player_counts = [], [], {}
        
        for n in range(num_lineups):
            sim_df = self.df.copy()
            # Randomize projections per run for variety
            sim_df['Sim_Score'] = sim_df['Final_Proj'] * np.random.normal(1, 0.15, len(sim_df))
            # Shark Score formula
            sim_df['Leverage'] = (sim_df['Sim_Score']**3) / (sim_df['Own'] + 1)

            prob = pulp.LpProblem(f"Vantage_Run_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')

            # Objective: Maximize Leverage + UTIL late-swap weight
            prob += pulp.lpSum([sim_df.loc[i, 'Leverage'] * choices[i][s] for i in sim_df.index for s in slots]) + \
                    pulp.lpSum([sim_df.loc[i, 'Time_Tier'] * 1000000 * choices[i]['UTIL'] for i in sim_df.index])

            # Constraints
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            for s in slots:
                prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index:
                prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            
            # Diversity Constraint: Max 5 players overlapping with any previous lineup
            for prev in indices_store:
                prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5

            # Global Exposure Cap
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0

            # Position Eligibility
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

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                lineup_dict = {s: sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1}
                lineup_list = list(lineup_dict.values())
                
                # Metrics Calculation
                sal = sum([p['Sal'] for p in lineup_list])
                own = sum([p['Own'] for p in lineup_list])
                win = self.simulate_win_pct(lineup_list)
                
                final_pool.append({
                    'players': lineup_dict,
                    'metrics': {'Sal': sal, 'Own': round(own, 1), 'Win': round(win, 2)}
                })
                
                curr_indices = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr_indices)
                for i in curr_indices:
                    name = sim_df.loc[i, 'Name']
                    player_counts[name] = player_counts.get(name, 0) + 1
                    
        return final_pool

# --- STREAMLIT UI ---
st.title("🏆 VANTAGE-V4: GPP TAKEDOWN ENGINE")
st.markdown("---")

uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    engine = AlphaVantageV4(raw_df)
    
    # Sidebar
    st.sidebar.header("🕹️ Architect Controls")
    out_list = st.sidebar.multiselect("Scrub OUT Players", raw_df['Name'].unique())
    mitchell_b = st.sidebar.slider("Donovan Mitchell Boost", 1.0, 1.5, 1.18)
    barnes_b = st.sidebar.slider("Scottie Barnes Boost", 1.0, 1.5, 1.25)
    exp_limit = st.sidebar.slider("Exposure Cap", 0.1, 1.0, 0.6)
    
    if st.button("🚀 GENERATE & SIMULATE WIN %"):
        with st.spinner("Crunching Math & Simulating Outcomes..."):
            # Setup Alpha
            engine.scrub_injuries(out_list)
            pace = {'IND': 1.08, 'NOP': 1.05, 'PHI': 1.03, 'CLE': 1.02, 'CHI': 1.04, 'BKN': 1.01, 'TOR': 1.06, 'LAC': 0.98}
            usage = {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b}
            dvp = {'PHI': {'PG': 1.12}, 'TOR': {'C': 1.20}, 'IND': {'SF': 1.15}}
            engine.apply_alpha(usage, pace, dvp)
            
            # Build Pool
            pool = engine.build_pool(15, exp_limit)
            
            # Display Results
            for i, l in enumerate(pool):
                m = l['metrics']
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Salary", f"${m['Sal']}")
                col2.metric("Total Ownership", f"{m['Own']}%")
                col3.metric("Proprietary Win %", f"{m['Win']}%")
                
                with st.expander(f"💎 ALPHA-V4 LINEUP #{i+1}"):
                    pdf = pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']]
                    st.table(pdf)
            
            # CSV EXPORT
            st.markdown("---")
            st.subheader("📤 Export to DraftKings")
            export_rows = [[l['players'][s]['Name'] for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']] for l in pool]
            export_df = pd.DataFrame(export_rows, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            st.download_button("💾 Download DK Upload File", data=csv_buffer.getvalue(), file_name="vantage_lock.csv", mime="text/csv")
