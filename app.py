import streamlit as st
import pandas as pd
import numpy as np
import pulp

# --- Vantage 99: Institutional Config ---
st.set_page_config(page_title="VANTAGE 99", layout="wide", page_icon="🧪")

class Vantage99:
    def __init__(self, proj_df, input_df, salary_df):
        # 1. THE REACTION: Merging the precursors
        self.df = pd.merge(proj_df, input_df[['Name', 'MaxExp']], on='Name', how='left')
        self.df = pd.merge(self.df, salary_df[['Name', 'Salary']], on='Name', how='left')
        
        # 2. PURIFICATION: Standardizing headers and types
        self.df = self.df.rename(columns={'Position': 'Pos', 'ProjPts': 'Proj', 'ProjOwn': 'Own', 'Salary': 'Sal'})
        self.df['Proj'] = pd.to_numeric(self.df['Proj'], errors='coerce').fillna(0)
        self.df['Sal'] = pd.to_numeric(self.df['Sal'], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df['Own'], errors='coerce').fillna(10)
        self.df['MaxExp'] = self.df['MaxExp'].fillna(100) / 100.0
        
        # 3. CONTAMINATION REMOVAL: Roster Legitimacy Filters
        # Confirmed OUT or Traded (Metcalf is a Steeler, Davis is Jags/Injured, Kittle/Warner OUT)
        self.blacklist = ['DK Metcalf', 'Gabe Davis', 'George Kittle', 'Fred Warner', 'Joshua Palmer']
        self.df = self.df[~self.df['Name'].isin(self.blacklist)]
        self.df = self.df[self.df['Pos'] != 'K'] # Strict Kickerless Flex focus

    def apply_scenario_catalyst(self, sim_df):
        """Molecular modifiers for the Saturday Jan 17, 2026 Game Scripts."""
        
        # --- SEATTLE / NINERS MOLECULES ---
        # Target Vacuum: No Kittle/Metcalf boosts JSN and Jennings
        sim_df.loc[sim_df['Name'] == 'Jaxon Smith-Njigba', 'Sim_Proj'] *= 1.35
        sim_df.loc[sim_df['Name'] == 'Jauan Jennings', 'Sim_Proj'] *= 1.40
        sim_df.loc[sim_df['Name'] == 'Kenneth Walker III', 'Sim_Proj'] *= 1.30 # Warner Void
        
        # Darnold Oblique: Efficiency cap
        sim_df.loc[sim_df['Name'] == 'Sam Darnold', 'Sim_Proj'] *= 0.90 

        # --- BILLS / BRONCOS MOLECULES ---
        # Gabe Davis OUT: Shakir Consolidation
        sim_df.loc[sim_df['Name'] == 'Khalil Shakir', 'Sim_Proj'] *= 1.25
        sim_df.loc[sim_df['Name'] == 'RJ Harvey', 'Sim_Proj'] *= 1.15 # Denver Sharp volume
        
        # Leverage Fade: Josh Allen ownership vs. Denver Defense ceiling
        sim_df.loc[sim_df['Name'] == 'Josh Allen', 'Sim_Proj'] *= 0.88 
        sim_df.loc[sim_df['Name'] == 'Denver Broncos', 'Sim_Proj'] *= 1.15 
        
        return sim_df

    def cook(self, n_lineups=20, num_games=2):
        final_pool = []
        player_indices = self.df.index
        
        # --- THE "BLUE" SCALE (Slate-Specific Hard Caps) ---
        # 135% Ownership Cap is the secret ingredient for the 2-game Saturday slate
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 135.0, 48200, 0.35, 3

        progress = st.progress(0)
        for n in range(n_lineups):
            sim_df = self.df.copy()
            
            # STOCHASTIC JITTER (Range of Outcomes)
            for i in player_indices:
                row = sim_df.loc[i]
                # High volatility for WR/TE fat-tails
                std = JITTER * (1.3 if row['Pos'] in ['WR', 'TE'] else 0.8)
                sim_df.at[i, 'Sim_Proj'] = np.random.normal(row['Proj'], row['Proj'] * std)

            # APPLY THE SCENARIO CATALYST
            sim_df = self.apply_scenario_catalyst(sim_df)

            # THE SOLVER (PuLP)
            prob = pulp.LpProblem(f"Batch_{n}", pulp.LpMaximize)
            choices = pulp.LpVariable.dicts("P", player_indices, cat='Binary')

            prob += pulp.lpSum([sim_df.loc[i, 'Sim_Proj'] * choices[i] for i in player_indices])
            prob += pulp.lpSum([choices[i] for i in player_indices]) == 9
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) >= SAL_FLOOR
            prob += pulp.lpSum([sim_df.loc[i, 'Own'] * choices[i] for i in player_indices]) <= OWN_CAP

            # Positional Constraints
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                mask = [choices[i] for i in player_indices if sim_df.loc[i, 'Pos'] == p]
                prob += pulp.lpSum(mask) >= mn
                prob += pulp.lpSum(mask) <= mx

            # Uniqueness Control: Forcing the field spread
            for prev in final_pool:
                prob += pulp.lpSum([choices[i] for i in prev.index]) <= (9 - MIN_UNIQUE)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                final_pool.append(sim_df.loc[[i for i in player_indices if choices[i].varValue == 1]])
            
            progress.progress((n+1)/n_lineups)
            
        return final_pool

# --- Streamlit UI ---
st.title("🧪 VANTAGE 99")
st.caption("99.1% Pure DFS Optimization | Saturday Divisional Special")

col1, col2, col3 = st.columns(3)
with col1: p_file = st.file_uploader("Projections", type="csv")
with col2: i_file = st.file_uploader("Inputs", type="csv")
with col3: s_file = st.file_uploader("DK Salaries", type="csv")

if p_file and i_file and s_file:
    engine = Vantage99(pd.read_csv(p_file), pd.read_csv(i_file), pd.read_csv(s_file))
    
    num_games = st.sidebar.slider("Games on Slate", 2, 16, 2)
    n_lineups = st.sidebar.slider("Batch Size", 1, 150, 20)
    
    if st.button("🔥 START THE COOK"):
        lineups = engine.cook(n_lineups, num_games)
        
        # EXPOSURE AUDIT
        st.header("📊 Batch Audit")
        full_roster = pd.concat(lineups)
        exp = (full_roster['Name'].value_counts() / len(lineups) * 100).round(1)
        st.dataframe(exp.rename("Exposure %").head(15))

        # LINEUP DELIVERY
        for i, l in enumerate(lineups):
            with st.expander(f"Lineup {i+1} | Proj: {l['Proj'].sum():.1f} | Sal: ${l['Sal'].sum()}"):
                st.table(l[['Pos', 'Name', 'Team', 'Sal', 'Proj']])
