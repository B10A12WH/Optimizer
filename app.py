import streamlit as st
import pandas as pd
import numpy as np
import pulp

# --- VANTAGE 99: SATURDAY DIVISIONAL EDITION ---
st.set_page_config(page_title="VANTAGE 99", layout="wide")

class Vantage99:
    def __init__(self, proj_df, input_df, salary_df):
        # 1. MERGE & CLEAN
        self.df = pd.merge(proj_df, input_df[['Name', 'MaxExp']], on='Name', how='left')
        self.df = pd.merge(self.df, salary_df[['Name', 'Salary']], on='Name', how='left')
        
        self.df = self.df.rename(columns={'Position': 'Pos', 'ProjPts': 'Proj', 'ProjOwn': 'Own', 'Salary': 'Sal'})
        self.df['Proj'] = pd.to_numeric(self.df['Proj'], errors='coerce').fillna(0)
        self.df['Sal'] = pd.to_numeric(self.df['Sal'], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df['Own'], errors='coerce').fillna(10)

        # 2. ROSTER LEGITIMACY (Jan 17, 2026)
        # Filters Metcalf (Steelers), Davis (Jags/Out), Kittle/Warner (Out)
        blacklist = ['DK Metcalf', 'Gabe Davis', 'George Kittle', 'Fred Warner', 'Joshua Palmer']
        self.df = self.df[~self.df['Name'].isin(blacklist)]
        self.df = self.df[self.df['Pos'] != 'K'] # No Kickers

    def cook(self, n_lineups=20):
        final_pool = []
        player_indices = self.df.index
        # 135% OWNERSHIP CAP + $48,200 FLOOR (The Sharp 2-Game Logic)
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 135.0, 48200, 0.35, 3

        for n in range(n_lineups):
            sim_df = self.df.copy()
            # Position-based Jitter
            for i in player_indices:
                row = sim_df.loc[i]
                std = JITTER * (1.3 if row['Pos'] in ['WR', 'TE'] else 0.8)
                sim_df.at[i, 'Sim_Proj'] = np.random.normal(row['Proj'], row['Proj'] * std)

            # --- SCENARIO CATALYST (The Multipliers) ---
            sim_df.loc[sim_df['Name'] == 'Jaxon Smith-Njigba', 'Sim_Proj'] *= 1.35
            sim_df.loc[sim_df['Name'] == 'Jauan Jennings', 'Sim_Proj'] *= 1.40
            sim_df.loc[sim_df['Name'] == 'Kenneth Walker III', 'Sim_Proj'] *= 1.30
            sim_df.loc[sim_df['Name'] == 'Khalil Shakir', 'Sim_Proj'] *= 1.25
            sim_df.loc[sim_df['Name'] == 'RJ Harvey', 'Sim_Proj'] *= 1.15
            sim_df.loc[sim_df['Name'] == 'Josh Allen', 'Sim_Proj'] *= 0.88 

            # THE SOLVER
            prob = pulp.LpProblem(f"Batch_{n}", pulp.LpMaximize)
            choices = pulp.LpVariable.dicts("P", player_indices, cat='Binary')

            prob += pulp.lpSum([sim_df.loc[i, 'Sim_Proj'] * choices[i] for i in player_indices])
            prob += pulp.lpSum([choices[i] for i in player_indices]) == 9
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) >= SAL_FLOOR
            prob += pulp.lpSum([sim_df.loc[i, 'Own'] * choices[i] for i in player_indices]) <= OWN_CAP

            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                mask = [choices[i] for i in player_indices if sim_df.loc[i, 'Pos'] == p]
                prob += pulp.lpSum(mask) >= mn
                prob += pulp.lpSum(mask) <= mx

            for prev in final_pool:
                prob += pulp.lpSum([choices[i] for i in prev.index]) <= (9 - MIN_UNIQUE)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                final_pool.append(sim_df.loc[[i for i in player_indices if choices[i].varValue == 1]])
        return final_pool

# --- UI ---
st.title("🧪 VANTAGE 99")
st.warning("Lockout in 40 minutes. Upload files to start the batch.")

col1, col2, col3 = st.columns(3)
with col1: p_file = st.file_uploader("Projections", type="csv")
with col2: i_file = st.file_uploader("Inputs", type="csv")
with col3: s_file = st.file_uploader("DK Salaries (Required)", type="csv")

if p_file and i_file and s_file:
    engine = Vantage99(pd.read_csv(p_file), pd.read_csv(i_file), pd.read_csv(s_file))
    if st.button("🔥 START THE COOK"):
        lineups = engine.cook(n_lineups=20)
        st.header("📋 Manual Entry Mode")
        for i, l in enumerate(lineups):
            st.subheader(f"Lineup {i+1}")
            # Format for quick reading
            names = l[['Pos', 'Name']].sort_values('Pos').values
            st.write(" | ".join([f"**{n[0]}**: {n[1]}" for n in names]))
            st.divider()
