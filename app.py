import streamlit as st
import pandas as pd
import numpy as np
import pulp

st.set_page_config(page_title="VANTAGE 99", layout="wide")

class Vantage99:
    def __init__(self, proj_df, input_df, sal_df):
        # 1. FORCE COLUMN NAMES (To bypass the Index Error)
        # Projections & Inputs usually follow the standard names
        proj_df.columns = [str(c).strip() for c in proj_df.columns]
        input_df.columns = [str(c).strip() for c in input_df.columns]
        sal_df.columns = [str(c).strip() for c in sal_df.columns]

        # 2. MERGE
        self.df = pd.merge(proj_df, input_df[['Name', 'MaxExp']], on='Name', how='left')
        
        # Check for Salary Column specifically
        # If the DK file uses 'Salary', 'salary', or 'Player Name'
        s_cols = sal_df.columns.tolist()
        name_key = next((c for c in s_cols if 'name' in c.lower()), 'Name')
        sal_key = next((c for c in s_cols if 'salary' in c.lower()), 'Salary')
        
        self.df = pd.merge(self.df, sal_df[[name_key, sal_key]], left_on='Name', right_on=name_key, how='left')
        
        # 3. PURIFY
        self.df = self.df.rename(columns={'Position': 'Pos', 'ProjPts': 'Proj', 'ProjOwn': 'Own', sal_key: 'Sal'})
        self.df['Proj'] = pd.to_numeric(self.df['Proj'], errors='coerce').fillna(0)
        self.df['Sal'] = pd.to_numeric(self.df['Sal'], errors='coerce').fillna(5000)
        self.df['Own'] = pd.to_numeric(self.df['Own'], errors='coerce').fillna(10)

        # 4. BLACKLIST
        blacklist = ['DK Metcalf', 'Gabe Davis', 'George Kittle', 'Fred Warner']
        self.df = self.df[~self.df['Name'].isin(blacklist)]

    def cook(self, n_lineups=20):
        pool = []
        player_indices = self.df.index
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 135.0, 48200, 0.35, 3

        for n in range(n_lineups):
            sim_df = self.df.copy()
            sim_df['sim_proj'] = np.random.normal(sim_df['Proj'], sim_df['Proj'] * JITTER)
            
            # Scenario Multipliers
            sim_df.loc[sim_df['Name'] == 'Jaxon Smith-Njigba', 'sim_proj'] *= 1.35
            sim_df.loc[sim_df['Name'] == 'Jauan Jennings', 'sim_proj'] *= 1.40
            sim_df.loc[sim_df['Name'] == 'Kenneth Walker III', 'sim_proj'] *= 1.30
            sim_df.loc[sim_df['Name'] == 'Josh Allen', 'sim_proj'] *= 0.88 

            prob = pulp.LpProblem(f"Batch_{n}", pulp.LpMaximize)
            choices = pulp.LpVariable.dicts("P", player_indices, cat='Binary')

            prob += pulp.lpSum([sim_df.loc[i, 'sim_proj'] * choices[i] for i in player_indices])
            prob += pulp.lpSum([choices[i] for i in player_indices]) == 9
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) >= SAL_FLOOR
            prob += pulp.lpSum([sim_df.loc[i, 'Own'] * choices[i] for i in player_indices]) <= OWN_CAP

            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                mask = [choices[i] for i in player_indices if sim_df.loc[i, 'Pos'] == p]
                prob += pulp.lpSum(mask) >= mn
                prob += pulp.lpSum(mask) <= mx

            for prev in pool:
                prob += pulp.lpSum([choices[i] for i in prev.index]) <= (9 - MIN_UNIQUE)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                pool.append(sim_df.loc[[i for i in player_indices if choices[i].varValue == 1]])
        return pool

# --- MAIN UI ---
st.title("🧪 VANTAGE 99: NO-CRASH EDITION")
p_f = st.file_uploader("1", type="csv")
i_f = st.file_uploader("2", type="csv")
s_f = st.file_uploader("3", type="csv")

if p_f and i_f and s_f:
    p_df = pd.read_csv(p_f, on_bad_lines='skip', engine='python')
    i_df = pd.read_csv(i_f, on_bad_lines='skip', engine='python')
    s_df = pd.read_csv(s_f, on_bad_lines='skip', engine='python')
    
    try:
        engine = Vantage99(p_df, i_df, s_df)
        if st.button("🔥 START"):
            results = engine.cook()
            for i, l in enumerate(results):
                st.write(f"Lineup {i+1}: " + " | ".join(l['Name'].tolist()))
    except Exception as e:
        st.error(f"Error: {e}. Check if file 3 is the correct DK Salary CSV.")
