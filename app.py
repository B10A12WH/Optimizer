import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

# --- VANTAGE 99: THE PERFECT COOK (STABILIZED) ---
st.set_page_config(page_title="VANTAGE 99", layout="wide", page_icon="🧪")

class Vantage99:
    def __init__(self, proj_df, input_df, sal_df):
        # 1. MOLECULAR CLEANING
        # DraftKings CSVs often have 'Salary ' (with a space) or 'Salary'
        sal_df.columns = [c.strip() for c in sal_df.columns]
        
        # 2. THE REACTION: Merging Data
        # We merge Projections with Inputs and then join Salaries by Name
        self.df = pd.merge(proj_df, input_df[['Name', 'MaxExp']], on='Name', how='left')
        
        # Salary files from DK can be messy; we ensure we find the right column
        sal_col = [c for c in sal_df.columns if 'Salary' in c][0]
        self.df = pd.merge(self.df, sal_df[['Name', sal_col]], on='Name', how='left')
        
        # 3. PURIFICATION
        self.df = self.df.rename(columns={
            'Position': 'Pos', 
            'ProjPts': 'Proj', 
            'ProjOwn': 'Own', 
            sal_col: 'Sal'
        })
        
        # Ensure numbers are pure
        self.df['Proj'] = pd.to_numeric(self.df['Proj'], errors='coerce').fillna(0)
        self.df['Sal'] = pd.to_numeric(self.df['Sal'], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df['Own'], errors='coerce').fillna(10)
        self.df['MaxExp'] = pd.to_numeric(self.df['MaxExp'], errors='coerce').fillna(100) / 100.0

        # 4. BLACKLIST (Metcalf, Kittle, Warner, Gabe Davis)
        blacklist = ['DK Metcalf', 'Gabe Davis', 'George Kittle', 'Fred Warner', 'Joshua Palmer']
        self.df = self.df[~self.df['Name'].isin(blacklist)]
        self.df = self.df[self.df['Pos'] != 'K'] # No Kickers

    def cook(self, n_lineups=20):
        final_pool = []
        player_indices = self.df.index
        # Saturday 2-Game Slate: 135% Own Cap, $48,200 Floor, 3 Unique Players
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 135.0, 48200, 0.35, 3

        for n in range(n_lineups):
            sim_df = self.df.copy()
            # Jitter based on position
            for i in player_indices:
                row = sim_df.loc[i]
                std = JITTER * (1.3 if row['Pos'] in ['WR', 'TE'] else 0.8)
                sim_df.at[i, 'Sim_Proj'] = np.random.normal(row['Proj'], row['Proj'] * std)

            # SCENARIO BOOSTS (Jan 17, 2026 Strategy)
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

            # Positional Guards
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                mask = [choices[i] for i in player_indices if sim_df.loc[i, 'Pos'] == p]
                prob += pulp.lpSum(mask) >= mn
                prob += pulp.lpSum(mask) <= mx

            # Uniqueness Constraint
            for prev in final_pool:
                prob += pulp.lpSum([choices[i] for i in prev.index]) <= (9 - MIN_UNIQUE)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                final_pool.append(sim_df.loc[[i for i in player_indices if choices[i].varValue == 1]])
        return final_pool

# --- STREAMLIT UI ---
st.title("🧪 VANTAGE 99")
st.warning("⚠️ SATURDAY LOCKOUT IMMINENT (4:30 PM)")

# Robust File Uploader
p_file = st.file_uploader("1. Projections (CSV)", type="csv")
i_file = st.file_uploader("2. Inputs (CSV)", type="csv")
s_file = st.file_uploader("3. DK Salaries (CSV)", type="csv")

if p_file and i_file and s_file:
    try:
        # Hot-fix for ParserError: skip bad lines and use Python engine
        p_df = pd.read_csv(p_file, on_bad_lines='skip', engine='python')
        i_df = pd.read_csv(i_file, on_bad_lines='skip', engine='python')
        s_df = pd.read_csv(s_file, on_bad_lines='skip', engine='python')
        
        # Verification
        if 'Name' not in s_df.columns:
            st.error("Check Salary CSV: No 'Name' column found.")
        else:
            engine = Vantage99(p_df, i_df, s_df)
            if st.button("🔥 START THE INDUSTRIAL COOK"):
                lineups = engine.cook(n_lineups=20)
                st.header("📋 MANUAL ENTRY GUIDE")
                for i, l in enumerate(lineups):
                    st.subheader(f"Lineup {i+1}")
                    names = l[['Pos', 'Name']].sort_values('Pos').values
                    st.write(" | ".join([f"**{n[0]}**: {n[1]}" for n in names]))
                    st.divider()
    except Exception as e:
        st.error(f"Contamination Error: {e}")
