import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

# --- VANTAGE 99: THE PERFECT COOK (V11.2) ---
st.set_page_config(page_title="VANTAGE 99", layout="wide", page_icon="🧪")

class Vantage99:
    def __init__(self, proj_df, input_df, sal_df):
        # 1. THE REACTION: Robust Column Finding
        # Standardize all headers to lowercase and strip spaces
        for d in [proj_df, input_df, sal_df]:
            d.columns = [str(c).strip().lower() for c in d.columns]
        
        # Identify critical columns in the Salary file
        # Matches 'name', 'player name', 'player', etc.
        name_col = [c for c in sal_df.columns if 'name' in c][0]
        # Matches 'salary'
        sal_col = [c for c in sal_df.columns if 'salary' in c][0]
        
        # 2. MOLECULAR MERGE
        # Standardize merge keys to Title Case to match names
        proj_df['name'] = proj_df['name'].str.title()
        input_df['name'] = input_df['name'].str.title()
        sal_df[name_col] = sal_df[name_col].str.title()

        self.df = pd.merge(proj_df, input_df[['name', 'maxexp']], on='name', how='left')
        self.df = pd.merge(self.df, sal_df[[name_col, sal_col]], left_on='name', right_on=name_col, how='left')
        
        # 3. PURIFICATION
        # Rename to engine-readable standards
        self.df = self.df.rename(columns={
            'position': 'pos', 
            'projpts': 'proj', 
            'projown': 'own', 
            sal_col: 'sal'
        })
        
        # Force numeric values and handle contaminates
        self.df['proj'] = pd.to_numeric(self.df['proj'], errors='coerce').fillna(0)
        self.df['sal'] = pd.to_numeric(self.df['sal'], errors='coerce').fillna(50000)
        self.df['own'] = pd.to_numeric(self.df['own'], errors='coerce').fillna(10)
        
        # 4. BLACKLIST (Metcalf, Davis, Kittle, Warner)
        blacklist = ['Dk Metcalf', 'Gabe Davis', 'George Kittle', 'Fred Warner']
        self.df = self.df[~self.df['name'].isin(blacklist)]
        self.df = self.df[self.df['pos'].str.upper() != 'K'] # No Kickers

    def cook(self, n_lineups=20):
        final_pool = []
        player_indices = self.df.index
        # Saturday 2-Game Slate Constants
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 135.0, 48200, 0.35, 3

        for n in range(n_lineups):
            sim_df = self.df.copy()
            # Apply Jitter
            for i in player_indices:
                row = sim_df.loc[i]
                std = JITTER * (1.3 if row['pos'].upper() in ['WR', 'TE'] else 0.8)
                sim_df.at[i, 'sim_proj'] = np.random.normal(row['proj'], row['proj'] * std)

            # SCENARIO CATALYSTS
            sim_df.loc[sim_df['name'] == 'Jaxon Smith-Njigba', 'sim_proj'] *= 1.35
            sim_df.loc[sim_df['name'] == 'Jauan Jennings', 'sim_proj'] *= 1.40
            sim_df.loc[sim_df['name'] == 'Kenneth Walker Iii', 'sim_proj'] *= 1.30
            sim_df.loc[sim_df['name'] == 'Khalil Shakir', 'sim_proj'] *= 1.25
            sim_df.loc[sim_df['name'] == 'Rj Harvey', 'sim_proj'] *= 1.15
            sim_df.loc[sim_df['name'] == 'Josh Allen', 'sim_proj'] *= 0.88 

            # THE SOLVER
            prob = pulp.LpProblem(f"Batch_{n}", pulp.LpMaximize)
            choices = pulp.LpVariable.dicts("P", player_indices, cat='Binary')

            prob += pulp.lpSum([sim_df.loc[i, 'sim_proj'] * choices[i] for i in player_indices])
            prob += pulp.lpSum([choices[i] for i in player_indices]) == 9
            prob += pulp.lpSum([sim_df.loc[i, 'sal'] * choices[i] for i in player_indices]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'sal'] * choices[i] for i in player_indices]) >= SAL_FLOOR
            prob += pulp.lpSum([sim_df.loc[i, 'own'] * choices[i] for i in player_indices]) <= OWN_CAP

            # Position Guards
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                mask = [choices[i] for i in player_indices if sim_df.loc[i, 'pos'].upper() == p]
                prob += pulp.lpSum(mask) >= mn
                prob += pulp.lpSum(mask) <= mx

            # Uniqueness
            for prev in final_pool:
                prob += pulp.lpSum([choices[i] for i in prev.index]) <= (9 - MIN_UNIQUE)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                final_pool.append(sim_df.loc[[i for i in player_indices if choices[i].varValue == 1]])
        return final_pool

# --- UI ---
st.title("🧪 VANTAGE 99")
st.warning("⚠️ 4:30 PM LOCKOUT IMMINENT")

p_file = st.file_uploader("1. Projections", type="csv")
i_file = st.file_uploader("2. Inputs", type="csv")
s_file = st.file_uploader("3. DK Salary CSV", type="csv")

if p_file and i_file and s_file:
    try:
        p_df = pd.read_csv(p_file, on_bad_lines='skip', engine='python')
        i_df = pd.read_csv(i_file, on_bad_lines='skip', engine='python')
        s_df = pd.read_csv(s_file, on_bad_lines='skip', engine='python')
        
        engine = Vantage99(p_df, i_df, s_df)
        if st.button("🔥 START THE INDUSTRIAL COOK"):
            lineups = engine.cook()
            st.header("📋 MANUAL ENTRY GUIDE")
            for i, l in enumerate(lineups):
                with st.expander(f"Lineup {i+1}"):
                    names = l[['pos', 'name']].sort_values('pos').values
                    st.write(" | ".join([f"**{n[0].upper()}**: {n[1]}" for n in names]))
    except Exception as e:
        st.error(f"Contamination Error: {e}")
