import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="VANTAGE ZERO | NFL ALPHA", layout="wide")

# --- 1. CONFIGURATION: VEGAS & TOURNAMENT SETTINGS ---
# To reach 1st place, we target high-scoring environments
# Update these values Sunday morning based on current lines
VEGAS_LINES = {
    'HOU': {'ou': 40.5, 'spread': 3},   # Underdog
    'NE':  {'ou': 40.5, 'spread': -3},  # Favorite
    'LAR': {'ou': 48.5, 'spread': -3.5},
    'CHI': {'ou': 48.5, 'spread': 3.5}
}
LEAGUE_AVG_PPG = 21.5  # Baseline for 'Value'

st.title("🏈 VANTAGE ZERO | NFL ALPHA")
st.markdown("---")

f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")

if f:
    try:
        df_raw = pd.read_csv(f)
        # Dynamic Header Detection
        header_idx = 0
        for i, row in df_raw.head(10).iterrows():
            if 'Name' in row.values and 'Salary' in row.values:
                header_idx = i
                break
        
        df = df_raw.iloc[header_idx+1:].copy()
        df.columns = df_raw.iloc[header_idx].values
        df = df.reset_index(drop=True)
        
        # --- 2. THE VEGAS PROJECTION LAYER ---
        # We transform season averages into matchup-specific expectations
        df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(5.0)
        df['Sal'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(50000)
        df['Team'] = df['TeamAbbrev'].astype(str)
        df['Own'] = pd.to_numeric(df.get('Ownership', 15.0), errors='coerce').fillna(15.0)

        def apply_vegas_lift(row):
            team = row['Team']
            if team in VEGAS_LINES:
                # Implied Total = (O/U / 2) - (Spread / 2)
                line = VEGAS_LINES[team]
                implied = (line['ou'] / 2) - (line['spread'] / 2)
                multiplier = implied / LEAGUE_AVG_PPG
                return row['Proj'] * multiplier
            return row['Proj']

        df['Proj'] = df.apply(apply_vegas_lift, axis=1)
        
        # Position Mapping
        for p in ['QB','RB','WR','TE','DST']:
            df[f'is_{p}'] = (df['Position'] == p).astype(int)
            
        if st.button("🚀 EXECUTE ALPHA SIMULATION"):
            n_p = len(df)
            projs = df['Proj'].values.astype(float)
            sals = df['Sal'].values.astype(float)
            owns = df['Own'].values.astype(float)
            
            # --- 3. CONSTRAINTS MATRIX ---
            A, bl, bu = [], [], []
            
            # Roster Basics
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49000); bu.append(50000)
            
            # Positional Limits (Classic DraftKings)
            A.append(df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(df['is_DST'].values); bl.append(1); bu.append(1)
            
            # Leverage Cap (Prevents 100% "Chalk" lineups)
            A.append(owns); bl.append(0); bu.append(125.0) 

            # --- 4. CONDITIONAL STACKING LOGIC ---
            # For every team, if QB is picked, at least 1 WR/TE must be picked
            for team in df['Team'].unique():
                qb_vars = df[(df['Team'] == team) & (df['is_QB'] == 1)].index.tolist()
                stack_vars = df[(df['Team'] == team) & ((df['is_WR'] == 1) | (df['is_TE'] == 1))].index.tolist()
                
                if qb_vars and stack_vars:
                    row = np.zeros(n_p)
                    for s in stack_vars: row[s] = 1 # Coefficient for receivers
                    for q in qb_vars: row[q] = -1   # Coefficient for QB
                    # Math: Sum(Receivers) - QB >= 0  => If QB=1, Sum must be >= 1
                    A.append(row); bl.append(0); bu.append(9)

            # Solve using Mixed-Integer Linear Programming
            res = milp(c=-projs, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                lineup = df.iloc[np.where(res.x > 0.5)[0]]
                st.subheader("🥇 ALPHA OPTIMAL (STACKED)")
                st.table(lineup[['Position', 'Name', 'Team', 'Salary', 'Proj', 'Own']])
                st.write(f"**Total Ownership:** {lineup['Own'].sum():.1f}%")
            else:
                st.error("Solver failed. Adjust constraints.")
                
    except Exception as e:
        st.error(f"Execution Error: {e}")
