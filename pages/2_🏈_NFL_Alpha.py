import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- VANTAGE ZERO: NFL ELITE (STABLE V65.0) ---
st.set_page_config(page_title="VANTAGE ZERO | NFL", layout="wide")

# --- 1. DYNAMIC CONFIGURATION ---
# Update these weekly for the most accurate projections
VEGAS_DATA = {
    'HOU': {'ou': 40.5, 'spr': 3},   # Implied Total: 18.75
    'NE':  {'ou': 40.5, 'spr': -3},  # Implied Total: 21.75
    'LAR': {'ou': 48.5, 'spr': -3.5},# Implied Total: 26.0
    'CHI': {'ou': 48.5, 'spr': 3.5}  # Implied Total: 22.5
}
LEAGUE_AVG_PPG = 21.5 

st.title("🏈 VANTAGE ZERO | NFL ALPHA")
st.markdown("---")

f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")

if f:
    try:
        # Load and clean headers to avoid 'AvgPointsPerGame' KeyErrors
        df_raw = pd.read_csv(f)
        header_idx = 0
        for i, row in df_raw.head(15).iterrows():
            if 'Name' in row.values and 'Salary' in row.values:
                header_idx = i
                break
        
        df = df_raw.iloc[header_idx+1:].copy()
        df.columns = [str(c).strip() for c in df_raw.iloc[header_idx].values]
        df = df.reset_index(drop=True)
        
        # 2. MATCHUP-BASED WEIGHTING (Vegas Layer)
        # Identifies high-scoring environments for upside
        def get_column(df, targets):
            standardized = {c.lower().replace(" ", ""): c for c in df.columns}
            for t in targets:
                if t in standardized: return standardized[t]
            return df.columns[0]

        proj_col = get_column(df, ['avgpointspergame', 'proj', 'fppg'])
        df['Proj'] = pd.to_numeric(df[proj_col], errors='coerce').fillna(5.0)
        df['Sal'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(50000)
        df['Team'] = df['TeamAbbrev'].astype(str)

        def apply_vegas(row):
            t = row['Team']
            if t in VEGAS_DATA:
                implied = (VEGAS_DATA[t]['ou'] / 2) - (VEGAS_DATA[t]['spr'] / 2)
                return row['Proj'] * (implied / LEAGUE_AVG_PPG)
            return row['Proj']
        df['Proj'] = df.apply(apply_vegas, axis=1)

        # 3. POSITION FLAGS
        for p in ['QB','RB','WR','TE','DST']:
            df[f'is_{p}'] = (df['Position'] == p).astype(int)

        # 4. SOLVER EXECUTION
        if st.button("🚀 EXECUTE ALPHA SIMULATION"):
            n_p = len(df)
            projs = df['Proj'].values.astype(float)
            sals = df['Sal'].values.astype(float)
            
            # --- TOURNAMENT STRATEGY: FLEX PENALTY ---
            # We slightly penalize TE projections in the solver to favor WR/RB FLEX
            solver_projs = projs.copy()
            solver_projs -= (df['is_TE'].values * 0.5)

            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # Total Players
            A.append(sals); bl.append(49000); bu.append(50000) # Salary Cap

            # Positional Constraints (DraftKings Classic)
            A.append(df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(df['is_DST'].values); bl.append(1); bu.append(1)

            # --- CORRELATION: QB-WR STACKING ---
            # Forces at least one WR/TE from the same team if a QB is selected
            for team in df['Team'].unique():
                q_idx = df[(df['Team'] == team) & (df['is_QB'] == 1)].index.tolist()
                s_idx = df[(df['Team'] == team) & ((df['is_WR'] == 1) | (df['is_TE'] == 1))].index.tolist()
                if q_idx and s_idx:
                    row = np.zeros(n_p)
                    for s in s_idx: row[s] = 1  # Receive
                    for q in q_idx: row[q] = -1 # Quarterback
                    A.append(row); bl.append(0); bu.append(8)

            # Solve using MILP
            res = milp(c=-solver_projs, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                lineup = df.iloc[np.where(res.x > 0.5)[0]]
                st.subheader("🥇 ALPHA OPTIMAL (STACKED)")
                st.table(lineup[['Position', 'Name', 'Team', 'Salary', 'Proj']])
                st.info("Strategy Note: Lineup includes a QB stack and prioritized RB/WR for FLEX position.")
            else:
                st.error("Solver failed. Adjust constraints or check player pool.")

    except Exception as e:
        st.error(f"Mobile/Data Error: {e}")
