import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="VANTAGE ZERO | NFL", layout="wide")

st.title("🏈 VANTAGE ZERO | NFL")
st.markdown("---")

# 1. LIVE VEGAS DATA (Update these for your specific slate)
# Formula: Implied Total = (O/U / 2) - (Spread / 2)
VEGAS_NFL = {
    'HOU': {'ou': 40.5, 'spr': 3},   # Implied: 18.75
    'NE':  {'ou': 40.5, 'spr': -3},  # Implied: 21.75
    'LAR': {'ou': 48.5, 'spr': -3.5},# Implied: 26.0
    'CHI': {'ou': 48.5, 'spr': 3.5}  # Implied: 22.5
}
LEAGUE_AVG = 21.5

f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")

if f:
    try:
        # Load and strip hidden spaces from headers immediately
        df_raw = pd.read_csv(f)
        header_idx = 0
        for i, row in df_raw.head(15).iterrows():
            if 'Name' in row.values and 'Salary' in row.values:
                header_idx = i
                break
        
        df = df_raw.iloc[header_idx+1:].copy()
        df.columns = [str(c).strip() for c in df_raw.iloc[header_idx].values]
        df = df.reset_index(drop=True)

        # --- FIX FOR 'AvgPointsPerGame' ERROR ---
        # We search for the closest matching column name to avoid KeyErrors
        target_col = next((c for c in df.columns if 'Avg' in c and 'Points' in c), None)
        if target_col:
            df['Proj'] = pd.to_numeric(df[target_col], errors='coerce').fillna(5.0)
        else:
            st.warning("Could not find 'AvgPointsPerGame'. Using default of 5.0.")
            df['Proj'] = 5.0

        # Mapping core variables
        df['Sal'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(50000)
        df['Team'] = df['TeamAbbrev'].astype(str)
        df['Own'] = pd.to_numeric(df.get('Ownership', 15.0), errors='coerce').fillna(15.0)

        # 2. VEGAS MULTIPLIER (The Pro Edge)
        def apply_vegas(row):
            t = row['Team']
            if t in VEGAS_NFL:
                implied = (VEGAS_NFL[t]['ou']/2) - (VEGAS_NFL[t]['spr']/2)
                return row['Proj'] * (implied / LEAGUE_AVG)
            return row['Proj']
        df['Proj'] = df.apply(apply_vegas, axis=1)

        # Position Flags
        for p in ['QB','RB','WR','TE','DST']:
            df[f'is_{p}'] = (df['Position'] == p).astype(int)
        
        st.success(f"Successfully loaded {len(df)} players with Vegas-adjusted projections.")

        if st.button("🚀 EXECUTE ALPHA SIMULATION"):
            n_p = len(df)
            projs = df['Proj'].values.astype(float)
            sals = df['Sal'].values.astype(float)
            owns = df['Own'].values.astype(float)
            
            # Constraints
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # Size
            A.append(sals); bl.append(49000); bu.append(50000) # Budget
            
            # Positional Limits
            A.append(df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(df['is_WR'].values); bl.append(3); bu.append(4)
