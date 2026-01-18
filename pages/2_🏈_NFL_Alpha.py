import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- VANTAGE ZERO: NFL CORE (STABLE V51.2) ---
st.set_page_config(page_title="VANTAGE ZERO | NFL", layout="wide")

st.title("🏈 VANTAGE ZERO | NFL")
st.markdown("---")

# 1. DATA LOADER
f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")

if f:
    try:
        df_raw = pd.read_csv(f)
        # Search for the header row
        header_idx = 0
        for i, row in df_raw.head(10).iterrows():
            if 'Name' in row.values and 'Salary' in row.values:
                header_idx = i
                break
        
        df = df_raw.iloc[header_idx+1:].copy()
        df.columns = df_raw.iloc[header_idx].values
        df = df.reset_index(drop=True)
        
        # 2. CORE SCRUB & MAPPING
        df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(5.0)
        df['Sal'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(50000)
        df['Team'] = df['TeamAbbrev'].astype(str)
        
        # Scrub: Texans@Patriots | Rams@Bears Active Roster
        # Removing confirmed OUTs for 1/18
        df = df[~df['Name'].isin(['Nico Collins', 'Justin Watson', 'Fred Warner'])]
        
        st.success(f"Loaded {len(df)} eligible players for Sunday Slate.")
        
        # 3. POSITION FLAGS
        for p in ['QB','RB','WR','TE','DST']:
            df[f'is_{p}'] = (df['Position'] == p).astype(int)
            
        # 4. SIMULATION TRIGGER
        if st.button("🚀 EXECUTE ALPHA SIMULATION"):
            n_p = len(df)
            projs = df['Proj'].values.astype(float)
            sals = df['Sal'].values.astype(float)
            
            # Constraints
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49000); bu.append(50000)
            
            # Positional Hardstops
            A.append(df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(df['is_TE'].values); bl.append(1); bu.append(2)
            A.append(df['is_DST'].values); bl.append(1); bu.append(1)
            
            # Solve
            res = milp(c=-projs, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                lineup = df.iloc[np.where(res.x > 0.5)[0]]
                st.subheader("🥇 ALPHA OPTIMAL")
                st.table(lineup[['Position', 'Name', 'TeamAbbrev', 'Salary', 'Proj']])
            else:
                st.error("Solver failed to find a valid lineup. Check your constraints.")
                
    except Exception as e:
        st.error(f"Mobile Error: {e}")
