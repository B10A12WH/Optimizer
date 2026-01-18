import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- VANTAGE ZERO: NFL ELITE (STABLE V66.0) ---
st.set_page_config(page_title="VANTAGE ZERO | NFL", layout="wide")

st.title("🏈 VANTAGE ZERO | NFL")
st.markdown("---")

# 1. LIVE VEGAS & SLATE DATA
# Houston@NE (3 PM), Rams@Chicago (6:30 PM)
VEGAS_NFL = {
    'HOU': {'ou': 40.5, 'spr': 3},   
    'NE':  {'ou': 40.5, 'spr': -3},  
    'LAR': {'ou': 48.5, 'spr': -3.5},
    'CHI': {'ou': 48.5, 'spr': 3.5}  
}
LATE_GAME_TEAMS = ['LAR', 'CHI'] 
LEAGUE_AVG_PPG = 21.5

f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")

if f:
    try:
        # DEEP HEADER SCAN: Fixes the 'Salary' KeyError by finding the real data start
        df_raw = pd.read_csv(f)
        header_idx = None
        for i, row in df_raw.head(15).iterrows():
            vals = [str(v).lower().strip() for v in row.values]
            if 'name' in vals and 'salary' in vals:
                header_idx = i
                break
        
        if header_idx is not None:
            df = df_raw.iloc[header_idx+1:].copy()
            df.columns = [str(c).strip() for c in df_raw.iloc[header_idx].values]
        else:
            df = df_raw.copy()
            df.columns = df.columns.str.strip()

        df = df.reset_index(drop=True)
        
        # 2. DYNAMIC COLUMN MAPPING
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        sal_key = cols.get('salary', 'Salary')
        proj_key = next((cols[k] for k in cols if k in ['proj', 'fppg', 'avgpointspergame']), df.columns[0])
        
        df['Proj'] = pd.to_numeric(df[proj_key], errors='coerce').fillna(5.0)
        df['Sal'] = pd.to_numeric(df[sal_key], errors='coerce').fillna(50000)
        df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        
        # 3. VEGAS WEIGHTING & POSITIONING
        def apply_vegas(row):
            t = row['Team']
            if t in VEGAS_NFL:
                line = VEGAS_NFL[t]
                implied = (line['ou']/2) - (line['spr']/2)
                return row['Proj'] * (implied / LEAGUE_AVG_PPG)
            return row['Proj']
        
        df['Proj'] = df.apply(apply_vegas, axis=1)
        df['is_late'] = df['Team'].isin(LATE_GAME_TEAMS).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']:
            df[f'is_{p}'] = (df['Position'] == p).astype(int)
        
        # Scrub confirmed OUTs
        df = df[~df['Name'].isin(['Nico Collins', 'Justin Watson', 'Fred Warner'])]

        if st.button("🚀 EXECUTE ALPHA SIMULATION"):
            n_p = len(df)
            projs = df['Proj'].values.astype(float)
            sals = df['Sal'].values.astype(float)
            
            # --- STRATEGIC FLEX TUNING ---
            # 1. Penalize TEs in Flex: TEs rarely outscore WR/RBs in GPPs
            # 2. Late Swap Bonus: Favor late-game players for the 9th spot
            solver_projs = projs.copy()
            solver_projs -= (df['is_TE'].values * 0.85)  # Discourage Flex-TE
            solver_projs += (df['is_late'].values * 0.15) # Flex-Late Swap preference
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49000); bu.append(50000)
            
            # Position Requirements
            A.append(df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(df['is_TE'].values); bl.append(1); bu.append(1) # FORCE 1 TE (NO FLEX TE)
            A.append(df['is_DST'].values); bl.append(1); bu.append(1)
            
            # Solve
            res = milp(c=-solver_projs, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                lineup = df.iloc[np.where(res.x > 0.5)[0]]
                st.subheader("🥇 ALPHA OPTIMAL")
                
                # Re-sort display for DK order (Putting latest player in FLEX)
                lineup = lineup.sort_values(['Position', 'is_late'], ascending=[True, True])
                st.table(lineup[['Position', 'Name', 'Team', 'Salary', 'Proj']])
                st.info("FLEX Priority: Logic forces WR/RB into FLEX and favors late-start players for swap flexibility.")
            else:
                st.error("Solver failed. Adjust constraints.")
                
    except Exception as e:
        st.error(f"Error: {e}")
