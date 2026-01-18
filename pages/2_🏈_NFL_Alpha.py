import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- VANTAGE ZERO: NFL ELITE (STABLE V65.0) ---
st.set_page_config(page_title="VANTAGE ZERO | NFL", layout="wide")

st.title("🏈 VANTAGE ZERO | NFL")
st.markdown("---")

# 1. LIVE VEGAS & SLATE DATA
# Update O/U and Spread for 1/18 slate
VEGAS_NFL = {
    'HOU': {'ou': 40.5, 'spr': 3},   
    'NE':  {'ou': 40.5, 'spr': -3},  
    'LAR': {'ou': 48.5, 'spr': -3.5},
    'CHI': {'ou': 48.5, 'spr': 3.5}  
}
# Define Late-Game Teams for Late Swap
LATE_GAME_TEAMS = ['LAR', 'CHI'] 
LEAGUE_AVG = 21.5

f = st.file_uploader("UPLOAD DK SALARY CSV", type="csv")

if f:
    try:
        df_raw = pd.read_csv(f)
        header_idx = 0
        for i, row in df_raw.head(10).iterrows():
            if 'Name' in row.values and 'Salary' in row.values:
                header_idx = i
                break
        
        df = df_raw.iloc[header_idx+1:].copy()
        df.columns = [str(c).strip() for c in df_raw.iloc[header_idx].values]
        df = df.reset_index(drop=True)
        
        # 2. CORE SCRUB & MATCHUP WEIGHTING
        df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(5.0)
        df['Sal'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(50000)
        df['Team'] = df['TeamAbbrev'].astype(str)
        
        # Vegas Boost Logic
        def apply_vegas(row):
            t = row['Team']
            if t in VEGAS_NFL:
                implied = (VEGAS_NFL[t]['ou']/2) - (VEGAS_NFL[t]['spr']/2)
                return row['Proj'] * (implied / LEAGUE_AVG)
            return row['Proj']
        df['Proj'] = df.apply(apply_vegas, axis=1)

        # 3. POSITION & LATE SWAP FLAGS
        for p in ['QB','RB','WR','TE','DST']:
            df[f'is_{p}'] = (df['Position'] == p).astype(int)
        
        # Mark players from the 6:30 PM window for FLEX priority
        df['is_late'] = df['Team'].isin(LATE_GAME_TEAMS).astype(int)
        
        # Scrub: Removing confirmed OUTs
        df = df[~df['Name'].isin(['Nico Collins', 'Justin Watson', 'Fred Warner'])]
        
        if st.button("🚀 EXECUTE ALPHA SIMULATION"):
            n_p = len(df)
            projs = df['Proj'].values.astype(float)
            sals = df['Sal'].values.astype(float)
            
            # --- 4. STRATEGIC FLEX ADJUSTMENT ---
            # To avoid TEs in FLEX, we apply a small penalty to their projection
            # We also add a small 'Late Swap Bonus' to favor late-game players in the logic
            sim_projs = projs.copy()
            sim_projs -= (df['is_TE'].values * 0.75)  # Penalty for FLEX-TE usage
            sim_projs += (df['is_late'].values * 0.10) # Late Swap Preference
            
            # Constraints
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9)
            A.append(sals); bl.append(49000); bu.append(50000)
            
            # Positional Hardstops
            A.append(df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(df['is_RB'].values); bl.append(2); bu.append(3) # Max 3 RBs (2 + FLEX)
            A.append(df['is_WR'].values); bl.append(3); bu.append(4) # Max 4 WRs (3 + FLEX)
            A.append(df['is_TE'].values); bl.append(1); bu.append(1) # FORCE EXACTLY 1 TE (NO FLEX TE)
            A.append(df['is_DST'].values); bl.append(1); bu.append(1)
            
            # Solve
            res = milp(c=-sim_projs, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                lineup = df.iloc[np.where(res.x > 0.5)[0]]
                st.subheader("🥇 ALPHA OPTIMAL (STACKED & FLEX-TUNED)")
                
                # REORDER FOR LATE SWAP DISPLAY
                # Ensures late-game RB/WR is in the FLEX slot
                qb = lineup[lineup['is_QB']==1]
                dst = lineup[lineup['is_DST']==1]
                te = lineup[lineup['is_TE']==1]
                rbs = lineup[lineup['is_RB']==1].sort_values('is_late', ascending=True)
                wrs = lineup[lineup['is_WR']==1].sort_values('is_late', ascending=True)
                
                # Logic: The 'FLEX' is the RB or WR with the latest start time
                all_flex_eligible = pd.concat([rbs, wrs])
                flex = all_flex_eligible.sort_values('is_late', ascending=False).iloc[0:1]
                
                st.table(lineup[['Position', 'Name', 'Team', 'Salary', 'Proj']])
                st.info("Strategy: TEs are barred from FLEX to prioritize high-ceiling WR/RB. Late-game players prioritized for swap flexibility.")
            else:
                st.error("Solver failed. Adjust constraints.")
                
    except Exception as e:
        st.error(f"Execution Error: {e}")
