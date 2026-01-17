import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime

# --- VANTAGE 99: INDUSTRIAL NBA LAB ---
st.set_page_config(page_title="VANTAGE 99 NBA", layout="wide", page_icon="🏀")

def deep_scan(df):
    """Bypasses DraftKings metadata to find actual column headers."""
    for i, row in df.head(15).iterrows():
        vals = [str(v).lower() for v in row.values]
        if 'name' in vals and 'salary' in vals:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

class VantageNBA:
    def __init__(self, s_df):
        s_df = deep_scan(s_df)
        s_df.columns = [str(c).strip() for c in s_df.columns]
        
        # Position Engine
        for p in ['PG','SG','SF','PF','C']: 
            s_df[f'is_{p}'] = s_df['Position'].str.contains(p).astype(int)
        s_df['is_G'] = ((s_df['is_PG']==1)|(s_df['is_SG']==1)).astype(int)
        s_df['is_F'] = ((s_df['is_SF']==1)|(s_df['is_PF']==1)).astype(int)
        
        # Late Swap Engine (Clock Audit)
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        s_df['Time'] = s_df['Game Info'].apply(get_time)
        
        # Industrial Projection Simulation (Pace Adjustment)
        # Multipliers: High-Total games (DEN, WAS, IND, DET)
        pace_mults = {'DEN': 1.12, 'WAS': 1.10, 'DET': 1.06, 'IND': 1.09, 'ATL': 1.08}
        s_df['Base'] = pd.to_numeric(s_df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        s_df['Proj'] = s_df['Base'] * s_df['TeamAbbrev'].map(pace_mults).fillna(1.0)
        
        self.df = s_df.reset_index(drop=True)

    def cook_and_audit(self, n=50):
        pool = []
        n_p = len(self.df)
        sal_vals = self.df['Salary'].values.astype(float)
        
        for _ in range(n):
            # 1. MONTE CARLO JITTER (Standard NBA Deviation)
            sim = np.random.normal(self.df['Proj'], self.df['Proj'] * 0.15).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(8); bu.append(8) # 8 Roster slots
            A.append(sal_vals); bl.append(49000.0); bu.append(50000.0) # Salary Floor/Ceil
            
            # Position Requirements
            for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: 
                A.append(self.df[c].values.astype(float)); bl.append(1.0); bu.append(8.0)
            A.append(self.df['is_G'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            A.append(self.df['is_F'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            
            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx].copy().reset_index(drop=True)
                
                # 2. ASSIGNMENT ENGINE (UTIL late-game lock)
                latest_time = l_df['Time'].max()
                latest_players = l_df[l_df['Time'] == latest_time]['ID'].tolist()
                
                M = np.zeros((8, 8))
                cost = np.zeros((8, 8))
                for i, p in l_df.iterrows():
                    for j, cond in enumerate(['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']):
                        if p[cond]: M[i, j] = 1
                    M[i, 7] = 1 # UTIL slot eligibility
                    if p['ID'] in latest_players: cost[i, 7] = -1000 # Force latest to UTIL
                
                A_as, bl_as, bu_as = [], [], []
                for i in range(8): r=np.zeros((8,8)); r[i,:]=1; A_as.append(r.flatten()); bl_as.append(1); bu_as.append(1)
                for j in range(8): c=np.zeros((8,8)); c[:,j]=1; A_as.append(c.flatten()); bl_as.append(1); bu_as.append(1)
                
                res_as = milp(c=cost.flatten(), constraints=LinearConstraint(A_as, bl_as, bu_as), integrality=np.ones(64), bounds=Bounds(0, M.flatten()))
                
                if res_as.success:
                    mapping = res_as.x.reshape((8, 8))
                    rost = {}
                    slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
                    for i in range(8):
                        for j in range(8):
                            if mapping[i, j] > 0.5: 
                                rost[slots[j]] = f"{l_df.iloc[i]['Name']} ({l_df.iloc[i]['ID']})"
                    
                    # Audit Verification
                    util_name = rost['UTIL']
                    util_id = int(re.search(r'\((\d+)\)', util_name).group(1))
                    is_late = l_df[l_df['ID'] == util_id]['Time'].iloc[0] == latest_time
                    
                    pool.append({
                        'roster': rost, 
                        'score': sim[idx].sum(), 
                        'LateSwap_Audit': "✅ PASS" if is_late else "⚠️ SEMI",
                        'Sal': int(l_df['Salary'].sum())
                    })
        
        # 3. INDUSTRIAL GRADING (Percentile-based)
        scores = [p['score'] for p in pool]
        for p in pool:
            perc = sum(p['score'] >= s for s in scores) / len(scores)
            if perc >= 0.95: p['Grade'] = "A+"
            elif perc >= 0.85: p['Grade'] = "A"
            elif perc >= 0.70: p['Grade'] = "B"
            else: p['Grade'] = "C"
        return pool

# --- UI ---
st.title("🧪 VANTAGE 99: INDUSTRIAL NBA")
f = st.file_uploader("Upload DK NBA Salary CSV", type="csv")
if f:
    if st.button("🔥 GENERATE GRADED LINEUP"):
        results = VantageNBA(pd.read_csv(f)).cook_and_audit(50)
        df_out = pd.DataFrame([r['roster'] for r in results])
        df_out['Grade'] = [r['Grade'] for r in results]
        df_out['Salary'] = [r['Sal'] for r in results]
        df_out['UTIL Audit'] = [r['LateSwap_Audit'] for r in results]
        
        # High Precision View
        st.subheader("📋 Industrial Grade Report")
        st.dataframe(df_out.sort_values('Grade'))
        st.download_button("📥 Download Upload File", df_out.drop(columns=['Grade','Salary','UTIL Audit']).to_csv(index=False), "Vantage_NBA_Takedown.csv")
