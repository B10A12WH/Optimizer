import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime

# --- VANTAGE 99: NBA TOURNAMENT GRADE LAB (V24.0) ---
st.set_page_config(page_title="VANTAGE 99 NBA", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 26px; color: #00ffcc; font-weight: bold; }
    .stAlert { border-radius: 10px; border: 1px solid #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

def deep_scan(df):
    for i, row in df.head(15).iterrows():
        vals = [str(v).lower() for v in row.values]
        if 'name' in vals and 'salary' in vals:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

class VantageNBA:
    def __init__(self, s_df, blacklisted_names=[]):
        s_df = deep_scan(s_df)
        s_df.columns = [str(c).strip() for c in s_df.columns]
        
        # 1. POSITION LOGIC
        for p in ['PG','SG','SF','PF','C']: 
            s_df[f'is_{p}'] = s_df['Position'].str.contains(p).astype(int)
        s_df['is_G'] = ((s_df['is_PG']==1)|(s_df['is_SG']==1)).astype(int)
        s_df['is_F'] = ((s_df['is_SF']==1)|(s_df['is_PF']==1)).astype(int)
        
        # 2. LATE SWAP CLOCK
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        s_df['Time'] = s_df['Game Info'].apply(get_time)
        
        # 3. PROJECTION ENGINE
        pace_mults = {'DEN': 1.12, 'WAS': 1.10, 'DET': 1.06, 'IND': 1.09, 'ATL': 1.08}
        s_df['Base'] = pd.to_numeric(s_df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        s_df['Proj'] = s_df['Base'] * s_df['TeamAbbrev'].map(pace_mults).fillna(1.0)
        
        # 4. SCRUB AUDITOR (Hardened with Jan 17 Injury List)
        # 100 = Guaranteed, 0 = Ruled Out
        scrubbed_out = [
            'Nikola Jokic', 'Pascal Siakam', 'Trae Young', 'Jalen Green', 
            'Andrew Nembhard', 'Jonas Valanciunas', 'Isaiah Hartenstein', 
            'Jaime Jaquez Jr.', 'Devin Vassell', 'Moussa Diabate', 
            'Christian Braun', 'T.J. McConnell', 'Zaccharie Risacher', 
            'Davion Mitchell', 'Jamaree Bouyea', 'Jayson Tatum'
        ]
        ruled_out = scrubbed_out + blacklisted_names
        s_df['Conf'] = 100
        s_df.loc[s_df['Name'].isin(ruled_out), 'Conf'] = 0
        
        self.df = s_df[s_df['Conf'] > 0].reset_index(drop=True)

    def cook_and_audit(self, n=50):
        pool = []
        n_p = len(self.df)
        proj_vals = self.df['Proj'].values.astype(float)
        sal_vals = self.df['Salary'].values.astype(float)
        
        for _ in range(n):
            sim = np.random.normal(proj_vals, proj_vals * 0.15).clip(min=0)
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(8); bu.append(8)
            A.append(sal_vals); bl.append(49200.0); bu.append(50000.0)
            for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: 
                A.append(self.df[c].values.astype(float)); bl.append(1.0); bu.append(8.0)
            A.append(self.df['is_G'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            A.append(self.df['is_F'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            
            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx].copy().reset_index(drop=True)
                
                # 5. ASSIGNMENT ENGINE
                latest_time = l_df['Time'].max()
                M, cost = np.zeros((8, 8)), np.zeros((8, 8))
                slots = ['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']
                for i, p in l_df.iterrows():
                    for j, c in enumerate(slots): 
                        if p[c]: M[i, j] = 1
                    M[i, 7] = 1 # UTIL
                    if p['Time'] == latest_time: cost[i, 7] = -5000
                
                A_as, bl_as, bu_as = [], [], []
                for i in range(8): r=np.zeros((8,8)); r[i,:]=1; A_as.append(r.flatten()); bl_as.append(1); bu_as.append(1)
                for j in range(8): c=np.zeros((8,8)); c[:,j]=1; A_as.append(c.flatten()); bl_as.append(1); bu_as.append(1)
                
                res_as = milp(c=cost.flatten(), constraints=LinearConstraint(A_as, bl_as, bu_as), integrality=np.ones(64), bounds=Bounds(0, M.flatten()))
                if res_as.success:
                    map_res = res_as.x.reshape((8, 8))
                    final_slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
                    rost = {final_slots[j]: f"{l_df.iloc[i]['Name']} ({l_df.iloc[i]['ID']})" for i in range(8) for j in range(8) if map_res[i, j] > 0.5}
                    pool.append({'roster': rost, 'score': sim[idx].sum(), 'Sal': int(l_df['Salary'].sum()), 'Conf': l_df['Conf'].mean()})
        
        scores = [p['score'] for p in pool]
        for p in pool:
            perc = sum(p['score'] >= s for s in scores) / len(scores)
            p['Grade'] = "A+" if perc >= 0.95 else "A" if perc >= 0.85 else "B" if perc >= 0.70 else "C"
        return sorted(pool, key=lambda x: x['score'], reverse=True)

# --- COMMAND CENTER UI ---
st.title("🏀 VANTAGE 99: INDUSTRIAL NBA COMMAND CENTER")
with st.expander("🛡️ INJURY STATUS AUDITOR (JAN 17 SCRUB)"):
    st.info("**Ruled Out (Nodes Dead):** Jokic, Siakam, T. Young, J. Green, Hartenstein, H. Jaquez Jr, Vassell, McConnell, C. Braun, Risacher, Valanciunas.")
    manual_bl = st.text_area("Add Manual Ruled-Out Players:", "")
    bl_list = [n.strip() for n in manual_bl.split("\n") if n.strip()]

f = st.file_uploader("Upload DraftKings Salary CSV", type="csv")
if f:
    engine = VantageNBA(pd.read_csv(f), bl_list)
    if st.button("🔥 GENERATE ALPHA GRADE LINEUP"):
        results = engine.cook_and_audit(250)
        if results:
            top = results[0]
            st.success(f"🏆 TOP ALPHA LINEUP: Grade {top['Grade']}")
            st.table(pd.DataFrame([top['roster']])[['PG','SG','SF','PF','C','G','F','UTIL']])
            st.download_button("📥 Download Upload File", pd.DataFrame([r['roster'] for r in results]).to_csv(index=False), "Vantage_NBA_Final.csv")
