import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime

# --- VANTAGE 99: INDUSTRIAL NBA COMMAND CENTER ---
st.set_page_config(page_title="VANTAGE 99 NBA", layout="wide", page_icon="🏀")

# --- FIXED CSS FOR PRODUCTION ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True) # Fixed the typo here

def deep_scan(df):
    for i, row in df.head(15).iterrows():
        vals = [str(v).lower() for v in row.values]
        if 'name' in vals and 'salary' in vals:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

class VantageNBA:
    def __init__(self, s_df, jitter=0.15):
        s_df = deep_scan(s_df)
        s_df.columns = [str(c).strip() for c in s_df.columns]
        for p in ['PG','SG','SF','PF','C']: 
            s_df[f'is_{p}'] = s_df['Position'].str.contains(p).astype(int)
        s_df['is_G'] = ((s_df['is_PG']==1)|(s_df['is_SG']==1)).astype(int)
        s_df['is_F'] = ((s_df['is_SF']==1)|(s_df['is_PF']==1)).astype(int)
        
        def get_time(x):
            m = re.search(r'(\d{2}:\d{2}[APM]+)', str(x))
            return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
        s_df['Time'] = s_df['Game Info'].apply(get_time)
        s_df['Proj'] = pd.to_numeric(s_df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        self.df = s_df.reset_index(drop=True)
        self.jitter = jitter

    def cook_and_audit(self, n=50):
        pool = []
        n_p = len(self.df)
        sal_vals = self.df['Salary'].values.astype(float)
        
        for _ in range(n):
            sim = np.random.normal(self.df['Proj'], self.df['Proj'] * self.jitter).clip(min=0)
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(8); bu.append(8)
            A.append(sal_vals); bl.append(49000.0); bu.append(50000.0)
            for c in ['is_PG','is_SG','is_SF','is_PF','is_C']: 
                A.append(self.df[c].values.astype(float)); bl.append(1.0); bu.append(8.0)
            A.append(self.df['is_G'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            A.append(self.df['is_F'].values.astype(float)); bl.append(3.0); bu.append(8.0)
            
            res = milp(c=-sim, constraints=LinearConstraint(A, bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                l_df = self.df.iloc[idx].copy().reset_index(drop=True)
                latest_time = l_df['Time'].max()
                
                # ASSIGNMENT ENGINE
                M = np.zeros((8, 8))
                cost = np.zeros((8, 8))
                for i, p in l_df.iterrows():
                    for j, cond in enumerate(['is_PG','is_SG','is_SF','is_PF','is_C','is_G','is_F']):
                        if p[cond]: M[i, j] = 1
                    M[i, 7] = 1 
                    if p['Time'] == latest_time: cost[i, 7] = -1000 
                
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
                            if mapping[i, j] > 0.5: rost[slots[j]] = f"{l_df.iloc[i]['Name']} ({l_df.iloc[i]['ID']})"
                    
                    pool.append({
                        'roster': rost, 'score': sim[idx].sum(), 
                        'LateSwap': "✅ PASS" if l_df[l_df['ID'] == int(re.search(r'\((\d+)\)', rost['UTIL']).group(1))]['Time'].iloc[0] == latest_time else "⚠️",
                        'Sal': int(l_df['Salary'].sum()), 'players': l_df['Name'].tolist()
                    })
        
        if not pool: return []
        scores = [p['score'] for p in pool]
        for p in pool:
            perc = sum(p['score'] >= s for s in scores) / len(scores)
            if perc >= 0.95: p['Grade'] = "A+"
            elif perc >= 0.85: p['Grade'] = "A"
            elif perc >= 0.70: p['Grade'] = "B"
            else: p['Grade'] = "C"
        return pool

# --- UI ---
st.title("🏀 VANTAGE 99: INDUSTRIAL COMMAND CENTER")
f = st.file_uploader("Upload DraftKings NBA Salary CSV", type="csv")

if f:
    engine = VantageNBA(pd.read_csv(f))
    if st.button("🚀 INITIATE INDUSTRIAL BATCH"):
        results = engine.cook_and_audit(50)
        
        if results:
            # Sort by Score for Grade A+ identification
            sorted_res = sorted(results, key=lambda x: x['score'], reverse=True)
            top_lineup = sorted_res[0]

            # --- TOP ALPHA DASHBOARD ---
            st.success(f"🏆 TOP ALPHA BUILD IDENTIFIED (Grade {top_lineup['Grade']})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Simulated Score", round(top_lineup['score'], 2))
            c2.metric("Salary Used", f"${top_lineup['Sal']}")
            c3.metric("UTIL Audit", top_lineup['LateSwap'])
            
            st.json(top_lineup['roster'])

            # --- TABS FOR BATCH DATA ---
            t1, t2 = st.tabs(["📊 Exposure Report", "🔍 Full Batch Audit"])
            with t1:
                all_p = [p for r in results for p in r['players']]
                exp = pd.Series(all_p).value_counts().reset_index()
                exp.columns = ['Player', 'Count']
                st.dataframe(exp)
            with t2:
                audit_df = pd.DataFrame([{'Grade': r['Grade'], 'Salary': r['Sal'], 'LateSwap': r['LateSwap']} for r in results])
                st.dataframe(audit_df)
