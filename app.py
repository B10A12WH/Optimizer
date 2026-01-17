import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
from datetime import datetime

st.set_page_config(page_title="VANTAGE 99 NBA", layout="wide", page_icon="🏀")

def parse_game_time(info_str):
    try:
        time_match = re.search(r'(\d{2}:\d{2}[APM]+)', str(info_str))
        return datetime.strptime(time_match.group(1), '%I:%M%p') if time_match else datetime.min
    except: return datetime.min

def deep_scan(df):
    for i, row in df.head(15).iterrows():
        row_vals = [str(v).lower() for v in row.values]
        if any('name' in v for v in row_vals) and any('salary' in v for v in row_vals):
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

class VantageNBA:
    def __init__(self, s_df):
        s_df = deep_scan(s_df)
        s_df.columns = [str(c).strip() for c in s_df.columns]
        
        # Position Logic
        for p in ['PG','SG','SF','PF','C']:
            s_df[f'is_{p}'] = s_df['Position'].str.contains(p).astype(int)
        s_df['is_G'] = ((s_df['is_PG']==1)|(s_df['is_SG']==1)).astype(int)
        s_df['is_F'] = ((s_df['is_SF']==1)|(s_df['is_PF']==1)).astype(int)
        
        # Late Swap Engine
        s_df['GameTime'] = s_df['Game Info'].apply(parse_game_time)
        
        # Vegas Simulation Logic
        pace_mults = {'DEN': 1.12, 'WAS': 1.10, 'DET': 1.06, 'IND': 1.09, 'ATL': 1.08, 'BOS': 1.05}
        s_df['Base'] = pd.to_numeric(s_df['AvgPointsPerGame'], errors='coerce').fillna(10.0).clip(lower=5.0)
        s_df['Proj'] = s_df['Base'] * s_df['TeamAbbrev'].map(pace_mults).fillna(1.0)
        
        self.df = s_df.reset_index(drop=True)

    def cook(self, n=50): # Cooking 50 to find the pure 'A' Grade
        pool = []
        n_p = len(self.df)
        for _ in range(n):
            sim = np.random.normal(self.df['Proj'], self.df['Proj'] * 0.2).clip(min=0)
            A, b_l, b_u = [], [], []
            A.append(np.ones(n_p)); b_l.append(8); b_u.append(8)
            A.append(self.df['Salary'].values.astype(float)); b_l.append(49000.0); b_u.append(50000.0)
            for p in ['is_PG','is_SG','is_SF','is_PF','is_C']:
                A.append(self.df[p].values.astype(float)); b_l.append(1.0); b_u.append(8.0)
            A.append(self.df['is_G'].values.astype(float)); b_l.append(3.0); b_u.append(8.0)
            A.append(self.df['is_F'].values.astype(float)); b_l.append(3.0); b_u.append(8.0)
            
            res = milp(c=-sim, constraints=LinearConstraint(A, b_l, b_u), integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                res_df = self.df.iloc[np.where(res.x > 0.5)[0]]
                pool.append({'df': res_df, 'score': sim[res.x > 0.5].sum()})
        return pool

    def assemble_and_grade(self, lineup_data, all_scores):
        lineup_df = lineup_data['df']
        score = lineup_data['score']
        
        # 1. ASSEMBLY (Late Swap UTIL)
        p_sorted = lineup_df.sort_values('GameTime', ascending=False)
        latest_id = p_sorted.iloc[0]['ID']
        roster = {}
        p_pool = lineup_df.copy()
        for slot, cond in [('PG','is_PG'),('SG','is_SG'),('SF','is_SF'),('PF','is_PF'),('C','is_C'),('G','is_G'),('F','is_F')]:
            match = p_pool[(p_pool[cond]==1) & (p_pool['ID'] != latest_id)].sort_values('Proj', ascending=False).head(1)
            if match.empty: match = p_pool[p_pool[cond]==1].sort_values('Proj', ascending=False).head(1)
            roster[slot] = f"{match.iloc[0]['Name']} ({match.iloc[0]['ID']})"
            p_pool = p_pool.drop(match.index)
        roster['UTIL'] = f"{p_pool.iloc[0]['Name']} ({p_pool.iloc[0]['ID']})"
        
        # 2. GRADING SYSTEM
        percentile = sum(score >= s for s in all_scores) / len(all_scores)
        if percentile >= 0.95: grade = "A+"
        elif percentile >= 0.85: grade = "A"
        elif percentile >= 0.70: grade = "B"
        else: grade = "C"
        
        # 3. AUDIT PROCESSES
        # Auditor 1: Roster Authenticity & Salary
        sal_check = lineup_df['Salary'].astype(int).sum() <= 50000
        util_check = p_pool.iloc[0]['GameTime'] == lineup_df['GameTime'].max()
        audit_1 = "✅ PASS" if (sal_check and util_check) else "❌ FAIL"
        
        # Auditor 2: Grade Integrity
        # Confirms the score actually belongs to the assigned percentile
        expected_grade = "A+" if score >= np.percentile(all_scores, 95) else "A" if score >= np.percentile(all_scores, 85) else "B" if score >= np.percentile(all_scores, 70) else "C"
        audit_2 = "✅ VERIFIED" if grade == expected_grade else "❌ DISCREPANCY"
        
        return roster, grade, audit_1, audit_2

# --- UI ---
st.title("🧪 VANTAGE 99: NBA INDUSTRIAL OPTIMIZER")
f_sal = st.file_uploader("Upload NBA Salary CSV", type="csv")

if f_sal:
    engine = VantageNBA(pd.read_csv(f_sal))
    if st.button("🔥 GENERATE THE PERFECT LINEUP"):
        batch = engine.cook(100) # Large batch for high-precision grading
        all_scores = [b['score'] for b in batch]
        final_lineups = []
        for b in batch:
            rost, grd, a1, a2 = engine.assemble_and_grade(b, all_scores)
            rost['Grade'] = grd
            rost['Roster Audit'] = a1
            rost['Grade Audit'] = a2
            final_lineups.append(rost)
        
        result_df = pd.DataFrame(final_lineups).sort_values('Grade', ascending=True)
        st.subheader("📊 Batched Performance Report")
        st.table(result_df[['Grade', 'Roster Audit', 'Grade Audit', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']].head(10))
        st.download_button("📥 Download Batch CSV", result_df.to_csv(index=False), "Vantage_NBA_Grades.csv")
