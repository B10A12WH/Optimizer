import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re

# --- VANTAGE 99: TOURNAMENT GRADE REBUILD ---
st.set_page_config(page_title="VANTAGE 99", layout="wide", page_icon="🧪")

def deep_scan_headers(df):
    """Finds the actual header row even if there is trash at the top."""
    for i, row in df.head(15).iterrows():
        row_vals = [str(v).lower() for v in row.values]
        if any('name' in v for v in row_vals) and any('salary' in v for v in row_vals):
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

def normalize(name, is_dst=False):
    """Molecular normalization for perfect file merging."""
    name = str(name).lower().strip()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    if is_dst:
        if '49ers' in name or 'ers' in name: return '49ers'
        return name.split()[-1]
    for s in [' jr', ' sr', ' iii', ' ii', ' iv']:
        if name.endswith(s): name = name[:-len(s)]
    return name.strip()

class VantageAlpha:
    def __init__(self, p_df, s_df):
        # 1. PARSE & CLEAN
        p_df, s_df = deep_scan_headers(p_df), deep_scan_headers(s_df)
        p_df.columns = [str(c).strip() for c in p_df.columns]
        s_df.columns = [str(c).strip() for c in s_df.columns]
        
        # 2. THE MERGE REACTION
        p_df['norm'] = p_df.apply(lambda x: normalize(x['Name'], x['Position'] == 'DST'), axis=1)
        s_df['norm'] = s_df.apply(lambda x: normalize(x['Name'], x['Position'] == 'DST'), axis=1)
        
        s_name_key = next((c for c in s_df.columns if 'name' in c.lower()), 'Name')
        s_sal_key = next((c for c in s_df.columns if 'salary' in c.lower()), 'Salary')
        s_id_key = next((c for c in s_df.columns if 'id' in c.lower() and 'name' not in c.lower()), 'ID')

        self.df = pd.merge(p_df, s_df[['norm', s_sal_key, s_id_key]], on='norm', how='inner')
        self.df = self.df.rename(columns={'Position': 'Pos', 'ProjPts': 'Proj', 'ProjOwn': 'Own', s_sal_key: 'Sal', s_id_key: 'ID'})
        
        # 3. SCENARIO CATALYSTS (Multipliers)
        # Apply the specific 2026 Weekend Multipliers
        boosts = {'jaxon smithnjigba': 1.35, 'jauan jennings': 1.40, 'kenneth walker': 1.30, 'khalil shakir': 1.25, 'rj harvey': 1.15, 'josh allen': 0.88}
        for n, m in boosts.items():
            self.df.loc[self.df['norm'] == n, 'Proj'] *= m
            
        # 4. BLACKLIST
        self.df = self.df[~self.df['norm'].isin(['dk metcalf', 'gabe davis', 'george kittle', 'fred warner'])]
        self.df = self.df[self.df['Pos'] != 'K'] # Force exclude kickers

    def run_batch(self, n=20):
        pool = []
        n_players = len(self.df)
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 155.0, 48200, 0.35, 3

        for _ in range(n):
            sim_proj = np.random.normal(self.df['Proj'], self.df['Proj'] * JITTER)
            A, b_l, b_u = [], [], []
            A.append(np.ones(n_players)); b_l.append(9); b_u.append(9) # Total 9
            A.append(self.df['Sal'].values); b_l.append(SAL_FLOOR); b_u.append(50000) # Salary
            A.append(self.df['Own'].values); b_l.append(0); b_u.append(OWN_CAP) # Own
            
            for p, (mn, mx) in [('QB',(1,1)),('RB',(2,3)),('WR',(3,4)),('TE',(1,2)),('DST',(1,1))]:
                mask = (self.df['Pos'] == p).astype(int).values
                A.append(mask); b_l.append(mn); b_u.append(mx)
            
            for prev in pool:
                m = np.zeros(n_players); m[prev] = 1
                A.append(m); b_l.append(0); b_u.append(9 - MIN_UNIQUE)
            
            res = milp(c=-sim_proj, constraints=LinearConstraint(A, b_l, b_u), integrality=np.ones(n_players), bounds=Bounds(0, 1))
            if res.success: pool.append(np.where(res.x > 0.5)[0])
        return [self.df.iloc[p] for p in pool]

# --- STREAMLIT FRONT END ---
st.title("🧪 VANTAGE 99")
col1, col2 = st.columns(2)
f_proj = col1.file_uploader("1. Projections CSV", type="csv")
f_sal = col2.file_uploader("2. DraftKings Salary CSV", type="csv")

if f_proj and f_sal:
    engine = VantageAlpha(pd.read_csv(f_proj), pd.read_csv(f_sal))
    if st.button("🚀 ASSEMBLE 20 LINEUPS"):
        lineups = engine.run_batch(20)
        final_rows = []
        for l in lineups:
            q = l[l['Pos']=='QB'].iloc[0]
            r = l[l['Pos']=='RB'].sort_values('Sal', ascending=False)
            w = l[l['Pos']=='WR'].sort_values('Sal', ascending=False)
            t = l[l['Pos']=='TE'].sort_values('Sal', ascending=False)
            d = l[l['Pos']=='DST'].iloc[0]
            
            def fmt(p): return f"{p['Name']} ({int(p['ID'])})"
            
            # EXACT ASSEMBLY: QB,RB,RB,WR,WR,WR,TE,FLEX,DST
            flex = r.iloc[2] if len(r)>2 else (w.iloc[3] if len(w)>3 else t.iloc[1])
            final_rows.append([fmt(q), fmt(r.iloc[0]), fmt(r.iloc[1]), fmt(w.iloc[0]), fmt(w.iloc[1]), fmt(w.iloc[2]), fmt(t.iloc[0]), fmt(flex), fmt(d)])
        
        out_df = pd.DataFrame(final_rows, columns=['QB','RB','RB','WR','WR','WR','TE','FLEX','DST'])
        st.table(out_df)
        st.download_button("📥 Download Footballers-Style CSV", out_df.to_csv(index=False), "Vantage99_Batch.csv")
