import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
import io

# --- VANTAGE 99: TOURNAMENT GRADE ASSEMBLY ---
st.set_page_config(page_title="VANTAGE 99", layout="wide", page_icon="🧪")

def deep_scan_headers(df):
    """Bypasses DraftKings' metadata rows to find actual column headers."""
    for i, row in df.head(15).iterrows():
        row_vals = [str(v).lower() for v in row.values]
        if any('name' in v for v in row_vals) and any('salary' in v for v in row_vals):
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

def normalize(name, is_dst=False):
    """Molecular cleaning for perfect file merging."""
    name = str(name).lower().strip()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    if is_dst:
        if '49ers' in name or 'ers' in name: return '49ers'
        return name.split()[-1]
    for s in [' jr', ' sr', ' iii', ' ii', ' iv']:
        if name.endswith(s): name = name[:-len(s)]
    return name.strip()

class VantageOptimizer:
    def __init__(self, p_df, s_df):
        # 1. PARSE & PURIFY
        p_df, s_df = deep_scan_headers(p_df), deep_scan_headers(s_df)
        p_df.columns = [str(c).strip() for c in p_df.columns]
        s_df.columns = [str(c).strip() for c in s_df.columns]
        
        # 2. MATCHING REACTION
        p_df['norm'] = p_df.apply(lambda x: normalize(x['Name'], x['Position'] == 'DST'), axis=1)
        s_df['norm'] = s_df.apply(lambda x: normalize(x['Name'], x['Position'] == 'DST'), axis=1)
        
        # Locate IDs and Salaries
        s_sal_key = next((c for c in s_df.columns if 'salary' in c.lower()), 'Salary')
        s_id_key = next((c for c in s_df.columns if 'id' in c.lower() and 'name' not in c.lower()), 'ID')

        self.df = pd.merge(p_df, s_df[['norm', s_sal_key, s_id_key]], on='norm', how='inner')
        self.df = self.df.rename(columns={'Position': 'Pos', 'ProjPts': 'Proj', 'ProjOwn': 'Own', s_sal_key: 'Sal', s_id_key: 'ID'})
        
        # 3. SCENARIO CATALYST (Multipliers)
        boosts = {'jaxon smithnjigba': 1.35, 'jauan jennings': 1.40, 'kenneth walker': 1.30, 'khalil shakir': 1.25, 'rj harvey': 1.15, 'josh allen': 0.88}
        for n, m in boosts.items():
            self.df.loc[self.df['norm'] == n, 'Proj'] *= m
            
        # 4. BLACKLIST
        self.df = self.df[~self.df['norm'].isin(['dk metcalf', 'gabe davis', 'george kittle', 'fred warner'])]
        self.df = self.df[self.df['Pos'] != 'K'] # Force exclude kickers

    def cook(self, n=20):
        pool = []
        player_indices = self.df.index
        # Slate logic: $48,200 floor | 155% Ownership Cap | 3 unique players
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 155.0, 48200, 0.35, 3

        for i in range(n):
            sim_df = self.df.copy()
            sim_df['sim_proj'] = np.random.normal(sim_df['Proj'], sim_df['Proj'] * JITTER)
            
            prob = pulp.LpProblem(f"Lineup_{i}", pulp.LpMaximize)
            choices = pulp.LpVariable.dicts("P", player_indices, cat='Binary')

            # Goal: Maximize Points
            prob += pulp.lpSum([sim_df.loc[idx, 'sim_proj'] * choices[idx] for idx in player_indices])
            
            # Constraints
            prob += pulp.lpSum([choices[idx] for idx in player_indices]) == 9
            prob += pulp.lpSum([sim_df.loc[idx, 'Sal'] * choices[idx] for idx in player_indices]) <= 50000
            prob += pulp.lpSum([sim_df.loc[idx, 'Sal'] * choices[idx] for idx in player_indices]) >= SAL_FLOOR
            prob += pulp.lpSum([sim_df.loc[idx, 'Own'] * choices[idx] for idx in player_indices]) <= OWN_CAP

            # Positional Logic
            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                mask = [choices[idx] for idx in player_indices if sim_df.loc[idx, 'Pos'] == p]
                prob += pulp.lpSum(mask) >= mn
                prob += pulp.lpSum(mask) <= mx

            # Portfolio Uniqueness
            for prev in pool:
                prob += pulp.lpSum([choices[idx] for idx in prev.index]) <= (9 - MIN_UNIQUE)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                pool.append(sim_df.loc[[idx for idx in player_indices if choices[idx].varValue == 1]])
        return pool

# --- UI LOGIC ---
st.title("🧪 VANTAGE 99: ASSEMBLED")
f1, f2 = st.file_uploader("1. Projections", type="csv"), st.file_uploader("2. DraftKings Salaries", type="csv")

if f1 and f2:
    try:
        engine = VantageOptimizer(pd.read_csv(f1), pd.read_csv(f2))
        if st.button("🚀 ASSEMBLE BATCH"):
            lineups = engine.cook(20)
            rows = []
            for l in lineups:
                q = l[l['Pos']=='QB'].iloc[0]
                r = l[l['Pos']=='RB'].sort_values('Sal', ascending=False)
                w = l[l['Pos']=='WR'].sort_values('Sal', ascending=False)
                t = l[l['Pos']=='TE'].sort_values('Sal', ascending=False)
                d = l[l['Pos']=='DST'].iloc[0]
                
                def fmt(p): return f"{p['Name']} ({int(p['ID'])})"
                
                # Assembly: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
                flex = r.iloc[2] if len(r)>2 else (w.iloc[3] if len(w)>3 else t.iloc[1])
                rows.append([fmt(q), fmt(r.iloc[0]), fmt(r.iloc[1]), fmt(w.iloc[0]), fmt(w.iloc[1]), fmt(w.iloc[2]), fmt(t.iloc[0]), fmt(flex), fmt(d)])
            
            # The Final Product
            out_df = pd.DataFrame(rows, columns=['QB','RB','RB','WR','WR','WR','TE','FLEX','DST'])
            
            # DISPLAY FIX: Rename columns temporarily for the UI to avoid the Arrow error
            display_df = out_df.copy()
            unique_cols = []
            counts = {}
            for col in display_df.columns:
                counts[col] = counts.get(col, 0) + 1
                unique_cols.append(f"{col} ({counts[col]})" if counts[col] > 1 else col)
            display_df.columns = unique_cols

            st.subheader("📋 Lineup Preview")
            st.dataframe(display_df)
            
            # Download button uses original out_df for DK compatibility
            csv = out_df.to_csv(index=False)
            st.download_button("📥 Download Uploadable CSV", csv, "Vantage99_Batch.csv", "text/csv")
    except Exception as e:
        st.error(f"Contamination Detected: {e}")
