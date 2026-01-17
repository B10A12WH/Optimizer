import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re

# --- VANTAGE 99: FOOTBALLERS EDITION ---
st.set_page_config(page_title="VANTAGE 99", layout="wide", page_icon="🧪")

def deep_scan(df):
    """Finds the actual header row even if there is trash at the top of the CSV."""
    for i, row in df.head(15).iterrows():
        row_vals = [str(v).lower() for v in row.values]
        if any('name' in v for v in row_vals) and any('salary' in v for v in row_vals):
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

def normalize(name, is_dst=False):
    """Standardizes names for perfect merging between files."""
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
        # 1. CLEAN DATA
        p_df, s_df = deep_scan(p_df), deep_scan(s_df)
        p_df.columns = [str(c).strip() for c in p_df.columns]
        s_df.columns = [str(c).strip() for c in s_df.columns]
        
        # 2. MOLECULAR MERGE
        p_df['norm'] = p_df.apply(lambda x: normalize(x['Name'], x['Position'] == 'DST'), axis=1)
        s_df['norm'] = s_df.apply(lambda x: normalize(x['Name'], x['Position'] == 'DST'), axis=1)
        
        # Find ID and Salary columns
        s_name_key = next((c for c in s_df.columns if 'name' in c.lower()), 'Name')
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

    def cook(self, n_lineups=20):
        final_pool = []
        player_indices = self.df.index
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 155.0, 48200, 0.35, 3

        for n in range(n_lineups):
            sim_df = self.df.copy()
            sim_df['sim_proj'] = np.random.normal(sim_df['Proj'], sim_df['Proj'] * JITTER)
            
            prob = pulp.LpProblem(f"Batch_{n}", pulp.LpMaximize)
            choices = pulp.LpVariable.dicts("P", player_indices, cat='Binary')

            prob += pulp.lpSum([sim_df.loc[i, 'sim_proj'] * choices[i] for i in player_indices])
            prob += pulp.lpSum([choices[i] for i in player_indices]) == 9
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i] for i in player_indices]) >= SAL_FLOOR
            prob += pulp.lpSum([sim_df.loc[i, 'Own'] * choices[i] for i in player_indices]) <= OWN_CAP

            for p, mn, mx in [('QB',1,1),('RB',2,3),('WR',3,4),('TE',1,2),('DST',1,1)]:
                mask = [choices[i] for i in player_indices if sim_df.loc[i, 'Pos'] == p]
                prob += pulp.lpSum(mask) >= mn
                prob += pulp.lpSum(mask) <= mx

            for prev in final_pool:
                prob += pulp.lpSum([choices[i] for i in prev.index]) <= (9 - MIN_UNIQUE)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                final_pool.append(sim_df.loc[[i for i in player_indices if choices[i].varValue == 1]])
        return final_pool

# --- UI ---
st.title("🧪 VANTAGE 99")
f_proj = st.file_uploader("1. Projections CSV", type="csv")
f_sal = st.file_uploader("2. DraftKings Salary CSV", type="csv")

if f_proj and f_sal:
    engine = VantageAlpha(pd.read_csv(f_proj), pd.read_csv(f_sal))
    if st.button("🚀 ASSEMBLE 20 LINEUPS"):
        lineups = engine.cook(20)
        rows = []
        for l in lineups:
            q = l[l['Pos']=='QB'].iloc[0]
            r = l[l['Pos']=='RB'].sort_values('Sal', ascending=False)
            w = l[l['Pos']=='WR'].sort_values('Sal', ascending=False)
            t = l[l['Pos']=='TE'].sort_values('Sal', ascending=False)
            d = l[l['Pos']=='DST'].iloc[0]
            
            def fmt(p): return f"{p['Name']} ({int(p['ID'])})"
            
            # Footballers Assembly: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
            rb1, rb2 = r.iloc[0], r.iloc[1]
            wr1, wr2, wr3 = w.iloc[0], w.iloc[1], w.iloc[2]
            te = t.iloc[0]
            flex = r.iloc[2] if len(r)>2 else (w.iloc[3] if len(w)>3 else t.iloc[1])
            rows.append([fmt(q), fmt(rb1), fmt(rb2), fmt(wr1), fmt(wr2), fmt(wr3), fmt(te), fmt(flex), fmt(d)])
        
        out_df = pd.DataFrame(rows, columns=['QB','RB','RB','WR','WR','WR','TE','FLEX','DST'])
        st.table(out_df)
        st.download_button("📥 Download Footballers-Style CSV", out_df.to_csv(index=False), "Vantage99_Batch.csv")
