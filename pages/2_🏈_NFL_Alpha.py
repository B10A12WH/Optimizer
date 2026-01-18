import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v67.0", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stButton>button { width: 100%; border-radius: 4px; border: 1px solid #30363d; background: #161b22; color: #c9d1d9; text-align: left; }
    .stButton>button:hover { border-color: #00ffcc; color: #00ffcc; }
    .audit-val { font-family: 'JetBrains Mono'; font-weight: bold; color: #00ffcc; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

class EliteOptimizerV67:
    def __init__(self, df):
        # Normalize headers to avoid KeyErrors
        cols = {c.lower().replace(" ", ""): c for c in df.columns}
        self.df = df.copy()
        
        # Robust column detection
        p_key = next((cols[k] for k in cols if k in ['proj', 'points', 'avgpointspergame']), df.columns[0])
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[cols.get('salary', 'Salary')]).fillna(50000)
        self.df['Pos'] = df[cols.get('position', 'Position')].astype(str)
        self.df['Team'] = df[cols.get('teamabbrev', 'TeamAbbrev')].astype(str)
        self.df['ID'] = df[cols.get('id', 'ID')].astype(str)
        
        # Vegas Data (Divisional Round 1/18)
        vegas_boosts = {'HOU': 0.88, 'NE': 1.08, 'LAR': 1.15, 'CHI': 1.04}
        self.df['Proj'] = self.df.apply(lambda x: x['Proj'] * vegas_boosts.get(x['Team'], 1.0), axis=1)
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        for p in ['QB','RB','WR','TE','DST']: 
            self.df[f'is_{p}'] = (self.df['Pos'] == p).astype(int)
            
        # Hard Scrub: confirmed OUTs
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def assemble(self, n=20, exp=0.45):
        n_p = len(self.df)
        raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64)
        late_mask = self.df['is_late'].values.astype(np.float64)
        is_te = self.df['is_TE'].values
        
        portfolio, counts = [], {name: 0 for name in self.df['Name']}
        
        for i in range(n):
            # FIXED: Shape Mismatch Guard
            # .flatten() ensures we are dealing with a 1D array of floats
            adj_p = (raw_p + (late_mask * 0.15) - (is_te * 0.85)).flatten()
            
            # Explicitly align mean (loc) and std (scale) shapes
            sim_p = np.random.normal(loc=adj_p, scale=np.abs(adj_p * 0.22), size=adj_p.shape).clip(min=0)
            
            A, bl, bu = [], [], []
            A.append(np.ones(n_p)); bl.append(9); bu.append(9) # 9 Players
            A.append(sals); bl.append(49000); bu.append(50000) # Salary Cap
            
            # DK Classic Constraints
            A.append(self.df['is_QB'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_RB'].values); bl.append(2); bu.append(3)
            A.append(self.df['is_WR'].values); bl.append(3); bu.append(4)
            A.append(self.df['is_TE'].values); bl.append(1); bu.append(1)
            A.append(self.df['is_DST'].values); bl.append(1); bu.append(1)

            # Exposure Governor
            for idx, name in enumerate(self.df['Name']):
                if counts[name] >= (n * exp):
                    m = np.zeros(n_p); m[idx] = 1; A.append(m); bl.append(0); bu.append(0)

            # Solve using MILP
            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), 
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            
            if res.success:
                portfolio.append(self.df.iloc[np.where(res.x > 0.5)[0]].copy())
                for name in portfolio[-1]['Name']: counts[name] += 1
        return portfolio

# --- MAIN RENDERING ---
st.title("🧬 VANTAGE 99 | v67.0 COMMAND")

f = st.file_uploader("LOAD DK DATASET", type="csv")
if f:
    df_raw = pd.read_csv(f)
    if "Field" in str(df_raw.columns): df_raw = pd.read_csv(f, skiprows=7)
    engine = EliteOptimizerV67(df_raw)
    
    if st.button("🚀 INITIATE ASSEMBLY"):
        st.session_state.portfolio = engine.assemble(n=20)
        st.session_state.sel_idx = 0

    if 'portfolio' in st.session_state:
        col_list, col_scout = st.columns([1, 2.2])
        with col_list:
            for i, l in enumerate(st.session_state.portfolio):
                if st.button(f"L{i+1} | {round(l['Proj'].sum(), 1)} PTS", key=f"btn_{i}"):
                    st.session_state.sel_idx = i

        with col_scout:
            l = st.session_state.portfolio[st.session_state.sel_idx]
            # DraftKings Standard Sorting Logic
            qb = l[l['is_QB']==1].iloc[0]
            dst = l[l['is_DST']==1].iloc[0]
            te = l[l['is_TE']==1].iloc[0]
            rbs = l[l['is_RB']==1].sort_values('Sal', ascending=False)
            wrs = l[l['is_WR']==1].sort_values('Sal', ascending=False)
            
            # Isolate FLEX
            core_ids = [qb['ID'], rbs.iloc[0]['ID'], rbs.iloc[1]['ID'], wrs.iloc[0]['ID'], wrs.iloc[1]['ID'], wrs.iloc[2]['ID'], te['ID'], dst['ID']]
            flex_row = l[~l['ID'].isin(core_ids)]
            flex = flex_row.iloc[0] if not flex_row.empty else None
            
            display_order = [
                ("QB", qb), ("RB", rbs.iloc[0]), ("RB", rbs.iloc[1]),
                ("WR", wrs.iloc[0]), ("WR", wrs.iloc[1]), ("WR", wrs.iloc[2]),
                ("TE", te), ("FLEX", flex), ("DST", dst)
            ]
            
            st.table(pd.DataFrame([{"Pos": lbl, "Name": p['Name'], "Team": p['Team'], "Salary": f"${int(p['Sal'])}", "Proj": round(p['Proj'], 2)} for lbl, p in display_order]))
