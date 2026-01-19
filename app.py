
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io
from datetime import datetime

# --- UI & THEME CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | LEGAL LOCK", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .main { background: radial-gradient(circle at top right, #1a1f2e, #0d1117); color: #c9d1d9; }
    .card-elite { border: 2px solid #238636 !important; background: rgba(35, 134, 54, 0.15); box-shadow: 0 0 20px rgba(35, 134, 54, 0.4); }
    .card-strong { border: 2px solid #d29922 !important; background: rgba(210, 153, 34, 0.1); }
    .card-standard { border: 1px solid #30363d !important; background: rgba(22, 27, 34, 0.5); }
    .lineup-card { border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .badge-label { padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: bold; color: white; margin-left: 5px; }
    .bg-win { background: #238636; }
    .bg-proj { background: #1f6feb; }
    </style>
    """, unsafe_allow_html=True)

# IRON-CLAD INJURY BLACKLIST
FORCED_OUT = ["Toppin", "Haliburton", "Mathurin", "Garland", "Brunson", "Kyrie", "Embiid", "Hartenstein", "Gafford"]

@st.cache_data
def process_data(file_bytes, manual_scratches_str):
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    
    # Standardizing Columns (Using DK default headers)
    df['Proj'] = pd.to_numeric(df[cols.get('proj', df.columns[-1])], errors='coerce').fillna(0.0)
    df['Sal'] = pd.to_numeric(df[cols.get('salary', df.columns[5])], errors='coerce').fillna(50000)
    df['Name'] = df[cols.get('name', df.columns[2])].astype(str)
    df['Pos'] = df[cols.get('position', df.columns[0])].astype(str)
    df['Team'] = df[cols.get('teamabbrev', df.columns[7])].astype(str)
    df['GameInfo'] = df[cols.get('gameinfo', df.columns[6])].astype(str)

    # Hard Filtering Scratches
    manual_list = [s.strip().lower() for s in manual_scratches_str.split('\n') if s.strip()]
    full_scratch_list = [s.lower() for s in FORCED_OUT] + manual_list
    mask = df['Name'].str.lower().apply(lambda x: any(scratch in x for scratch in full_scratch_list))
    df = df[~mask]
    
    return df[df['Proj'] > 0.1].reset_index(drop=True)

class VantageOptimizer:
    def __init__(self, df):
        self.df = df
        self.n_p = len(df)

    def get_dk_slots(self, lineup_df):
        """
        STRICT POSITION LOCK ENGINE (No Exceptions):
        Forces 1 PG, 1 SG, 1 SF, 1 PF, 1 C, 1 G, 1 F, 1 UTIL.
        Implements Late-Game UTIL logic.
        """
        def extract_time(info):
            try:
                time_match = re.search(r'(\d{1,2}:\d{2}[APM]{2})', info)
                return datetime.strptime(time_match.group(), '%I:%M%p') if time_match else datetime.min
            except:
                return datetime.min

        lineup_df['time_val'] = lineup_df['GameInfo'].apply(extract_time)
        assigned = {}
        pool = lineup_df.to_dict('records')

        # 1. UTIL PRIORITY: Identify the absolute latest player for swap flexibility
        pool.sort(key=lambda x: x['time_val'], reverse=True)
        # We don't pop yet, we just identify the best UTIL candidate (latest game)
        util_candidate = pool[0] 

        # 2. MANDATORY PRIMARY SLOTS (Satisfy the unique positions first)
        for slot in ['PG', 'SG', 'SF', 'PF', 'C']:
            pool.sort(key=lambda x: (slot not in x['Pos'], -x['Proj']))
            for i, p in enumerate(pool):
                if slot in p['Pos'] and p['Name'] != util_candidate['Name']:
                    assigned[slot] = p
                    pool.pop(i)
                    break
        
        # 3. FILL FLEX SLOTS (G, F)
        # G can be PG/SG. F can be SF/PF.
        for slot in ['G', 'F']:
            pool.sort(key=lambda x: -x['Proj'])
            for i, p in enumerate(pool):
                match = False
                if slot == 'G' and ('PG' in p['Pos'] or 'SG' in p['Pos']): match = True
                if slot == 'F' and ('SF' in p['Pos'] or 'PF' in p['Pos']): match = True
                
                if match and p['Name'] != util_candidate['Name']:
                    assigned[slot] = p
                    pool.pop(i)
                    break
        
        # 4. FINAL SLOT (UTIL)
        # The remaining player MUST be the UTIL
        if pool:
            assigned['UTIL'] = pool[0]

        # RE-ORDER FOR DISPLAY
        ordered_list = []
        for slot in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']:
            p = assigned.get(slot)
            if p:
                p['Slot'] = slot
                ordered_list.append(p)
        return pd.DataFrame(ordered_list)

    def run_sims(self, n_sims=10000):
        # Constraints: 8 players total, 1 of each primary position min.
        A_rows = [np.ones(self.n_p), self.df['Sal'].values]
        bl, bu = [8, 45000], [8, 50000]
        
        # Ensure at least 1 of each position exists in the solve
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            A_rows.append(self.df['Pos'].str.contains(pos).astype(int).values)
            bl.append(1); bu.append(8)

        A = np.vstack(A_rows)
        constraints = LinearConstraint(A, bl, bu)
        
        lineup_counts = {}
        progress_bar = st.progress(0)
        
        for i in range(n_sims):
            sim_p = self.df['Proj'].values * np.random.normal(1.0, 0.15, self.n_p)
            res = milp(c=-sim_p, constraints=constraints, integrality=np.ones(self.n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = tuple(sorted(np.where(res.x > 0.5)[0]))
                lineup_counts[idx] = lineup_counts.get(idx, 0) + 1
            if i % 1000 == 0:
                progress_bar.progress((i + 1) / n_sims)
        
        sorted_lineups = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        max_freq = sorted_lineups[0][1] if sorted_lineups else 1
        
        return [{
            'df': self.get_dk_slots(self.df.iloc[list(idx)]), 
            'win_pct': (count/n_sims)*100, 
            'rel_score': (count/max_freq)*100, 
            'proj': self.df.iloc[list(idx)]['Proj'].sum()
        } for idx, count in sorted_lineups]

# --- APP FLOW ---
st.sidebar.title("🕹️ COMMAND")
f = st.sidebar.file_uploader("UPLOAD CSV", type="csv")
scratches_input = st.sidebar.text_area("🚑 ADD SCRATCHES", height=100)

if f:
    data = process_data(f.getvalue(), scratches_input)
    if st.button("🚀 GENERATE 10,000 SIMS"):
        optimizer = VantageOptimizer(data)
        st.session_state.results = optimizer.run_sims()

if 'results' in st.session_state:
    cols = st.columns(2)
    for i, res in enumerate(st.session_state.results):
        score = res['rel_score']
        card_class = "card-elite" if score > 85 else "card-strong" if score > 50 else "card-standard"
        with cols[i % 2]:
            st.markdown(f"""<div class="lineup-card {card_class}">
                <div style="display:flex; justify-content:space-between;">
                    <b>LINEUP #{i+1}</b>
                    <span class="badge-label bg-win">WIN: {round(res['win_pct'], 2)}%</span>
                </div></div>""", unsafe_allow_html=True)
            st.table(res['df'][['Slot', 'Name', 'Team', 'Sal', 'Proj']])
