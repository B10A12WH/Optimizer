import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io
from datetime import datetime
import concurrent.futures
import os

# --- UI & THEME CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | NBA OPTIMIZER", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    .main { background: radial-gradient(circle at top right, #1a1f2e, #0d1117); color: #c9d1d9; }
    .card-elite { border: 1px solid #238636; background: rgba(35, 134, 54, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .card-strong { border: 1px solid #d29922; background: rgba(210, 153, 34, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .card-standard { border: 1px solid #30363d; background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; color: #8b949e; font-size: 12px; border-bottom: 1px solid #30363d; padding-bottom: 5px; text-transform: uppercase; }
    td { padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 14px; }
    .pos { color: #58a6ff; font-weight: bold; width: 50px; }
    .name { color: #e6edf3; font-weight: 600; }
    .meta { color: #8b949e; font-size: 12px; }
    .sal { color: #7ee787; font-family: monospace; text-align: right; }
    .proj { color: #c9d1d9; font-weight: bold; text-align: right; }
    .ceil { color: #d29922; font-weight: bold; text-align: right; font-family: monospace; }
    .badge { background: #238636; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
    .hammer-badge { background: #8250df; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; margin-left: 5px; }
    </style>
    """, unsafe_allow_html=True)

FORCED_OUT = ["Kuminga", "Toppin", "Haliburton", "Mathurin", "Garland", "Brunson", "Kyrie", "Embiid", "Hartenstein", "Gafford"]

@st.cache_data
def process_data(file_bytes, manual_scratches_str):
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    df['Proj'] = pd.to_numeric(df[cols.get('proj', df.columns[-1])], errors='coerce').fillna(0.0)
    df['Sal'] = pd.to_numeric(df[cols.get('salary', df.columns[5])], errors='coerce').fillna(50000)
    df['Name'] = df[cols.get('name', df.columns[2])].astype(str).str.strip()
    df['Pos'] = df[cols.get('position', df.columns[0])].astype(str).str.strip()
    df['Team'] = df[cols.get('teamabbrev', df.columns[7])].astype(str)
    df['GameInfo'] = df[cols.get('gameinfo', df.columns[6])].astype(str)
    manual_list = [s.strip().lower() for s in manual_scratches_str.split('\n') if s.strip()]
    full_scratch_list = [s.lower() for s in FORCED_OUT] + manual_list
    mask = df['Name'].str.lower().apply(lambda x: any(scratch in x for scratch in full_scratch_list))
    df = df[~mask]
    df = df[~df['Name'].str.contains("Toppin|Kuminga", case=False)]
    return df[df['Proj'] > 0.1].reset_index(drop=True)

def solve_batch_task(args):
    (indices, proj_matrix, A_matrix, bl, bu, n_p) = args
    constraints = LinearConstraint(A_matrix, bl, bu)
    batch_counts = {}
    for i, _ in enumerate(indices):
        res = milp(c=-proj_matrix[i], constraints=constraints, integrality=np.ones(n_p), bounds=Bounds(0, 1))
        if res.success:
            lineup_idx = tuple(sorted(np.where(res.x > 0.5)[0]))
            batch_counts[lineup_idx] = batch_counts.get(lineup_idx, 0) + 1
    return batch_counts

class VantageOptimizer:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_p = len(df)

    def get_dk_slots(self, lineup_df):
        """
        ADVANCED DEPTH-FIRST SEARCH SLOTTING:
        Ensures players are assigned to valid DK positions (PG, SG, SF, PF, C, G, F, UTIL).
        """
        players = lineup_df.to_dict('records')
        slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        def fits(p, s):
            pos = p['Pos']
            if s == 'UTIL': return True
            if s == 'G': return 'PG' in pos or 'SG' in pos
            if s == 'F': return 'SF' in pos or 'PF' in pos
            return s in pos

        # We will try to fill the slots one by one using recursion
        solution = [None] * 8

        def dfs(slot_idx, available_mask):
            if slot_idx == 8:
                return True
            
            for p_idx in range(8):
                if not (available_mask & (1 << p_idx)):
                    if fits(players[p_idx], slots[slot_idx]):
                        solution[slot_idx] = players[p_idx].copy()
                        solution[slot_idx]['Slot'] = slots[slot_idx]
                        if dfs(slot_idx + 1, available_mask | (1 << p_idx)):
                            return True
            return False

        if dfs(0, 0):
            return pd.DataFrame(solution)
        
        # Absolute Emergency Fallback
        lineup_df = lineup_df.copy()
        lineup_df['Slot'] = 'UTIL'
        return lineup_df

    def run_sims(self, n_sims=8000):
        np.random.seed(42)
        def parse_time(info):
            try:
                m = re.search(r'(\d{1,2}:\d{2}[APM]{2})', info)
                return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
            except: return datetime.min

        self.df['time_obj'] = self.df['GameInfo'].apply(parse_time)
        latest_time = self.df['time_obj'].max()
        is_hammer = (self.df['time_obj'] == latest_time).astype(int)
        
        unique_teams = self.df['Team'].unique()
        team_map = {t: i for i, t in enumerate(unique_teams)}
        player_team_indices = self.df['Team'].map(team_map).values
        
        vols = np.where(self.df['Sal'] < 4000, 0.25, np.where(self.df['Sal'] < 8000, 0.20, 0.15))
        self.df['Ceil'] = self.df['Proj'] + (self.df['Proj'] * vols * 2.33)

        A_rows = [np.ones(self.n_p), self.df['Sal'].values]
        bl, bu = [8, 45000], [8, 50000]
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            A_rows.append(self.df['Pos'].str.contains(p).astype(int).values); bl.append(1); bu.append(8)
        
        A_rows.append(self.df['Pos'].apply(lambda x: 'PG' in x or 'SG' in x).astype(int).values); bl.append(3); bu.append(8)
        A_rows.append(self.df['Pos'].apply(lambda x: 'SF' in x or 'PF' in x).astype(int).values); bl.append(3); bu.append(8)
        A_rows.append(self.df['Pos'].apply(lambda x: 'C' in x and not ('SF' in x or 'PF' in x)).astype(int).values); bl.append(0); bu.append(2)
        if is_hammer.sum() > 0:
            A_rows.append(is_hammer.values); bl.append(1); bu.append(8)

        A_matrix = np.vstack(A_rows)
        team_noise = np.random.normal(1.0, 0.15, (n_sims, len(unique_teams)))
        player_noise = 1.0 + (np.random.normal(0, 1, (n_sims, self.n_p)) * vols)
        
        all_projs = np.zeros((n_sims, self.n_p))
        for i in range(n_sims):
            all_projs[i] = self.df['Proj'].values * ((player_noise[i] * 0.6) + (team_noise[i][player_team_indices] * 0.4))

        n_workers = os.cpu_count() or 4
        chunk = n_sims // n_workers
        tasks = [(range(i*chunk, (i+1)*chunk if i<n_workers-1 else n_sims), all_projs[i*chunk:(i+1)*chunk if i<n_workers-1 else n_sims], A_matrix, bl, bu, self.n_p) for i in range(n_workers)]
        
        lineup_counts = {}
        progress = st.progress(0)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exec:
            for i, res in enumerate(exec.map(solve_batch_task, tasks)):
                for idx, c in res.items(): lineup_counts[idx] = lineup_counts.get(idx, 0) + c
                progress.progress((i + 1) / n_workers)

        sorted_lineups = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        max_f = sorted_lineups[0][1] if sorted_lineups else 1
        return [{'df': self.get_dk_slots(self.df.iloc[list(idx)]), 'rel_score': int((c/max_f)*100), 'proj': self.df.iloc[list(idx)]['Proj'].sum(), 'ceil': self.df.iloc[list(idx)]['Ceil'].sum(), 'hammer_count': self.df.iloc[list(idx)]['time_obj'].apply(lambda t: t == latest_time).sum()} for idx, c in sorted_lineups]

# --- UI FLOW ---
st.sidebar.title("🕹️ COMMAND")
f = st.sidebar.file_uploader("UPLOAD CSV", type="csv")
scratches = st.sidebar.text_area("🚑 ADD SCRATCHES", height=100)

if f:
    data = process_data(f.getvalue(), scratches)
    if st.button("🚀 GENERATE SIMS"):
        st.session_state.results = VantageOptimizer(data).run_sims()

if 'results' in st.session_state:
    cols = st.columns(2)
    for i, res in enumerate(st.session_state.results):
        card = "card-elite" if res['rel_score'] >= 90 else "card-strong" if res['rel_score'] >= 75 else "card-standard"
        rows = "".join([f"<tr><td class='pos'>{r['Slot']}</td><td><span class='name'>{r['Name']}</span> <span class='meta'>({r['Team']})</span></td><td class='sal'>${int(r['Sal'])}</td><td class='proj'>{round(r['Proj'],1)}</td><td class='ceil'>{round(r['Ceil'],1)}</td></tr>" for _, r in res['df'].iterrows()])
        with cols[i % 2]:
            st.markdown(f"<div class='{card}'><div style='display:flex; justify-content:space-between; margin-bottom:10px;'><div><b>LINEUP #{i+1}</b>{'<span class="hammer-badge">🔨 HAMMER</span>' if res['hammer_count']>0 else ''}</div><div><span class='badge'>SCORE: {res['rel_score']}</span> <span class='badge' style='background:#d29922;'>CEIL: {round(res['ceil'],1)}</span></div></div><table><thead><tr><th>POS</th><th>PLAYER</th><th>SAL</th><th>PROJ</th><th>99%</th></tr></thead><tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)
