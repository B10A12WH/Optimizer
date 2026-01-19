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
st.set_page_config(page_title="VANTAGE 99 | GPP DIVERSIFIER", layout="wide", page_icon="🏀")

st.markdown("""
    <style>
    .main { background: radial-gradient(circle at top right, #1a1f2e, #0d1117); color: #c9d1d9; }
    .card-elite { border: 1px solid #238636; background: rgba(35, 134, 54, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .card-standard { border: 1px solid #30363d; background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; color: #8b949e; font-size: 11px; border-bottom: 1px solid #30363d; padding-bottom: 5px; text-transform: uppercase; }
    td { padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 13px; }
    .pos { color: #58a6ff; font-weight: bold; width: 45px; }
    .name { color: #e6edf3; font-weight: 600; }
    .meta { color: #8b949e; font-size: 11px; }
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
    return df[df['Proj'] > 0.1].reset_index(drop=True)

class VantageOptimizer:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_p = len(df)

    def get_dk_slots(self, lineup_df):
        players = lineup_df.sort_values('time_obj', ascending=False).to_dict('records')
        flex_slots = ['UTIL', 'F', 'G']
        specific_slots = ['C', 'PF', 'SF', 'SG', 'PG']
        
        def fits(p, s):
            pos = p['Pos']
            if s == 'UTIL': return True
            if s == 'G': return 'PG' in pos or 'SG' in pos
            if s == 'F': return 'SF' in pos or 'PF' in pos
            return s in pos

        final_assignment = {}
        used_mask = 0

        for s_name in flex_slots:
            for i, p in enumerate(players):
                if not (used_mask & (1 << i)) and fits(p, s_name):
                    final_assignment[s_name] = p.copy()
                    final_assignment[s_name]['Slot'] = s_name
                    used_mask |= (1 << i)
                    break

        remaining_slots = [s for s in specific_slots if s not in final_assignment]
        
        def backtrack_specific(slot_idx, current_mask):
            if slot_idx == len(remaining_slots): return True
            target_slot = remaining_slots[slot_idx]
            for i, p in enumerate(players):
                if not (current_mask & (1 << i)) and fits(p, target_slot):
                    final_assignment[target_slot] = p.copy()
                    final_assignment[target_slot]['Slot'] = target_slot
                    if backtrack_specific(slot_idx + 1, current_mask | (1 << i)):
                        return True
            return False

        if backtrack_specific(0, used_mask):
            display_order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            return pd.DataFrame([final_assignment[s] for s in display_order])
        return None

    def run_gpp_sims(self, n_lineups=10, min_uniques=3):
        np.random.seed(42)
        
        def parse_time(info):
            try:
                m = re.search(r'(\d{1,2}:\d{2}[APM]{2})', info)
                return datetime.strptime(m.group(1), '%I:%M%p') if m else datetime.min
            except: return datetime.min

        self.df['time_obj'] = self.df['GameInfo'].apply(parse_time)
        latest_time = self.df['time_obj'].max()
        is_hammer = (self.df['time_obj'] == latest_time).astype(int)
        vols = np.where(self.df['Sal'] < 4000, 0.25, np.where(self.df['Sal'] < 8000, 0.20, 0.15))
        self.df['Ceil'] = self.df['Proj'] + (self.df['Proj'] * vols * 2.33)

        # Base Constraints
        A_base = [np.ones(self.n_p), self.df['Sal'].values]
        bl_base, bu_base = [8, 45000], [8, 50000]
        
        for p in ['PG', 'SG', 'SF', 'PF', 'C']:
            A_base.append(self.df['Pos'].str.contains(p).astype(int).values); bl_base.append(1); bu_base.append(8)
        A_base.append(self.df['Pos'].apply(lambda x: 'PG' in x or 'SG' in x).astype(int).values); bl_base.append(3); bu_base.append(8)
        A_base.append(self.df['Pos'].apply(lambda x: 'SF' in x or 'PF' in x).astype(int).values); bl_base.append(3); bu.append(8)
        A_base.append(self.df['Pos'].apply(lambda x: 'C' in x and not ('SF' in x or 'PF' in x)).astype(int).values); bl_base.append(0); bu_base.append(2)
        if is_hammer.sum() > 0: A_base.append(is_hammer.values); bl_base.append(1); bu_base.append(8)

        existing_lineups = []
        final_results = []
        
        progress = st.progress(0)
        
        for n in range(n_lineups):
            A_run = list(A_base)
            bl_run = list(bl_base)
            bu_run = list(bu_base)
            
            # UNIQUE CONSTRAINT: Force new lineup to differ from all previous ones
            for old_idx in existing_lineups:
                row = np.zeros(self.n_p)
                row[list(old_idx)] = 1
                A_run.append(row)
                bl_run.append(0)
                bu_run.append(8 - min_uniques)
            
            # Generate GPP Noise for this specific run
            sim_p = self.df['Proj'].values * (1.0 + (np.random.normal(0, 1, self.n_p) * vols))
            
            res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A_run), bl_run, bu_run), 
                       integrality=np.ones(self.n_p), bounds=Bounds(0, 1))
            
            if res.success:
                idx = tuple(sorted(np.where(res.x > 0.5)[0]))
                existing_lineups.append(idx)
                ldf = self.get_dk_slots(self.df.iloc[list(idx)])
                if ldf is not None:
                    final_results.append({
                        'df': ldf,
                        'proj': self.df.iloc[list(idx)]['Proj'].sum(),
                        'ceil': self.df.iloc[list(idx)]['Ceil'].sum(),
                        'hammer_count': self.df.iloc[list(idx)]['time_obj'].apply(lambda t: t == latest_time).sum()
                    })
            progress.progress((n + 1) / n_lineups)
            
        return final_results

# --- UI FLOW ---
st.sidebar.title("🕹️ COMMAND")
f = st.sidebar.file_uploader("UPLOAD CSV", type="csv")
scratches = st.sidebar.text_area("🚑 ADD SCRATCHES", height=100)

if f:
    data = process_data(f.getvalue(), scratches)
    if st.button("🚀 GENERATE 10 GPP LINEUPS"):
        st.session_state.results = VantageOptimizer(data).run_gpp_sims(n_lineups=10, min_uniques=3)

if 'results' in st.session_state:
    cols = st.columns(2)
    for i, res in enumerate(st.session_state.results):
        rows = "".join([f"<tr><td class='pos'>{r['Slot']}</td><td><span class='name'>{r['Name']}</span> <span class='meta'>({r['Team']})</span></td><td class='sal'>${int(r['Sal'])}</td><td class='proj'>{round(r['Proj'],1)}</td><td class='ceil'>{round(r['Ceil'],1)}</td></tr>" for _, r in res['df'].iterrows()])
        with cols[i % 2]:
            st.markdown(f"<div class='card-standard'><div style='display:flex; justify-content:space-between; margin-bottom:10px;'><div><b>GPP LINEUP #{i+1}</b>{' <span class=\"hammer-badge\">🔨 HAMMER</span>' if res['hammer_count']>0 else ''}</div><div><span class='badge' style='background:#d29922;'>CEIL: {round(res['ceil'],1)}</span></div></div><table><thead><tr><th>POS</th><th>PLAYER</th><th>SAL</th><th>PROJ</th><th>99%</th></tr></thead><tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)
