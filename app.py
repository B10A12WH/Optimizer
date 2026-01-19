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
    
    /* CARD STYLES */
    .card-elite { border: 1px solid #238636; background: rgba(35, 134, 54, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .card-strong { border: 1px solid #d29922; background: rgba(210, 153, 34, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .card-standard { border: 1px solid #30363d; background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    
    /* TABLE STYLES */
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; color: #8b949e; font-size: 12px; border-bottom: 1px solid #30363d; padding-bottom: 5px; text-transform: uppercase; }
    td { padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 14px; }
    .pos { color: #58a6ff; font-weight: bold; width: 50px; }
    .name { color: #e6edf3; font-weight: 600; }
    .meta { color: #8b949e; font-size: 12px; }
    .sal { color: #7ee787; font-family: monospace; text-align: right; }
    .proj { color: #c9d1d9; font-weight: bold; text-align: right; }
    .ceil { color: #d29922; font-weight: bold; text-align: right; font-family: monospace; }
    
    /* BADGES */
    .badge { background: #238636; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
    .hammer-badge { background: #8250df; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; margin-left: 5px; }
    </style>
    """, unsafe_allow_html=True)

# IRON-CLAD INJURY BLACKLIST
FORCED_OUT = ["Kuminga", "Toppin", "Haliburton", "Mathurin", "Garland", "Brunson", "Kyrie", "Embiid", "Hartenstein", "Gafford"]

@st.cache_data
def process_data(file_bytes, manual_scratches_str):
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    
    # Standardize
    df['Proj'] = pd.to_numeric(df[cols.get('proj', df.columns[-1])], errors='coerce').fillna(0.0)
    df['Sal'] = pd.to_numeric(df[cols.get('salary', df.columns[5])], errors='coerce').fillna(50000)
    df['Name'] = df[cols.get('name', df.columns[2])].astype(str).str.strip()
    df['Pos'] = df[cols.get('position', df.columns[0])].astype(str).str.strip()
    df['Team'] = df[cols.get('teamabbrev', df.columns[7])].astype(str)
    df['GameInfo'] = df[cols.get('gameinfo', df.columns[6])].astype(str)
    
    # Filtering
    manual_list = [s.strip().lower() for s in manual_scratches_str.split('\n') if s.strip()]
    full_scratch_list = [s.lower() for s in FORCED_OUT] + manual_list
    
    mask = df['Name'].str.lower().apply(lambda x: any(scratch in x for scratch in full_scratch_list))
    df = df[~mask]
    
    # NUCLEAR FILTER
    df = df[~df['Name'].str.contains("Toppin", case=False)]
    df = df[~df['Name'].str.contains("Kuminga", case=False)]
    
    return df[df['Proj'] > 0.1].reset_index(drop=True)

# --- WORKER FUNCTION ---
def solve_batch_task(args):
    (indices, proj_matrix, A_matrix, bl, bu, n_p) = args
    constraints = LinearConstraint(A_matrix, bl, bu)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_p)
    batch_counts = {}
    
    for i, sim_idx in enumerate(indices):
        sim_p = proj_matrix[i] 
        res = milp(c=-sim_p, constraints=constraints, integrality=integrality, bounds=bounds)
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
        PLAYER-CENTRIC SOLVER:
        Re-creates the DataFrame from the assigned players list to avoid KeyErrors.
        """
        # Convert to list of dicts for the recursive solver
        players = lineup_df.to_dict('records')
        
        all_slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        def get_valid_slots_for_player(p):
            pos = p['Pos']
            valid = []
            if 'PG' in pos: valid.extend(['PG', 'G', 'UTIL'])
            if 'SG' in pos: valid.extend(['SG', 'G', 'UTIL'])
            if 'SF' in pos: valid.extend(['SF', 'F', 'UTIL'])
            if 'PF' in pos: valid.extend(['PF', 'F', 'UTIL'])
            if 'C' in pos: valid.extend(['C', 'UTIL'])
            return sorted(list(set(valid)))

        # 1. Sort by restrictiveness
        for p in players:
            p['valid_slots'] = get_valid_slots_for_player(p)
            p['n_options'] = len(p['valid_slots'])
            
        players.sort(key=lambda x: x['n_options'])
        
        # 2. Recursive Placement
        def place_player(player_idx, open_slots):
            if player_idx == len(players):
                return True
            
            p = players[player_idx]
            
            for slot in p['valid_slots']:
                if slot in open_slots:
                    fits = False
                    if slot == 'UTIL': fits = True
                    elif slot == 'G': fits = ('PG' in p['Pos'] or 'SG' in p['Pos'])
                    elif slot == 'F': fits = ('SF' in p['Pos'] or 'PF' in p['Pos'])
                    else: fits = (slot in p['Pos'])
                    
                    if fits:
                        p['Slot'] = slot
                        new_open = open_slots.copy()
                        new_open.remove(slot)
                        if place_player(player_idx + 1, new_open):
                            return True
            return False

        initial_slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        success = place_player(0, initial_slots)
        
        if success:
            # Re-create DataFrame from the UPDATED players list
            result_df = pd.DataFrame(players)
            
            display_map = {k: v for v, k in enumerate(all_slots)}
            result_df['sort_val'] = result_df['Slot'].map(display_map)
            return result_df.sort_values('sort_val').drop(['valid_slots', 'n_options', 'sort_val'], axis=1, errors='ignore')
        else:
            # Fallback
            lineup_df = lineup_df.copy()
            lineup_df['Slot'] = 'UTIL'
            return lineup_df

    def run_sims(self, n_sims=8000): 
        np.random.seed(42)

        def parse_time(info):
            try:
                match = re.search(r'(\d{1,2}:\d{2}[APM]{2})', info)
                if match: return datetime.strptime(match.group(1), '%I:%M%p')
            except: pass
            return datetime.min

        self.df['time_obj'] = self.df['GameInfo'].apply(parse_time)
        latest_time = self.df['time_obj'].max()
        is_hammer = (self.df['time_obj'] == latest_time).astype(int)
        
        unique_teams = self.df['Team'].unique()
        team_map = {t: i for i, t in enumerate(unique_teams)}
        player_team_indices = self.df['Team'].map(team_map).values
        n_teams = len(unique_teams)
        
        volatility_arr = []
        for sal in self.df['Sal']:
            if sal < 4000: volatility_arr.append(0.25)
            elif sal < 8000: volatility_arr.append(0.20)
            else: volatility_arr.append(0.15)
        volatility_arr = np.array(volatility_arr)
        
        self.df['Ceil'] = self.df['Proj'] + (self.df['Proj'] * volatility_arr * 2.33)

        A_rows = [np.ones(self.n_p), self.df['Sal'].values]
        bl, bu = [8, 45000], [8, 50000]
        
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            A_rows.append(self.df['Pos'].str.contains(pos).astype(int).values)
            bl.append(1); bu.append(8)
            
        is_guard = self.df['Pos'].apply(lambda x: 'PG' in x or 'SG' in x).astype(int)
        A_rows.append(is_guard.values)
        bl.append(3); bu.append(8)

        is_forward = self.df['Pos'].apply(lambda x: 'SF' in x or 'PF' in x).astype(int)
        A_rows.append(is_forward.values)
        bl.append(3); bu.append(8)
        
        is_pure_center = self.df['Pos'].apply(lambda x: 'C' in x and not ('SF' in x or 'PF' in x)).astype(int)
        A_rows.append(is_pure_center.values)
        bl.append(0); bu.append(2)
        
        if is_hammer.sum() > 0:
            A_rows.append(is_hammer.values)
            bl.append(1); bu.append(8)

        A_matrix = np.vstack(A_rows)
        
        team_noise_matrix = np.random.normal(1.0, 0.15, (n_sims, n_teams))
        base_noise = np.random.normal(0, 1, (n_sims, self.n_p))
        player_noise_matrix = 1.0 + (base_noise * volatility_arr)
        
        all_sim_projections = np.zeros((n_sims, self.n_p))
        for i in range(n_sims):
            sim_team_noise = team_noise_matrix[i][player_team_indices]
            combined_noise = (player_noise_matrix[i] * 0.6) + (sim_team_noise * 0.4)
            all_sim_projections[i] = self.df['Proj'].values * combined_noise

        n_workers = os.cpu_count() or 4
        chunk_size = n_sims // n_workers
        tasks = []
        for i in range(n_workers):
            start = i * chunk_size
            end = start + chunk_size if i < n_workers - 1 else n_sims
            batch_proj = all_sim_projections[start:end]
            batch_indices = range(start, end)
            tasks.append((batch_indices, batch_proj, A_matrix, bl, bu, self.n_p))
            
        lineup_counts = {}
        progress_bar = st.progress(0)
        completed_sims = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for batch_result in executor.map(solve_batch_task, tasks):
                for idx, count in batch_result.items():
                    lineup_counts[idx] = lineup_counts.get(idx, 0) + count
                completed_sims += chunk_size
                progress_bar.progress(min(completed_sims / n_sims, 1.0))

        sorted_lineups = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        max_freq = sorted_lineups[0][1] if sorted_lineups else 1
        
        return [{
            'df': self.get_dk_slots(self.df.iloc[list(idx)]), 
            'rel_score': int((count/max_freq)*100), 
            'proj': self.df.iloc[list(idx)]['Proj'].sum(),
            'ceil': self.df.iloc[list(idx)]['Ceil'].sum(),
            'hammer_count': self.df.iloc[list(idx)]['time_obj'].apply(lambda t: t == latest_time).sum()
        } for idx, count in sorted_lineups]

# --- APP FLOW ---
st.sidebar.title("🕹️ COMMAND")
f = st.sidebar.file_uploader("UPLOAD CSV", type="csv")
scratches_input = st.sidebar.text_area("🚑 ADD SCRATCHES", height=100)

if f:
    data = process_data(f.getvalue(), scratches_input)
    if st.button("🚀 GENERATE SIMS (MULTI-CORE)"):
        optimizer = VantageOptimizer(data)
        st.session_state.results = optimizer.run_sims()

if 'results' in st.session_state:
    cols = st.columns(2)
    for i, res in enumerate(st.session_state.results):
        score = res['rel_score']
        card_class = "card-elite" if score >= 90 else "card-strong" if score >= 75 else "card-standard"
        
        rows_html = ""
        for _, row in res['df'].iterrows():
            slot = row.get('Slot', 'ERR')
            name = row.get('Name', 'Unknown')
            team = row.get('Team', 'N/A')
            sal = int(row.get('Sal', 0))
            proj = round(row.get('Proj', 0.0), 1)
            ceil = round(row.get('Ceil', 0.0), 1)
            
            rows_html += f"""
            <tr>
                <td class="pos">{slot}</td>
                <td><span class="name">{name}</span> <span class="meta">({team})</span></td>
                <td class="sal">${sal}</td>
                <td class="proj">{proj}</td>
                <td class="ceil">{ceil}</td>
            </tr>"""
        
        hammer_badge = ""
        if res.get('hammer_count', 0) > 0:
            hammer_badge = '<span class="hammer-badge">🔨 HAMMER</span>'

        with cols[i % 2]:
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <div>
                        <span style="font-weight:bold; font-size:1.1em; color: white;">LINEUP #{i+1}</span>
                        {hammer_badge}
                    </div>
                    <div>
                        <span class="badge" style="background:#238636;">SCORE: {score}</span>
                        <span class="badge" style="background:#d29922;">CEIL: {round(res['ceil'], 1)}</span>
                    </div>
                </div>
                <table>
                    <thead>
                        <tr><th>POS</th><th>PLAYER</th><th>SAL</th><th>PROJ</th><th>99%</th></tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
