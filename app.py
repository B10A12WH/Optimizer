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
    
    /* CARD STYLES */
    .card-elite { border: 1px solid #238636; background: rgba(35, 134, 54, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .card-strong { border: 1px solid #d29922; background: rgba(210, 153, 34, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .card-standard { border: 1px solid #30363d; background: rgba(48, 54, 61, 0.2); padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    
    /* TABLE STYLES */
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; color: #8b949e; font-size: 12px; border-bottom: 1px solid #30363d; padding-bottom: 5px; }
    td { padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 14px; }
    .pos { color: #58a6ff; font-weight: bold; width: 50px; }
    .name { color: #e6edf3; font-weight: 600; }
    .meta { color: #8b949e; font-size: 12px; }
    .sal { color: #7ee787; font-family: monospace; text-align: right; }
    .proj { color: #c9d1d9; font-weight: bold; text-align: right; }
    
    /* BADGES */
    .badge { background: #238636; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# INJURY BLACKLIST
FORCED_OUT = ["Toppin", "Haliburton", "Mathurin", "Garland", "Brunson", "Kyrie", "Embiid", "Hartenstein", "Gafford"]

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
    
    # Hard Logic for Obi Toppin
    df = df[~df['Name'].str.contains("Toppin", case=False)]
    
    return df[df['Proj'] > 0.1].reset_index(drop=True)

class VantageOptimizer:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.n_p = len(df)

    def get_dk_slots(self, lineup_df):
        """
        SCARCITY SOLVER:
        Fills the hardest positions (C, then Positions, then G/F) first.
        This prevents the "All UTIL" error by ensuring flexible players 
        aren't wasted on easy slots.
        """
        lineup_df = lineup_df.copy()
        players = lineup_df.to_dict('records')
        
        # We solve in this specific order to maximize success rate
        # C is usually the hardest to fill if we have multiple.
        solve_order = ['C', 'PG', 'SG', 'SF', 'PF', 'G', 'F', 'UTIL']
        
        # But we want to DISPLAY in this order
        display_order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        # Helper to check if player fits slot
        def fits(player, slot):
            pos = player['Pos']
            if slot == 'UTIL': return True
            if slot == 'G': return ('PG' in pos or 'SG' in pos)
            if slot == 'F': return ('SF' in pos or 'PF' in pos)
            return slot in pos

        # Recursive solver
        # We map assignment by slot name for easier re-ordering later
        assignment = {}
        
        def solve(order_idx, available_players):
            if order_idx == 8:
                return True # All slots filled
            
            slot_name = solve_order[order_idx]
            
            # Try every available player for this slot
            for i, p in enumerate(available_players):
                if fits(p, slot_name):
                    assignment[slot_name] = p
                    remaining = available_players[:i] + available_players[i+1:]
                    if solve(order_idx + 1, remaining):
                        return True
            return False

        success = solve(0, players)
        
        if success:
            # Reconstruct DataFrame in the strict display order
            ordered_data = []
            for slot in display_order:
                p = assignment[slot]
                p['Slot'] = slot
                ordered_data.append(p)
            return pd.DataFrame(ordered_data)
        else:
            # Fallback (Assign UTIL to prevent crash, but this implies MILP constraints failed)
            lineup_df['Slot'] = 'UTIL'
            return lineup_df

    def run_sims(self, n_sims=5000):
        # 1. IDENTIFY LATEST GAME (THE HAMMER)
        def parse_time(info):
            try:
                match = re.search(r'(\d{1,2}:\d{2}[APM]{2})', info)
                if match:
                    return datetime.strptime(match.group(1), '%I:%M%p')
            except:
                pass
            return datetime.min

        self.df['time_obj'] = self.df['GameInfo'].apply(parse_time)
        latest_time = self.df['time_obj'].max()
        is_hammer = (self.df['time_obj'] == latest_time).astype(int)
        
        # --- MILP SETUP ---
        A_rows = [np.ones(self.n_p), self.df['Sal'].values]
        bl, bu = [8, 45000], [8, 50000]
        
        # 1. Positional Constraints
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            A_rows.append(self.df['Pos'].str.contains(pos).astype(int).values)
            bl.append(1); bu.append(8)
            
        # 2. Flex Constraints (Crucial for valid lineups)
        # Guard Eligible
        is_guard = self.df['Pos'].apply(lambda x: 'PG' in x or 'SG' in x).astype(int)
        A_rows.append(is_guard.values)
        bl.append(3); bu.append(8) # Min 3 Guards (PG, SG, G)

        # Forward Eligible
        is_forward = self.df['Pos'].apply(lambda x: 'SF' in x or 'PF' in x).astype(int)
        A_rows.append(is_forward.values)
        bl.append(3); bu.append(8) # Min 3 Forwards (SF, PF, F)
        
        # 3. PURE CENTER LIMIT (The fix for "All UTIL")
        # Players who are C but NOT F-eligible (can't play SF/PF) cannot exceed 2 (C + UTIL)
        is_pure_center = self.df['Pos'].apply(lambda x: 'C' in x and not ('SF' in x or 'PF' in x)).astype(int)
        A_rows.append(is_pure_center.values)
        bl.append(0); bu.append(2) # Max 2 pure centers
        
        # 4. HAMMER CONSTRAINT (Must have at least 1 player from latest game)
        if is_hammer.sum() > 0:
            A_rows.append(is_hammer.values)
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
            
            if i % 500 == 0:
                progress_bar.progress((i + 1) / n_sims)
        
        sorted_lineups = sorted(lineup_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        max_freq = sorted_lineups[0][1] if sorted_lineups else 1
        
        return [{
            'df': self.get_dk_slots(self.df.iloc[list(idx)]), 
            'win_pct': (count/n_sims)*100, 
            'rel_score': (count/max_freq)*100, 
            'proj': self.df.iloc[list(idx)]['Proj'].sum(),
            'hammer_count': self.df.iloc[list(idx)]['time_obj'].apply(lambda t: t == latest_time).sum()
        } for idx, count in sorted_lineups]

# --- APP FLOW ---
st.sidebar.title("🕹️ COMMAND")
f = st.sidebar.file_uploader("UPLOAD CSV", type="csv")
scratches_input = st.sidebar.text_area("🚑 ADD SCRATCHES", height=100)

if f:
    data = process_data(f.getvalue(), scratches_input)
    if st.button("🚀 GENERATE SIMS"):
        optimizer = VantageOptimizer(data)
        st.session_state.results = optimizer.run_sims()

if 'results' in st.session_state:
    cols = st.columns(2)
    for i, res in enumerate(st.session_state.results):
        score = res['rel_score']
        card_class = "card-elite" if score > 85 else "card-strong" if score > 50 else "card-standard"
        
        # Construct Rows Strictly
        rows_html = ""
        for _, row in res['df'].iterrows():
            slot = row.get('Slot', 'ERR')
            name = row.get('Name', 'Unknown')
            team = row.get('Team', 'N/A')
            sal = int(row.get('Sal', 0))
            proj = round(row.get('Proj', 0.0), 1)
            
            rows_html += f"""
            <tr>
                <td class="pos">{slot}</td>
                <td><span class="name">{name}</span> <span class="meta">({team})</span></td>
                <td class="sal">${sal}</td>
                <td class="proj">{proj}</td>
            </tr>"""
        
        # Add Hammer Badge if applicable
        hammer_badge = ""
        if res.get('hammer_count', 0) > 0:
            hammer_badge = '<span class="badge" style="background:#8250df; margin-left:5px;">🔨 HAMMER</span>'

        with cols[i % 2]:
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <div>
                        <span style="font-weight:bold; font-size:1.1em; color: white;">LINEUP #{i+1}</span>
                        {hammer_badge}
                    </div>
                    <span class="badge">WIN: {round(res['win_pct'], 1)}%</span>
                </div>
                <table>
                    <thead>
                        <tr><th>POS</th><th>PLAYER</th><th>SAL</th><th>PROJ</th></tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
