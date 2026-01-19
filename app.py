import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re

# --- ELITE UI & MULTI-SPORT CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | DK ENTRY ORDER", layout="wide", page_icon="⚡")

class VantageUnifiedOptimizer:
    def __init__(self, df, sport="NBA"):
        self.df = df.copy()
        self.sport = sport
        self._clean_data()

    def _clean_data(self):
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'avgpoints'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        self.df = self.df[self.df['Proj'] > 0.5].reset_index(drop=True)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    def get_legal_slots(self, lineup_df):
        """
        HARD LOCK: Assigns players to specific DraftKings legal slots in correct entry order.
        Order: PG, SG, SF, PF, C, G, F, UTIL 
        """
        if self.sport == "NFL":
            # NFL Order: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
            slots = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
        else:
            # NBA Order: PG, SG, SF, PF, C, G, F, UTIL 
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        
        assigned = []
        remaining_players = lineup_df.to_dict('records')

        for slot in slots:
            for i, player in enumerate(remaining_players):
                can_fill = False
                pos = player['Pos']
                
                # Logic for NBA Slot Filling 
                if self.sport == "NBA":
                    if slot in pos: can_fill = True
                    elif slot == 'G' and ('PG' in pos or 'SG' in pos): can_fill = True
                    elif slot == 'F' and ('SF' in pos or 'PF' in pos): can_fill = True
                    elif slot == 'UTIL': can_fill = True
                
                # Logic for NFL Slot Filling 
                elif self.sport == "NFL":
                    if slot == player['Pos']: can_fill = True
                    elif slot == 'FLEX' and any(x in pos for x in ['RB', 'WR', 'TE']): can_fill = True
                
                if can_fill:
                    p_copy = player.copy()
                    p_copy['Slot'] = slot
                    assigned.append(p_copy)
                    remaining_players.pop(i)
                    break
        
        return pd.DataFrame(assigned)

    def get_legal_constraints(self):
        n_p = len(self.df)
        A, bl, bu = [], [], []
        # Total Players
        total = 9 if self.sport == "NFL" else 8
        A.append(np.ones(n_p)); bl.append(total); bu.append(total)
        # Salary
        A.append(self.df['Sal'].values); bl.append(45000); bu.append(50000)

        if self.sport == "NFL":
            # DraftKings NFL Positions 
            for p in ['QB', 'RB', 'WR', 'TE', 'DST']:
                mask = (self.df['Pos'] == p).astype(int).values
                if p == 'QB': bl.append(1); bu.append(1)
                elif p == 'RB': bl.append(2); bu.append(3)
                elif p == 'WR': bl.append(3); bu.append(4)
                elif p == 'TE': bl.append(1); bu.append(2)
                else: bl.append(1); bu.append(1)
                A.append(mask)
        else:
            # DraftKings NBA Positions (Accounting for multi-eligibility) 
            for p in ['PG', 'SG', 'SF', 'PF', 'C']:
                A.append(self.df['Pos'].str.contains(p).astype(int).values); bl.append(1); bu.append(5)

        return np.vstack(A), bl, bu

    def run_alpha_sims(self, n_lineups=10):
        n_p = len(self.df)
        A, bl, bu = self.get_legal_constraints()
        lineup_pool = []
        
        # Performance Sim logic
        raw_p = self.df['Proj'].values
        for i in range(500):
            sim_p = np.random.normal(raw_p, raw_p * 0.2).clip(min=0)
            res = milp(c=-sim_p, constraints=LinearConstraint(A, bl, bu),
                       integrality=np.ones(n_p), bounds=Bounds(0, 1))
            if res.success:
                idx = np.where(res.x > 0.5)[0]
                lineup_pool.append(self.df.iloc[idx])
            if len(lineup_pool) >= n_lineups: break
        return lineup_pool

# --- UI ---
st.title("⚡ VANTAGE 99 | ENTRY READY")
mode = st.sidebar.radio("SPORT", ["NBA", "NFL"])
f = st.file_uploader("UPLOAD SALARY CSV", type="csv")

if f:
    engine = VantageUnifiedOptimizer(pd.read_csv(f), sport=mode)
    if st.button("🚀 GENERATE ORDERED LINEUPS"):
        results = engine.run_alpha_sims()
        for i, ldf in enumerate(results):
            # Apply the Slot Assignment Engine 
            ordered_df = engine.get_legal_slots(ldf)
            with st.expander(f"LINEUP #{i+1} | Total Proj: {round(ordered_df['Proj'].sum(), 1)}"):
                # Display in legal DraftKings format 
                st.table(ordered_df[['Slot', 'Name', 'Team', 'Sal', 'Proj']])
