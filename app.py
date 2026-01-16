import streamlit as st
import pandas as pd
import numpy as np
import pulp
import logging

# Set up Page
st.set_page_config(page_title="VANTAGE-1 ALPHA", layout="wide")

# --- ENGINE CLASS DEFINITION ---
class AlphaVantageV3:
    def __init__(self, df):
        self.raw_df = df
        time_map = {'CLE': 1, 'PHI': 1, 'NOP': 1, 'IND': 1, 'CHI': 2, 'BKN': 2, 'LAC': 2, 'TOR': 2}
        self.raw_df['Time_Tier'] = self.raw_df['Team'].map(time_map).fillna(1)
        mapping = {'Name': 'Name', 'Salary': 'Sal', 'dk_points': 'Base', 'My Own': 'Own', 'Pos': 'Pos', 'Team': 'Team', 'Opp': 'Opp'}
        self.raw_df = self.raw_df.rename(columns=mapping)
        self.raw_df['Base'] = pd.to_numeric(self.raw_df['Base'], errors='coerce').fillna(0)
        self.raw_df['Own'] = pd.to_numeric(self.raw_df['Own'], errors='coerce').fillna(5)

    def scrub_injuries(self, out_list):
        self.raw_df = self.raw_df[~self.raw_df['Name'].isin(out_list)]

    def apply_alpha(self, usage_boosts, pace_map, dvp_map):
        self.df = self.raw_df.copy()
        def calc(row):
            score = row['Base'] * pace_map.get(row['Team'], 1.0)
            score *= usage_boosts.get(row['Name'], 1.0)
            p_pos = row['Pos'].split('/')[0]
            dvp_mult = dvp_map.get(row['Opp'], {}).get(p_pos, 1.0)
            return score * dvp_mult
        self.df['Alpha_Proj'] = self.df.apply(calc, axis=1)
        self.df['Final_Proj'] = (self.df['Alpha_Proj'] * 0.8) + (self.df['Base'] * 0.2)
        self.df = self.df[self.df['Final_Proj'] > 5].reset_index(drop=True)

    def build_pool(self, num_lineups, exposure_limit):
        indices_store, final_pool, player_counts = [], [], {}
        for n in range(num_lineups):
            sim_df = self.df.copy()
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.18, len(sim_df))
            sim_df['Leverage'] = (sim_df['Sim']**3) / (sim_df['Own'] + 1)
            prob = pulp.LpProblem(f"Alpha_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            prob += pulp.lpSum([sim_df.loc[i, 'Leverage'] * choices[i][s] for i in sim_df.index for s in slots]) + \
                    pulp.lpSum([sim_df.loc[i, 'Time_Tier'] * 1000000 * choices[i]['UTIL'] for i in sim_df.index])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            for prev in indices_store: prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5
            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exposure_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or (s == 'PG' and 'PG' in p_pos) or (s == 'SG' and 'SG' in p_pos) or (s == 'SF' and 'SF' in p_pos) or (s == 'PF' and 'PF' in p_pos) or (s == 'C' and 'C' in p_pos) or (s == 'G' and ('PG' in p_pos or 'SG' in p_pos)) or (s == 'F' and ('SF' in p_pos or 'PF' in p_pos))
                    if not eligible: prob += choices[i][s] == 0
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob.status] == 'Optimal':
                lineup = {s: sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1}
                curr = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr)
                final_pool.append(lineup)
                for i in curr: player_counts[sim_df.loc[i, 'Name']] = player_counts.get(sim_df.loc[i, 'Name'], 0) + 1
        return final_pool

# --- STREAMLIT UI ---
st.title("🚀 VANTAGE-1: Institutional DFS Engine")

uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = AlphaVantageV3(raw_data)
    
    st.sidebar.header("🕹️ News Architect")
    out_list = st.sidebar.multiselect("Scrub OUT Players", raw_data['Name'].unique())
    mitchell_b = st.sidebar.slider("Donovan Mitchell Boost", 1.0, 1.5, 1.18)
    barnes_b = st.sidebar.slider("Scottie Barnes Boost", 1.0, 1.5, 1.25)
    exp_limit = st.sidebar.slider("Exposure Cap", 0.1, 1.0, 0.6)
    
    if st.button("🔥 GENERATE SHARK LINEUPS"):
        engine.scrub_injuries(out_list)
        pace = {'IND': 1.08, 'NOP': 1.05, 'PHI': 1.03, 'CLE': 1.02, 'CHI': 1.04, 'BKN': 1.01, 'TOR': 1.06, 'LAC': 0.98}
        usage = {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b}
        dvp = {'PHI': {'PG': 1.12}, 'TOR': {'C': 1.20}, 'IND': {'SF': 1.15}}
        engine.apply_alpha(usage, pace, dvp)
        
        pool = engine.build_pool(15, exp_limit)
        for i, l in enumerate(pool):
            with st.expander(f"🏆 LINEUP #{i+1}"):
                st.table(pd.DataFrame(l).T[['Name', 'Team', 'Sal']])
