import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import re
import io

# --- ELITE DYNAMIC UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | 3:30PM INJURY MOD", layout="wide", page_icon="⚡")

# Updated Global Scratches from the 3:30 PM PDF
# Added players like Day'Ron Sharpe and Cam Thomas who were just confirmed OUT.
OFFICIAL_330_SCRATCHES = [
    "Obi Toppin", "Tyrese Haliburton", "Bennedict Mathurin", "Isaiah Jackson",
    "Darius Garland", "Max Strus", "Sam Merrill", "Dean Wade",
    "Isaiah Hartenstein", "Jalen Williams", "Brooks Barnhizer",
    "Jalen Brunson", "Daniel Gafford", "Dereck Lively II", "Kyrie Irving",
    "Day'Ron Sharpe", "Cam Thomas", "Egor Demin", "Kristaps Porzingis", 
    "Zaccharie Risacher", "Bradley Beal", "Kawhi Leonard", "Bilal Coulibaly"
]

class VantageUnifiedOptimizer:
    def __init__(self, df, sport="NBA", manual_scratches=[]):
        self.df = df.copy()
        self.sport = sport
        # Combine Official 3:30PM report with any manual user input
        self.excluded_names = list(set(OFFICIAL_330_SCRATCHES + manual_scratches))
        self._clean_data()

    def _clean_data(self):
        cols = {c.lower().replace(" ", ""): c for c in self.df.columns}
        self.df['Proj'] = pd.to_numeric(self.df[self._hunt(['proj', 'fppg', 'avgpoints'], cols)], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(self.df[self._hunt(['salary', 'cost'], cols)], errors='coerce').fillna(50000)
        self.df['Own'] = pd.to_numeric(self.df[self._hunt(['own', 'roster'], cols)], errors='coerce').fillna(5.0)
        self.df['Pos'] = self.df[self._hunt(['pos', 'position'], cols)].astype(str)
        self.df['Team'] = self.df[self._hunt(['team', 'tm', 'abb'], cols)].astype(str)
        self.df['Name'] = self.df[self._hunt(['name', 'player'], cols)].astype(str)
        self.df['Name+ID'] = self.df[self._hunt(['name+id'], cols)].astype(str)
        
        # 1. HARD SCRATCH: Remove players from the 3:30PM Official List
        clean_excludes = [p.strip().lower() for p in self.excluded_names if p.strip()]
        self.df = self.df[~self.df['Name'].str.lower().isin(clean_excludes)]
        
        # 2. AUTO-TAG: Filter platform-marked (OUT) tags
        self.df = self.df[~self.df['Name+ID'].str.contains(r'\(OUT\)', flags=re.IGNORECASE, na=False)]
        
        # 3. Standard active filter
        self.df = self.df[self.df['Proj'] > 0.5].reset_index(drop=True)

    def _hunt(self, keys, col_map):
        for k in keys:
            for actual_col in col_map:
                if k in actual_col: return col_map[actual_col]
        return self.df.columns[0]

    # ... [Keeping get_dk_slots and run_alpha_sims from previous version] ...

# --- UI INTERFACE ---
st.sidebar.title("🕹️ COMMAND")
# Allowing users to add even more late-breaking scratches
manual_input = st.sidebar.text_area("🚑 EXTRA LATE SCRATCHES", placeholder="One name per line...")
manual_list = manual_input.split('\n') if manual_input else []

uploaded_file = st.sidebar.file_uploader("SALARY CSV", type="csv")

if uploaded_file:
    engine = VantageUnifiedOptimizer(pd.read_csv(uploaded_file), manual_scratches=manual_list)
    
    # 3:30 PM Injury Status Display
    st.sidebar.info(f"3:30PM Injury Report Loaded. {len(OFFICIAL_330_SCRATCHES)} players scratched.")
    
    # Verification Warning for Toppin
    if any("Toppin" in name for name in engine.excluded_names):
        st.sidebar.success("✅ Obi Toppin (OUT) removed from builds.")

    if st.button("🚀 EXECUTE 10,000 SIMS"):
        # run_alpha_sims logic remains same as 10k version
        st.session_state.results = engine.run_alpha_sims(n_sims=10000)
