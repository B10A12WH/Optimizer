import streamlit as st
import pandas as pd
import numpy as np
import pulp

st.set_page_config(page_title="VANTAGE 99", layout="wide")

def universal_parser(df):
    """Deep scans the dataframe to find and promote the correct header row."""
    # Check if headers are already correct
    cols = [str(c).lower() for c in df.columns]
    if any('name' in c for c in cols) and any('salary' in c for c in cols):
        return df
    
    # Scan the first 15 rows for 'Name' and 'Salary'
    for i, row in df.head(15).iterrows():
        row_vals = [str(v).lower() for v in row.values]
        if any('name' in v for v in row_vals) and any('salary' in v for v in row_vals):
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i].values
            return new_df.reset_index(drop=True)
    return df

class Vantage99:
    def __init__(self, p_df, i_df, s_df):
        # 1. CLEAN THE METADATA
        p_df = universal_parser(p_df)
        i_df = universal_parser(i_df)
        s_df = universal_parser(s_df)

        # Standardize headers
        p_df.columns = [str(c).strip() for c in p_df.columns]
        i_df.columns = [str(c).strip() for c in i_df.columns]
        s_df.columns = [str(c).strip() for c in s_df.columns]

        # 2. FIND DYNAMIC KEYS
        # Logic to find the Name and Salary columns in the DK file
        s_name_key = next((c for c in s_df.columns if 'name' in c.lower()), 'Name')
        s_sal_key = next((c for c in s_df.columns if 'salary' in c.lower()), 'Salary')

        # 3. MOLECULAR MERGE
        self.df = pd.merge(p_df, i_df[['Name', 'MaxExp']], on='Name', how='left')
        self.df = pd.merge(self.df, s_df[[s_name_key, s_sal_key]], left_on='Name', right_on=s_name_key, how='left')
        
        # 4. PURIFICATION
        self.df = self.df.rename(columns={'Position': 'Pos', 'ProjPts': 'Proj', 'ProjOwn': 'Own', s_sal_key: 'Sal'})
        self.df['Proj'] = pd.to_numeric(self.df['Proj'], errors='coerce').fillna(0)
        self.df['Sal'] = pd.to_numeric(self.df['Sal'], errors='coerce').fillna(5000)
        self.df['Own'] = pd.to_numeric(self.df['Own'], errors='coerce').fillna(10)

        # Blacklist OUT players
        blacklist = ['DK Metcalf', 'Gabe Davis', 'George Kittle', 'Fred Warner']
        self.df = self.df[~self.df['Name'].isin(blacklist)]
        self.df = self.df[self.df['Pos'] != 'K']

    def cook(self, n_lineups=20):
        pool = []
        player_indices = self.df.index
        # GPP 2-Game Constraints
        OWN_CAP, SAL_FLOOR, JITTER, MIN_UNIQUE = 135.0, 48200, 0.35, 3

        for n in range(n_lineups):
            sim_df = self.df.copy()
            sim_df['sim_proj'] = np.random.normal(sim_df['Proj'], sim_df['Proj'] * JITTER)
            
            # Robust Scenario Multipliers (Substring matching to avoid case/suffix issues)
            boosts = {
                'Jaxon Smith-Njigba': 1.35, 'Jauan Jennings': 1.40, 
                'Kenneth Walker': 1.30, 'Khalil Shakir':
