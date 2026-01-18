import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- ELITE NBA UI CONFIG ---
st.set_page_config(page_title="VANTAGE 99 | v91.0 STABILITY", layout="wide", page_icon="🏀")

class EliteNBAGPPOptimizerV91:
    def __init__(self, df):
        self.df = df.copy()
        raw_cols = {c.lower().replace(" ", "").replace("%", ""): c for c in df.columns}
        def hunt(keys, default_val=None):
            for k in keys:
                if k in raw_cols: return raw_cols[k]
            return default_val

        p_key = hunt(['proj', 'fppg', 'points', 'avgpointspergame'], df.columns[0])
        s_key = hunt(['salary', 'sal', 'cost'], 'Salary')
        o_key = hunt(['ownership', 'own', 'projown', 'roster'], None)
        
        self.df['Proj'] = pd.to_numeric(df[p_key], errors='coerce').fillna(0.0)
        self.df['Sal'] = pd.to_numeric(df[s_key], errors='coerce').fillna(50000)
        self.df['Pos'] = df[hunt(['position', 'pos'], 'Position')].astype(str)
        self.df['ID'] = df[hunt(['id', 'playerid'], 'ID')].astype(str)
        self.df['Own'] = pd.to_numeric(df[o_key], errors='coerce').fillna(15.0) if o_key else 15.0

        # Positional & Strategy Masks
        for p in ['PG', 'SG', 'SF', 'PF', 'C']: self.df[f'mask_{p}'] = self.df['Pos'].str.contains(p).astype(int)
        self.df['mask_G'] = (self.df['mask_PG'] | self.df['mask_SG']).astype(int)
        self.df['mask_F'] = (self.df['mask_SF'] | self.df['mask_PF']).astype(int)
        
        self.df['is_Star'] = (self.df['Sal'] >= 9500).astype(int)
        self.df['is_Punt'] = (self.df['Sal'] <= 4200).astype(int)
        self.df['is_Contrarian'] = (self.df['Own'] < 10.0).astype(int)

    def assemble(self, n_final=10, total_sims=10000):
        n_p = len(self.df); raw_p = self.df['Proj'].values.astype(np.float64)
        sals = self.df['Sal'].values.astype(np.float64); owns = self.df['Own'].values.astype(np.float64)
        sim_matrix = np.random.normal(loc=raw_p, scale=np.abs(raw_p * 0.28), size=(total_sims, n_p)).clip(min=0)
        
        # PROGRESSIVE RELAXATION SYSTEM
        configs = [
            {"name": "Nuclear", "sal": 49600, "stars": 2, "own": 125.0, "con": 3},
            {"name": "Alpha", "sal": 49200, "stars": 1, "own": 140.0, "con": 2},
            {"name": "Standard GPP", "sal": 48500, "stars": 1, "own": 160.0, "con": 1}
        ]
        
        for cfg in configs:
            sim_pool = []
            for i in range(min(total_sims, 500)):
                sim_p = sim_matrix[i]
                A, bl, bu = [], [], []
                A.append(np.ones(n_p)); bl.append(8); bu.append(8) 
                A.append(sals); bl.append(cfg['sal']); bu.append(50000)
                A.append(owns); bl.append(0); bu.append(cfg['own'])
                A.append(self.df['is_Star'].values); bl.append(cfg['stars']); bu.append(4)
                A.append(self.df['is_Punt'].values); bl.append(1); bu.append(3)
                A.append(self.df['is_Contrarian'].values); bl.append(cfg['con']); bu.append(8)

                for p in ['PG', 'SG', 'SF', 'PF', 'C']: A.append(self.df[f'mask_{p}'].values); bl.append(1); bu.append(8)
                A.append(self.df['mask_G'].values); bl.append(3); bu.append(8)
                A.append(self.df['mask_F'].values); bl.append(3); bu.append(8)

                res = milp(c=-sim_p, constraints=LinearConstraint(np.vstack(A), bl, bu), integrality=np.ones(n_p), bounds=Bounds(0, 1))
                if res.success:
                    idx = np.where(res.x > 0.5)[0]
                    sim_pool.append({'idx': tuple(idx), 'sim_score': sim_p[idx].sum()})

            if len(sim_pool) >= n_final:
                st.info(f"Successfully built using **{cfg['name']}** constraints.")
                sorted_pool = sorted(sim_pool, key=lambda x: x['sim_score'], reverse=True)
                final_portfolio, used_hashes = [], set()
                for entry in sorted_pool:
                    if entry['idx'] not in used_hashes:
                        final_portfolio.append(self.df.iloc[list(entry['idx'])].copy())
                        used_hashes.add(entry['idx'])
                    if len(final_portfolio) >= n_final: break
                return final_portfolio
        return []
