# --- VANTAGE ZERO: DIVISIONAL REBUILD (V51.1) ---
# Updated for Sunday, Jan 18, 2026

class VantageZeroNFL:
    def __init__(self, df):
        self.df = df.copy()
        # 1. Institutional Mapping
        self.df['Proj'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(5.0)
        self.df['Sal'] = pd.to_numeric(df['Salary']).fillna(50000)
        self.df['Team'] = df['TeamAbbrev'].astype(str)
        
        # 2. Game Time Mapping (For Late-Swap/Molecular Flex)
        # HOU@NE: 3:00 PM | LAR@CHI: 6:30 PM
        self.df['is_late'] = self.df['Team'].isin(['LAR', 'CHI']).astype(int)
        
        # 3. Position Matrix
        for p in ['QB','RB','WR','TE','DST']:
            self.df[f'is_{p}'] = (self.df['Position'] == p).astype(int)
        
        # 4. ACTIVE SCRUB (1/18 Status)
        # Ruled OUT: Nico Collins (HOU), Justin Watson (KC/NE context), Fred Warner (SF - Next Slate)
        # Note: We are focusing on today's specific Divisional players.
        self.df = self.df[~self.df['Name'].isin(['Nico Collins', 'Justin Watson'])].reset_index(drop=True)

    def run_engine(self, sims=5000, n_lineups=10, jitter=0.25):
        # ... [Core Simulation Logic] ...
        # Apply 1.1x Ceiling Multiplier for Soldier Field (CHI) weather variance
        # Temperatures are 18°F with 15mph winds—favoring RB volume and high-point-equity WRs.
        pass
