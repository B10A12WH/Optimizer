import streamlit as st
import pandas as pd
import numpy as np
import pulp
import io

# --- VANTAGE-V5.5 PRO: AUTO-AUDIT EDITION ---
st.set_page_config(page_title="VANTAGE-V5.5 PRO", layout="wide", page_icon="🏆")

class VantageProV5:
    def __init__(self, df):
        self.raw_df = df.copy()
        # Clean column headers
        self.raw_df.columns = [c.strip() for c in self.raw_df.columns]
        
        # Strategic Mapping
        mapping = {
            'Name': 'Name', 
            'Salary': 'Sal', 
            'dk_points': 'Market_Proj', 
            'Adj Own': 'Own', 
            'Pos': 'Pos', 
            'Team': 'Team',
            'Status': 'Status'
        }
        self.raw_df = self.raw_df.rename(columns=mapping)
        
        # Standardize numeric columns
        for col in ['Sal', 'Market_Proj', 'Own']:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0 if col != 'Own' else 5)

    def get_auto_scrubs(self):
        """Rule-based Automation: Scans for 'Out' status and zero projections."""
        # 1. Detect official injury tags in CSV ('O', 'OUT', 'INJ')
        csv_outs = self.raw_df[self.raw_df['Status'].isin(['O', 'OUT', 'INJ'])]['Name'].tolist()
        
        # 2. Detect players with zeroed-out projections (Market reaction)
        zero_proj = self.raw_df[self.raw_df['Market_Proj'] <= 0]['Name'].tolist()
        
        # 3. Combine and remove duplicates
        return list(set(csv_outs + zero_proj))

    def calculate_vantage_grade(self, win_pct, total_own, total_sal):
        """Proprietary Lineup Grader: Quality Audit."""
        ceiling = min(win_pct * 12, 45) 
        if total_own < 70: leverage = 20
        elif 70 <= total_own <= 105: leverage = 35 
        elif 105 < total_own <= 130: leverage = 25 
        else: leverage = 10 
        efficiency = (total_sal / 50000) * 20
        score = ceiling + leverage + efficiency
        
        if score >= 90: return "A+", score
        elif score >= 82: return "A", score
        elif score >= 75: return "B+", score
        elif score >= 68: return "B", score
        else: return "C", score

    def generate_proprietary_projections(self, alpha_weight, usage_boosts):
        self.df = self.raw_df.copy()
        def blend(row):
            boost = usage_boosts.get(row['Name'], 1.0)
            my_alpha = row['Market_Proj'] * boost
            return (my_alpha * alpha_weight) + (row['Market_Proj'] * (1 - alpha_weight))
        self.df['Final_Proj'] = self.df.apply(blend, axis=1)

    def simulate_win_pct(self, lineup_players):
        sims = 200
        scores = [sum([p['Final_Proj'] * np.random.normal(1.0, 0.22) for p in lineup_players]) for _ in range(sims)]
        target_score = 305 
        wins = sum(1 for s in scores if s >= target_score)
        if wins == 0: return round((max(scores) / target_score) * 0.45, 2), np.mean(scores)
        return (wins / sims) * 100, np.mean(scores)

    def build_pool(self, num_lineups, exp_limit, final_scrub_list, leverage_weight, min_sal):
        # Apply the final Auto-Scrubbed list
        filtered_df = self.df[~self.df['Name'].isin(final_scrub_list)].reset_index(drop=True)
        final_pool, player_counts, indices_store = [], {}, []
        progress_bar = st.progress(0)
        
        for n in range(num_lineups):
            sim_df = filtered_df.copy()
            sim_df['Sim'] = sim_df['Final_Proj'] * np.random.normal(1, 0.18, len(sim_df))
            sim_df['Lev_Fact'] = 1 + ((sim_df['Own'] / 100) * leverage_weight)
            sim_df['Shark_Score'] = (sim_df['Sim']**3) / sim_df['Lev_Fact']
            
            prob = pulp.LpProblem(f"V5_{n}", pulp.LpMaximize)
            slots = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            choices = pulp.LpVariable.dicts("C", (sim_df.index, slots), cat='Binary')
            
            prob += pulp.lpSum([sim_df.loc[i, 'Shark_Score'] * choices[i][s] for i in sim_df.index for s in slots])
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) <= 50000
            prob += pulp.lpSum([sim_df.loc[i, 'Sal'] * choices[i][s] for i in sim_df.index for s in slots]) >= min_sal
            
            for s in slots: prob += pulp.lpSum([choices[i][s] for i in sim_df.index]) == 1
            for i in sim_df.index: prob += pulp.lpSum([choices[i][s] for s in slots]) <= 1
            
            for prev in indices_store:
                prob += pulp.lpSum([choices[i][s] for i in prev for s in slots]) <= 5

            for i in sim_df.index:
                if player_counts.get(sim_df.loc[i, 'Name'], 0) >= (num_lineups * exp_limit):
                    prob += pulp.lpSum([choices[i][s] for s in slots]) == 0
            
            for i in sim_df.index:
                p_pos = str(sim_df.loc[i, 'Pos'])
                for s in slots:
                    eligible = (s == 'UTIL') or \
                               (s == 'PG' and 'PG' in p_pos) or (s == 'SG' and 'SG' in p_pos) or \
                               (s == 'SF' and 'SF' in p_pos) or (s == 'PF' and 'PF' in p_pos) or \
                               (s == 'C' and 'C' in p_pos) or \
                               (s == 'G' and ('PG' in p_pos or 'SG' in p_pos)) or \
                               (s == 'F' and ('SF' in p_pos or 'PF' in p_pos))
                    if not eligible: prob += choices[i][s] == 0

            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=5))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                lineup = {s: sim_df.loc[i] for s in slots for i in sim_df.index if choices[i][s].varValue == 1}
                l_list = list(lineup.values())
                win_pct, avg_score = self.simulate_win_pct(l_list)
                total_own = sum([p['Own'] for p in l_list])
                total_sal = sum([p['Sal'] for p in l_list])
                grade, score = self.calculate_vantage_grade(win_pct, total_own, total_sal)
                
                final_pool.append({
                    'players': lineup,
                    'metrics': {
                        'Sal': total_sal, 'Own': round(total_own, 1), 
                        'Win': win_pct, 'Grade': grade, 
                        'Score': score, 'Avg': round(avg_score, 1)
                    }
                })
                
                curr_idx = [i for i in sim_df.index if any(choices[i][s].varValue == 1 for s in slots)]
                indices_store.append(curr_idx)
                for i in curr_idx:
                    player_counts[sim_df.loc[i, 'Name']] = player_counts.get(sim_df.loc[i, 'Name'], 0) + 1
            progress_bar.progress((n + 1) / num_lineups)
            
        return final_pool

# --- UI SECTION ---
uploaded_file = st.file_uploader("Upload SaberSim CSV", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    engine = VantageProV5(raw_data)
    
    st.sidebar.header("🕹️ Architect Controls")
    num_lineups = st.sidebar.slider("Number of Lineups", 1, 50, 15)
    min_sal = st.sidebar.slider("Min Salary Target", 48000, 50000, 49750)
    
    alpha_weight = st.sidebar.slider("Alpha System Weight", 0.0, 1.0, 0.75)
    leverage_weight = st.sidebar.slider("Leverage Strength", 0.0, 2.0, 1.15)
    exp_limit = st.sidebar.slider("Exposure Cap", 0.1, 1.0, 0.5)
    
    # Manual Scrubs removed as per request - Automation takes the wheel
    mitchell_b = st.sidebar.slider("Mitchell Boost", 1.0, 1.5, 1.22)
    barnes_b = st.sidebar.slider("Barnes Boost", 1.0, 1.5, 1.28)
    
    if st.button("🔥 GENERATE PORTFOLIO"):
        # Run Auto-Audit automation
        auto_scrubs = engine.get_auto_scrubs()
        st.info(f"🤖 Auto-Audit: {len(auto_scrubs)} players scrubbed (Official 'OUT' status or 0.0 Proj).")
        
        with st.status("Simulating Outcomes...", expanded=True) as status:
            engine.generate_proprietary_projections(alpha_weight, {'Donovan Mitchell': mitchell_b, 'Scottie Barnes': barnes_b})
            pool = engine.build_pool(num_lineups, exp_limit, auto_scrubs, leverage_weight, min_sal)
            status.update(label="Portfolio Ready!", state="complete", expanded=False)
        
        # Exposure Summary
        st.subheader("📊 Portfolio Exposure Audit")
        all_players = []
        for l in pool: all_players.extend([p['Name'] for p in l['players'].values()])
        exp_df = pd.Series(all_players).value_counts(normalize=True).mul(100).round(1).reset_index()
        exp_df.columns = ['Name', 'Exposure %']
        st.dataframe(exp_df, use_container_width=True, height=250)

        st.markdown("---")
        for i, l in enumerate(pool):
            m = l['metrics']
            with st.expander(f"[{m['Grade']}] Lineup #{i+1} | Win%: {m['Win']}% | Own: {m['Own']}% | Grade Score: {round(m['Score'], 1)}"):
                st.write(f"**Vantage Score:** Ceiling potential detected at **{m['Avg']}** points.")
                display_df = pd.DataFrame(l['players']).T[['Name', 'Team', 'Sal', 'Own']]
                display_df.columns = ['Player', 'Team', 'Salary', 'Ownership %']
                st.table(display_df)
            
        # Export logic
        export_rows = [[l['players'][s]['Name'] for s in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']] for l in pool]
        export_df = pd.DataFrame(export_rows, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'])
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button("💾 Download DK Upload", data=csv_buffer.getvalue(), file_name="vantage_lock.csv", mime="text/csv")
