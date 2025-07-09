import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
import numpy as np  # << moved here so it’s always available


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Unbiased XI Selector", layout="wide")

# ------------------ SESSION INIT ------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "df" not in st.session_state:
    st.session_state.df = None

# ------------------ LOTTIE LOADER ------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

lottie_cricket = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_vu1huepg.json")
lottie_team = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_kuhijlvx.json")

# ------------------ STYLING ------------------
st.markdown("""
    <style>
        .stApp { background-color: #0d1b2a; color: white; }
        h1, h4 { color: #fcbf49; }
        .stButton > button { background-color: #1d3557; color: white; font-weight: bold; border-radius: 10px; padding: 10px 20px; }
        .stDownloadButton > button { background-color: #2a9d8f; color: white; }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("📊 Unbiased XI Tools")
selected_feature = st.sidebar.radio("Select Feature", 
                                    ["Main App Flow", "XI Cohesion Score", "Pressure Heatmap XI", "Tactical Role Analyzer", "Impact-weighted index", "Role Balance Auditor"])

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align: center;'>🏏 Unbiased XI Selector App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Make Data-Driven Cricket Selections Without Bias</h4>", unsafe_allow_html=True)
if lottie_cricket:
    st_lottie(lottie_cricket, height=150, key="cricket_header")

# ------------------ MAIN APP FLOW ------------------
if st.session_state.step == 0:
    uploaded_file = st.file_uploader("Upload your Final XI CSV (with all required columns for all features)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.unbiased_xi_df = df
        st.session_state.step = 1
        st.success("File uploaded successfully and available across all features!")

if selected_feature == "Main App Flow" and st.session_state.df is not None:
    df = st.session_state.df

    if st.button("Show Player Data"):
        st.subheader("Uploaded Player Data")
        format_filter = st.selectbox("Filter by Format", options=df["Format"].unique())
        role_filter = st.multiselect("Filter by Role", options=df["Role"].unique(), default=df["Role"].unique())
        min_matches = st.slider("Minimum Matches", min_value=0, max_value=50, value=0)

        filtered_df = df[
            (df["Format"] == format_filter) &
            (df["Role"].isin(role_filter)) &
            (df["Matches"] >= min_matches)
        ]
        st.dataframe(filtered_df)

    if st.button("Detect Biased Players"):
        st.subheader("Biased Players Detected")

        df["Performance_score"] = df["Runs"] / (df["Matches"] * 100)
        df["Bias_score"] = (df["Fame_index"] + df["Endorsement_score"]) / 200
        df["Is_Biased"] = df["Bias_score"] > 0.75

        st.session_state.df = df
        st.session_state.biased_df = df[df["Is_Biased"]]

        st.dataframe(st.session_state.biased_df[["Player Name", "Role", "Performance_score", "Bias_score"]])

        pie = px.pie(df, names="Role", title="Role Distribution")
        st.plotly_chart(pie, use_container_width=True)

        bar = px.bar(df.sort_values("Performance_score", ascending=False).head(10),
                     x="Player Name", y="Performance_score", color="Role",
                     hover_data=["Fame_index", "Endorsement_score", "Bias_score"])
        st.plotly_chart(bar, use_container_width=True)

    if st.button("Generate Final Unbiased XI"):
        st.subheader("Final Unbiased XI")
        if lottie_team:
            st_lottie(lottie_team, height=200, key="final_xi")

        unbiased_df = df[df["Is_Biased"] == False]
        top_11 = unbiased_df.sort_values(by="Performance_score", ascending=False).head(11)
        st.session_state.unbiased_xi_df = top_11
        st.dataframe(top_11[["Player Name", "Role", "Performance_score", "Bias_score"]])

        scatter = px.scatter(top_11,
                             x="Bias_score", y="Performance_score", color="Role",
                             size="Performance_score", hover_name="Player Name",
                             title="Bias vs Performance (Top XI)")
        st.plotly_chart(scatter, use_container_width=True)

        csv = top_11.to_csv(index=False).encode('utf-8')
        st.download_button("Download Final Unbiased XI CSV", data=csv, file_name="Unbiased_XI.csv", mime="text/csv")


# ------------------ XI COHESION SCORE ------------------
elif selected_feature == "XI Cohesion Score":
    cohesion_file = st.file_uploader("📂 Upload CSV with XI Cohesion Columns", type="csv")

    required_columns = [
        "Player Name", "Primary skill type", "Batting role", "Bowling role",
        "Batting position", "SR_PP", "SR_MO", "SR_DO", "Overs bowled_PP",
        "Overs bowled_MO", "Overs bowled_DO", "Opponent suitability score",
        "Matchup type", "Fielding rating", "Franchise/National team (for synergy calculation)",
        "Handedness batting", "Handedness bowling", "Partnership runs",
        "Wickets in tandem", "Matches played together", "Positional synergy index"
    ]

    if cohesion_file:
        df = pd.read_csv(cohesion_file)

        if all(col in df.columns for col in required_columns):
            df["Primary skill type"].fillna("Unknown", inplace=True)

            st.success("✅ File loaded with all required columns.")
            
            # Final Selection
            selected_players = st.multiselect("✅ Select Final XI Players", df["Player Name"].tolist(), default=df["Player Name"].tolist())
            df_selected = df[df["Player Name"].isin(selected_players)].copy()

            st.subheader("📘 XI Cohesion Score Explained")
            st.markdown("""
                - The **XI Cohesion Score** measures the overall chemistry, synergy, and past performance of players playing together.
                - It factors in shared matches, partnerships, bowling in tandem, and synergy index.
                - **Formula:** Average of all players' *Positional synergy index* (0–10 scale).
            """)

            cohesion_score = round(df_selected["Positional synergy index"].mean(), 2)

            if cohesion_score >= 8.5:
                color = "#2ecc71"
                level = "🟢 Excellent Cohesion"
                suggestion = "✅ Maintain core squad. High familiarity and role clarity."
            elif cohesion_score >= 7.0:
                color = "#f1c40f"
                level = "🟡 Good Cohesion"
                suggestion = "🔄 Minor role/position tweaks can improve team chemistry."
            elif cohesion_score >= 5.0:
                color = "#e67e22"
                level = "🟠 Moderate Cohesion"
                suggestion = "⚙️ Strengthen partnerships and balance roles better."
            else:
                color = "#e74c3c"
                level = "🔴 Low Cohesion"
                suggestion = "❌ Rethink squad or improve synergy-focused strategies."

            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; color: white; font-size: 20px;'>
                <b>{level}</b><br>
                Cohesion Score: <b>{cohesion_score}</b><br>
                {suggestion}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Cohesion Score Visualization")
            st.progress(min(cohesion_score / 10, 1.0))

            st.subheader("📋 Player Synergy Table")
            st.dataframe(df_selected[["Player Name", "Partnership runs", "Wickets in tandem", "Matches played together", "Positional synergy index"]])

            fig = px.bar(
                df_selected,
                x="Player Name",
                y="Positional synergy index",
                color="Primary skill type",
                title="Synergy Index per Player"
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_data = df_selected.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Final Selected XI", csv_data, "cohesion_score_output.csv", "text/csv")

            # Flag players for replacement
            if cohesion_score < 5.0:
                st.warning("⚠️ Cohesion Score is not optimal. You can manually add new players to boost team synergy.")

                low_synergy_threshold = st.slider("🧪 Synergy Threshold to Replace", 0.0, 10.0, 5.0)
                low_synergy_players = df_selected[df_selected["Positional synergy index"] < low_synergy_threshold]

                if not low_synergy_players.empty:
                    st.markdown("### 🚫 Players below threshold (Consider replacing):")
                    st.dataframe(low_synergy_players[["Player Name", "Positional synergy index"]])

                df_filtered = df_selected[df_selected["Positional synergy index"] >= low_synergy_threshold]

                num_new = st.number_input("➕ How many new players to add?", min_value=1, step=1)
                new_players = []

                for i in range(int(num_new)):
                    with st.expander(f"Add Player {i+1}", expanded=False):
                        name = st.text_input(f"Player Name {i+1}", key=f"name_{i}")
                        skill_type = st.selectbox(f"Primary Skill Type {i+1}", ["Batter", "Bowler", "All-rounder"], key=f"skill_{i}")
                        pos_index = st.slider(f"Positional synergy index {i+1}", 0.0, 10.0, 5.0, key=f"synergy_{i}")
                        matches_played = st.number_input(f"Matches Played Together {i+1}", min_value=0, key=f"matches_{i}")
                        partnership = st.number_input(f"Partnership Runs {i+1}", min_value=0, key=f"partnership_{i}")
                        tandem_wickets = st.number_input(f"Wickets in Tandem {i+1}", min_value=0, key=f"wickets_{i}")

                        new_players.append({
                            "Player Name": name,
                            "Primary skill type": skill_type,
                            "Partnership runs": partnership,
                            "Wickets in tandem": tandem_wickets,
                            "Matches played together": matches_played,
                            "Positional synergy index": pos_index
                        })

                if st.button("📌 Add New Players & Recalculate"):
                    new_df = pd.DataFrame(new_players)
                    combined_df = pd.concat([df_filtered, new_df], ignore_index=True)
                    new_score = round(combined_df["Positional synergy index"].mean(), 2)

                    st.success(f"✅ Updated XI Cohesion Score: {new_score}")
                    st.dataframe(combined_df[["Player Name", "Primary skill type", "Positional synergy index"]])

                    fig2 = px.bar(
                        combined_df,
                        x="Player Name",
                        y="Positional synergy index",
                        color="Primary skill type",
                        title="Updated Synergy Index per Player"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    csv_combined = combined_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Updated Final Unbiased XI", csv_combined, "updated_cohesion_score.csv", "text/csv")

        else:
            missing = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing))


        # ------------------ PRESSURE HEATMAP XI ------------------
elif selected_feature == "Pressure Heatmap XI":
    pressure_file = st.file_uploader("📂 Upload CSV with Pressure Metrics", type="csv")

    if pressure_file:
        df = pd.read_csv(pressure_file)

        if all(col in df.columns for col in ["Player Name", "Role", "Performance_score", "Pressure_score"]):
            import numpy as np

            phase_options = ["Powerplay", "Middle Overs", "Death Overs"]
            situation_options = ["Chasing", "Defending", "Clutch Moments"]

            np.random.seed(42)
            df["Phase Suitability"] = np.random.choice(phase_options, size=len(df))
            df["Match Situation"] = np.random.choice(situation_options, size=len(df))
            df["Pressure Zone"] = pd.cut(df["Pressure_score"], bins=[0, 0.55, 0.65, 1], labels=["Low", "Medium", "High"])
            df["Impact Rating"] = round((df["Performance_score"] * 0.6 + df["Pressure_score"] * 0.4) * 10, 2)

            st.success("✅ File processed with Pressure Heatmap logic.")
            st.subheader("🔥 Pressure Heatmap XI")
            st.markdown("Visualize average impact rating under pressure situations.")

            heatmap_data = df.pivot_table(
                index="Match Situation",
                columns="Pressure Zone",
                values="Impact Rating",
                aggfunc="mean"
            )

            fig = px.imshow(
                heatmap_data,
                text_auto=True,
                color_continuous_scale="RdYlGn",
                title="Average Impact Rating by Match Situation & Pressure Zone"
            )
            st.plotly_chart(fig, use_container_width=True)

            top_pressure_players = df[df["Pressure Zone"] == "High"].sort_values(by="Impact Rating", ascending=False).head(11)
            st.markdown("### 💪 Top XI Under Pressure")
            st.dataframe(top_pressure_players[["Player Name", "Role", "Phase Suitability", "Match Situation", "Impact Rating"]])

            csv_out = top_pressure_players.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Pressure Heatmap XI", csv_out, "pressure_heatmap_xi.csv", "text/csv")
        else:
            st.error("❌ Missing required columns: 'Player Name', 'Role', 'Performance_score', 'Pressure_score'")



       # ------------------ TACTICAL ROLE ANALYZER (With Calculation & Visualizations) ------------------

elif selected_feature == "Tactical Role Analyzer":
    st.subheader("📝 Tactical Role Analyzer (With Tactical Role Assignment & Insights)")

    tactical_file = st.file_uploader("📂 Upload CSV with Tactical Metrics", type="csv")

    if tactical_file:
        df = pd.read_csv(tactical_file)

        required_columns = [
            "Player Name", "Role", "batting_hand", "PP_SR", "MO_SR", "DO_SR",
            "SR_vs_spin", "SR_vs_pace", "bowling_style", "PP_ER", "MO_ER",
            "DO_ER", "wicket_taking_percentage", "ER_vs_LHB", "ER_vs_RHB"
        ]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all tactical metrics.")

            # Phase Suitability
            def assign_phase_suitability(row):
                if row["Role"] == "Batter":
                    if row["PP_SR"] >= 130:
                        return "Powerplay"
                    elif row["DO_SR"] >= 140:
                        return "Death Overs"
                    else:
                        return "Middle Overs"
                else:
                    if row["PP_ER"] <= 7:
                        return "Powerplay"
                    elif row["DO_ER"] <= 8:
                        return "Death Overs"
                    else:
                        return "Middle Overs"

            df["phase_suitability"] = df.apply(assign_phase_suitability, axis=1)

            # Matchup Strength
            def assign_matchup_strength(row):
                if row["Role"] == "Batter":
                    if row["SR_vs_spin"] >= 135:
                        return "Strong vs Spin"
                    elif row["SR_vs_pace"] >= 140:
                        return "Strong vs Pace"
                    else:
                        return "Average"
                else:
                    if row["ER_vs_LHB"] <= 7:
                        return "Strong vs LHB"
                    elif row["ER_vs_RHB"] <= 7:
                        return "Strong vs RHB"
                    else:
                        return "Average"

            df["matchup_strength"] = df.apply(assign_matchup_strength, axis=1)

            # Recommended and Backup Tactical Roles
            def assign_roles(row):
                if row["Role"] == "Batter":
                    if row["PP_SR"] >= 130:
                        return ("Powerplay Aggressor", "Anchor")
                    elif row["DO_SR"] >= 140:
                        return ("Finisher", "Middle Overs Stabilizer")
                    else:
                        return ("Middle Overs Accumulator", "Finisher")
                else:
                    if row["PP_ER"] <= 7:
                        return ("Powerplay Swing Bowler", "Containment Spinner")
                    elif row["DO_ER"] <= 8:
                        return ("Death Overs Yorker Specialist", "Match-Up Bowler")
                    else:
                        return ("Middle Overs Enforcer", "Containment Spinner")

            roles = df.apply(assign_roles, axis=1)
            df["recommended_tactical_role"] = roles.apply(lambda x: x[0])
            df["backup_tactical_role"] = roles.apply(lambda x: x[1])

            # Tactical Role Fit Score (randomized demo — can replace with your logic)
            np.random.seed(42)
            df["tactical_role_fit_score"] = np.random.randint(75, 96, size=len(df))

            # Final Tactical Table
            tactical_df = df[[
                "Player Name", "Role", "phase_suitability", "matchup_strength",
                "recommended_tactical_role", "tactical_role_fit_score", "backup_tactical_role"
            ]]

            st.subheader("📋 Final Tactical Role Table")
            st.dataframe(tactical_df)

            # Download CSV
            csv_data = tactical_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Tactical Role CSV",
                data=csv_data,
                file_name="tactical_role_output.csv",
                mime="text/csv"
            )

            # 📊 Charts and Visualizations
            st.subheader("📈 Tactical Insights")

            pie_chart = px.pie(
                tactical_df, names="recommended_tactical_role",
                title="Distribution of Recommended Tactical Roles"
            )
            st.plotly_chart(pie_chart, use_container_width=True)

            bar_chart = px.bar(
                tactical_df.sort_values("tactical_role_fit_score", ascending=False),
                x="Player Name", y="tactical_role_fit_score",
                color="Role", text_auto=True,
                title="Tactical Role Fit Score per Player"
            )
            st.plotly_chart(bar_chart, use_container_width=True)

            scatter_plot = px.scatter(
                tactical_df,
                x="matchup_strength", y="tactical_role_fit_score",
                color="Role", size="tactical_role_fit_score",
                hover_name="Player Name",
                title="Matchup Strength vs Tactical Role Fit Score"
            )
            st.plotly_chart(scatter_plot, use_container_width=True)

        else:
            missing = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing))
  

   # -------------------- Impact- Weighted contribution index -------------------- #

elif selected_feature == "Impact-weighted index":
    st.subheader("📊 Impact-Weighted Contribution Validator")

    impact_file = st.file_uploader("📂 Upload Unbiased XI CSV (with input columns)", type="csv")

    impact_required_columns = [
        "Player Name", "Batting Position", "Primary Role", "Recent Matches",
        "Runs", "Balls Faced", "Wickets", "Overs Bowled", "Bowling Economy",
        "Batting Phase Strength", "Bowling Phase Strength", "Match Impact Rating",
        "Fitness Status", "Pressure Performance", "Pitch Suitability",
        "Opposition Match-up Rating", "Captaincy Score",
        "Format", "Avg", "SR", "50s", "100s", "Pressure_score", "Fame_index", 
        "Endorsement_score", "Performance_score", "Bias_score", "Is_Biased"
    ]

    role_score_map = {
        "Opener": 9.0, "Anchor": 8.0, "Floater": 8.5, "Finisher": 9.2,
        "All-rounder": 8.5, "Spinner": 7.5, "Fast Bowler": 8.0, "Death Specialist": 9.0
    }
    pitch_map = {"Pace": 8.5, "Spin": 7.5, "Balanced": 8.0}

    if impact_file:
        df = pd.read_csv(impact_file)

        if all(col in df.columns for col in impact_required_columns):
            st.success("✅ File loaded with all required input columns.")

            def compute_scores(input_df):
                input_df["Impact Score"] = round((input_df["Runs"] / input_df["Balls Faced"]) * input_df["Match Impact Rating"], 2)
                input_df["Role Match Score"] = input_df["Primary Role"].map(role_score_map).fillna(7.0)
                input_df["Phase Balance Tag"] = input_df.apply(lambda row: f"{'PP-✔️' if row['Batting Phase Strength']=='Powerplay' else 'PP-❌'}, "
                                                                    f"{'M-✔️' if row['Batting Phase Strength']=='Middle' else 'M-❌'}, "
                                                                    f"{'D-✔️' if row['Batting Phase Strength']=='Death' else 'D-❌'}", axis=1)
                input_df["Match-up Alert"] = input_df["Opposition Match-up Rating"].apply(lambda x: "✅" if x >= 7 else "⚠️")
                anchor_count = input_df["Primary Role"].value_counts().get("Anchor", 0)
                input_df["Role Balance Alert"] = ["❌ Too Many Anchors" if anchor_count > 2 else "✅ Balanced"] * len(input_df)
                input_df["Captaincy Validation"] = input_df["Captaincy Score"].apply(lambda x: "✅" if x >= 7 else "❌")
                input_df["Pitch Fit Score"] = input_df["Pitch Suitability"].map(pitch_map).fillna(7.0)
                input_df["Overall Contribution Index"] = round(
                    (input_df["Impact Score"] + input_df["Role Match Score"] + input_df["Pitch Fit Score"]) / 3, 2
                )
                input_df["Selected in Final XI"] = input_df["Overall Contribution Index"] >= 8.0
                return input_df

            df = compute_scores(df)
            selected_xi = df[df["Selected in Final XI"] == True]
            not_selected = df[df["Selected in Final XI"] == False]

            st.subheader("📋 Initial Validated XI Table")
            st.dataframe(df)

            if len(selected_xi) == 11:
                st.success("✅ All 11 players selected with high contribution score.")
                st.dataframe(selected_xi)

                csv_data = selected_xi.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Final Unbiased XI", csv_data, "Final_Unbiased_XI.csv", "text/csv")

            else:
                needed = 11 - len(selected_xi)
                st.warning(f"⚠️ Only {len(selected_xi)} players selected. {needed} replacements needed to complete the XI.")

                st.markdown("### 👇 Add Replacement Players")

                manual_players = []
                num_replacements = st.number_input("🧍 Enter number of replacements to input", min_value=needed, step=1)

                for i in range(int(num_replacements)):
                    with st.expander(f"➕ Replacement Player {i+1}", expanded=False):
                        name = st.text_input(f"Player Name {i+1}", key=f"name_{i}")
                        batting_pos = st.number_input(f"Batting Position {i+1}", min_value=1, max_value=11, key=f"batpos_{i}")
                        role = st.selectbox(f"Primary Role {i+1}", role_score_map.keys(), key=f"role_{i}")
                        matches = st.number_input(f"Recent Matches {i+1}", min_value=1, max_value=50, key=f"matches_{i}")
                        runs = st.number_input(f"Runs {i+1}", min_value=0, key=f"runs_{i}")
                        balls = st.number_input(f"Balls Faced {i+1}", min_value=1, key=f"balls_{i}")
                        wickets = st.number_input(f"Wickets {i+1}", min_value=0, key=f"wickets_{i}")
                        overs = st.number_input(f"Overs Bowled {i+1}", min_value=0.0, key=f"overs_{i}")
                        econ = st.number_input(f"Bowling Economy {i+1}", min_value=0.0, key=f"econ_{i}")
                        bat_phase = st.selectbox(f"Batting Phase Strength {i+1}", ["Powerplay", "Middle", "Death"], key=f"bphase_{i}")
                        bowl_phase = st.selectbox(f"Bowling Phase Strength {i+1}", ["Powerplay", "Middle", "Death"], key=f"bowphase_{i}")
                        impact = st.slider(f"Match Impact Rating {i+1}", 0.0, 10.0, 7.0, key=f"impact_{i}")
                        fitness = st.selectbox(f"Fitness Status {i+1}", ["Fit", "Unfit"], key=f"fit_{i}")
                        pressure = st.slider(f"Pressure Performance {i+1}", 0.0, 10.0, 7.0, key=f"pressure_{i}")
                        pitch = st.selectbox(f"Pitch Suitability {i+1}", pitch_map.keys(), key=f"pitch_{i}")
                        matchup = st.slider(f"Opposition Match-up Rating {i+1}", 0.0, 10.0, 7.0, key=f"matchup_{i}")
                        captaincy = st.slider(f"Captaincy Score {i+1}", 0.0, 10.0, 5.0, key=f"captaincy_{i}")
                        format = st.text_input(f"Format {i+1}", key=f"format_{i}")
                        avg = st.text_input(f"Avg {i+1}", key=f"avg_{i}")
                        sr = st.text_input(f"SR {i+1}", key=f"sr_{i}")
                        fifty = st.text_input(f"50s {i+1}", key=f"fifty_{i}")
                        century = st.text_input(f"100s {i+1}", key=f"century_{i}")
                        fame = st.text_input(f"Fame_index {i+1}", key=f"fame_{i}")
                        endorse = st.text_input(f"Endorsement_score {i+1}", key=f"endorse_{i}")
                        perf = st.text_input(f"Performance_score {i+1}", key=f"perf_{i}")
                        bias = st.text_input(f"Bias_score {i+1}", key=f"bias_{i}")
                        is_biased = st.checkbox(f"Is_Biased {i+1}", key=f"isbiased_{i}")

                        manual_players.append({
                            "Player Name": name, "Batting Position": batting_pos, "Primary Role": role, "Recent Matches": matches,
                            "Runs": runs, "Balls Faced": balls, "Wickets": wickets, "Overs Bowled": overs, "Bowling Economy": econ,
                            "Batting Phase Strength": bat_phase, "Bowling Phase Strength": bowl_phase, "Match Impact Rating": impact,
                            "Fitness Status": fitness, "Pressure Performance": pressure, "Pitch Suitability": pitch,
                            "Opposition Match-up Rating": matchup, "Captaincy Score": captaincy, "Format": format, "Avg": avg,
                            "SR": sr, "50s": fifty, "100s": century, "Pressure_score": pressure, "Fame_index": fame,
                            "Endorsement_score": endorse, "Performance_score": perf, "Bias_score": bias, "Is_Biased": is_biased
                        })

                if st.button("✅ Add & Recalculate XI"):
                    new_df = pd.DataFrame(manual_players)
                    combined_df = pd.concat([df, new_df], ignore_index=True)
                    combined_df = compute_scores(combined_df)
                    final_xi = combined_df.sort_values("Overall Contribution Index", ascending=False).head(11)
                    final_xi["Selected in Final XI"] = True

                    st.success("✅ Final XI Created After Replacements")
                    st.dataframe(final_xi)

                    csv_final = final_xi.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Final XI (After Replacement)", csv_final, "Final_Unbiased_XI_After_Replacement.csv", "text/csv")

        else:
            missing_cols = [col for col in impact_required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n- " + "\n- ".join(missing_cols))



     # ----------------------- Role Balance Auditor --------------------- #

elif selected_feature == "Role Balance Auditor":
    st.subheader("⚖️ Role Balance Auditor (With Role Distribution & Alerts)")

    role_file = st.file_uploader("📂 Upload CSV with Player Roles", type="csv")

    if role_file:
        df = pd.read_csv(role_file)

        required_columns = ["Player Name", "Primary Role", "Batting Position", "Format"]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns.")

            # Recommended min and max for roles
            role_limits = {
                "Opener": (2, 3),
                "Anchor": (1, 2),
                "Floater": (1, 2),
                "Finisher": (1, 2),
                "All-rounder": (1, 3),
                "Spinner": (1, 2),
                "Fast Bowler": (2, 3),
                "Death Specialist": (1, 2)
            }

            # Count role occurrences
            role_counts = df["Primary Role"].value_counts().reset_index()
            role_counts.columns = ["Primary Role", "Count"]

            # Balance Status
            def get_balance_status(role, count):
                if role not in role_limits:
                    return "⚠️ Unknown Role"
                min_r, max_r = role_limits[role]
                if count < min_r:
                    return "⚠️ Too Few"
                elif count > max_r:
                    return "⚠️ Too Many"
                else:
                    return "✅ Balanced"

            role_counts["Balance Status"] = role_counts.apply(
                lambda row: get_balance_status(row["Primary Role"], row["Count"]), axis=1
            )

            # Merge with player data
            audit_df = df.merge(role_counts, on="Primary Role", how="left")

            # Reorder columns
            audit_df = audit_df[[
                "Player Name", "Primary Role", "Batting Position", "Format", "Count", "Balance Status"
            ]]

            # Display DataFrame
            st.subheader("📋 Role Balance Report")
            st.dataframe(audit_df)

            # Download option
            csv_data = audit_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Role Balance CSV",
                data=csv_data,
                file_name="role_balance_audit.csv",
                mime="text/csv"
            )

            # Charts
            st.subheader("📈 Role Distribution Overview")

            pie_chart = px.pie(
                role_counts, names="Primary Role", values="Count",
                title="Role Distribution in Current XI"
            )
            st.plotly_chart(pie_chart, use_container_width=True)

            bar_chart = px.bar(
                role_counts, x="Primary Role", y="Count", color="Balance Status", text_auto=True,
                title="Role Count with Balance Status"
            )
            st.plotly_chart(bar_chart, use_container_width=True)

        else:
            missing = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing))


# ------------------ RESET APP ------------------
st.markdown("---")
if st.button("🔄 Restart App"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()
