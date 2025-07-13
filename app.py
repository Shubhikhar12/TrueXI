import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
selected_feature = st.sidebar.radio("Select Feature", [
    "Main App Flow", 
    "XI Cohesion score", 
    "Pressure Heatmap XI", 
    "Tactical Role Analyzer", 
    "Impact-weighted index", 
    "Role Balance Auditor"
])

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align: center;'>🏏 Unbiased XI Selector App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Make Data-Driven Cricket Selections Without Bias</h4>", unsafe_allow_html=True)
if lottie_cricket:
    st_lottie(lottie_cricket, height=150, key="cricket_header")

# ------------------ MAIN APP FLOW ------------------
if selected_feature == "Main App Flow":

    # Step 0: Upload File
    if st.session_state.step == 0:
        uploaded_file = st.file_uploader("📁 Upload Final XI CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.dropna(inplace=True)
            st.session_state.df = df
            st.session_state.unbiased_xi_df = df
            st.session_state.step = 1
            st.success("✅ File uploaded successfully.")

    if st.session_state.df is not None:
        df = st.session_state.df

        # Show Player Data
        if st.button("Show Player Data"):
            st.subheader("📋 Uploaded Player Data")
            format_filter = st.selectbox("Filter by Format", options=df["Format"].unique())
            role_filter = st.multiselect("Filter by Role", options=df["Role"].unique(), default=df["Role"].unique())
            min_matches = st.slider("Minimum Matches", 0, 50, 0)
            filtered_df = df[(df["Format"] == format_filter) & (df["Role"].isin(role_filter)) & (df["Matches"] >= min_matches)]
            st.dataframe(filtered_df)

        # Detect Biased Players
        if st.button("Detect Biased Players"):
            st.subheader("🕵‍♂ Biased Players Detected")
            min_matches = st.slider("Minimum Matches for Bias Detection", 0, 50, 0)
            df = df[df["Matches"] >= min_matches]

            scaler = MinMaxScaler()
            df["Batting Avg (scaled)"] = scaler.fit_transform(df[["Batting Avg"]])
            df["Batting SR (scaled)"] = scaler.fit_transform(df[["Batting SR"]])
            df["Wickets (scaled)"] = scaler.fit_transform(df[["Wickets"]])
            df["Bowling Econ (scaled)"] = 1 - scaler.fit_transform(df[["Bowling Economy"]])

            def compute_performance(row):
                if row["Role"] == "Batter":
                    return row["Batting Avg (scaled)"] * 0.6 + row["Batting SR (scaled)"] * 0.4
                elif row["Role"] == "Bowler":
                    return row["Wickets (scaled)"] * 0.6 + row["Bowling Econ (scaled)"] * 0.4
                elif row["Role"] == "All-rounder":
                    batting = row["Batting Avg (scaled)"] * 0.5 + row["Batting SR (scaled)"] * 0.2
                    bowling = row["Wickets (scaled)"] * 0.2 + row["Bowling Econ (scaled)"] * 0.1
                    return batting + bowling
                return 0

            # --- Performance Score ---
            df["Performance_score_raw"] = df.apply(compute_performance, axis=1)
            df["Performance_score"] = scaler.fit_transform(df[["Performance_score_raw"]])

            # --- Fame Score ---
            fame_scaled = scaler.fit_transform(df[["Fame_index"]])
            endorse_scaled = scaler.fit_transform(df[["Endorsement_score"]])
            df["Fame_score"] = fame_scaled * 0.6 + endorse_scaled * 0.4

            # --- Bias Detection ---
            fame_threshold = df["Fame_score"].quantile(0.75)
            performance_threshold = df["Performance_score"].quantile(0.25)
            df["Is_Biased"] = (df["Fame_score"] > fame_threshold) & (df["Performance_score"] < performance_threshold)

            st.session_state.df = df

            st.dataframe(df[df["Is_Biased"]][["Player Name", "Role", "Fame_score", "Performance_score", "Is_Biased"]])

            # Optional Scatter Plot
            fig = px.scatter(df, x="Fame_score", y="Performance_score", color="Is_Biased",
                             hover_data=["Player Name", "Role"], title="Fame vs Performance Bias Map")
            st.plotly_chart(fig, use_container_width=True)

        # Generate Final Unbiased XI
        if st.button("Generate Final Unbiased XI"):
            st.subheader("🏆 Final Unbiased XI")
            df = st.session_state.df
            unbiased_df = df[df["Is_Biased"] == False]

            batters = unbiased_df[unbiased_df["Role"] == "Batter"].nlargest(5, "Performance_score")
            bowlers = unbiased_df[unbiased_df["Role"] == "Bowler"].nlargest(4, "Performance_score")
            allrounders = unbiased_df[unbiased_df["Role"] == "All-rounder"].nlargest(2, "Performance_score")

            final_xi = pd.concat([batters, bowlers, allrounders])
            st.dataframe(final_xi[["Player Name", "Role", "Performance_score", "Fame_score", "Is_Biased"]])

            csv = final_xi.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Final XI CSV", csv, "final_xi.csv", "text/csv")

# ------------------ XI COHESION SCORE ------------------
elif selected_feature == "XI Cohesion score":
    st.subheader("📘 XI Cohesion Score Analyzer")
    cohesion_file = st.file_uploader("📂 Upload CSV with XI Cohesion Columns", type="csv", key="cohesion_upload")

    if cohesion_file:
        df = pd.read_csv(cohesion_file)
        df.dropna(inplace=True)

        required_columns = [
            "Player Name", "Primary skill type", "Batting role", "Bowling role",
            "Batting position", "SR_PP", "SR_MO", "SR_DO", "Overs bowled_PP",
            "Overs bowled_MO", "Overs bowled_DO", "Opponent suitability score",
            "Matchup type", "Fielding rating", "Franchise/National team (for synergy calculation)",
            "Handedness batting", "Handedness bowling", "Partnership runs",
            "Wickets in tandem", "Matches played together", "Positional synergy index"
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing))
        else:
            df["Primary skill type"].fillna("Unknown", inplace=True)
            st.session_state['original_df'] = df.copy()

            st.success("✅ File processed with Cohesion Score logic.")
            st.markdown("Analyze how well your players know and support each other.")

            # Initial cohesion score
            cohesion_score = round(df["Positional synergy index"].mean(), 2)

            # Display block
            def display_cohesion_status(score):
                if score >= 8.5:
                    color, level, suggestion = "#2ecc71", "🟢 Excellent Cohesion", "✅ Maintain core squad. High familiarity."
                elif score >= 7.0:
                    color, level, suggestion = "#f1c40f", "🟡 Good Cohesion", "🔄 Minor role tweaks can help."
                elif score >= 5.0:
                    color, level, suggestion = "#e67e22", "🟠 Moderate Cohesion", "⚙ Strengthen partnerships and synergy."
                else:
                    color, level, suggestion = "#e74c3c", "🔴 Low Cohesion", "❌ Rework squad or focus on synergy."

                st.markdown(f"""
                    <div style='background-color: {color}; padding: 20px; border-radius: 10px; color: white; font-size: 20px;'>
                        <b>{level}</b><br>
                        Cohesion Score: <b>{score}</b><br>
                        {suggestion}
                    </div>
                """, unsafe_allow_html=True)

            display_cohesion_status(cohesion_score)

            st.markdown("### 📊 XI Cohesion Score Visualization")
            st.progress(min(cohesion_score / 10, 1.0))

            st.markdown("### 🧩 Player Synergy Table")
            st.dataframe(df[[
                "Player Name", "Partnership runs", "Wickets in tandem",
                "Matches played together", "Positional synergy index"
            ]])

            fig = px.bar(
                df, x="Player Name", y="Positional synergy index",
                color="Primary skill type", title="Synergy Index per Player"
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Cohesion CSV", csv_out, "cohesion_score_output.csv", "text/csv")

            # Manual adjustments
            if cohesion_score < 5.0:
                st.warning("📝 Low cohesion detected. You can manually tweak your XI below.")

                with st.expander("🛠 Modify XI Manually"):
                    # 1️⃣ Remove a player
                    player_to_remove = st.selectbox("❌ Select player to remove", options=df["Player Name"].unique(), key="remove_player")
                    if st.button("Remove Selected Player"):
                        df = df[df["Player Name"] != player_to_remove]
                        st.session_state['modified_df'] = df.copy()
                        st.success(f"✅ Removed {player_to_remove} from XI.")

                    st.divider()

                    # 2️⃣ Add a new player
                    st.markdown("### ➕ Add New Player Manually")
                    player_name = st.text_input("Player Name", key="new_player_name")
                    synergy_index = st.slider("Positional synergy index", 0.0, 10.0, 5.0)
                    matches_together = st.number_input("Matches played together", min_value=0)
                    partnership_runs = st.number_input("Partnership runs", min_value=0)
                    wickets_tandem = st.number_input("Wickets in tandem", min_value=0)
                    skill_type = st.selectbox("Primary skill type", ["Batter", "Bowler", "All-rounder"])

                    if st.button("Add New Player"):
                        new_player = pd.DataFrame([{
                            "Player Name": player_name,
                            "Primary skill type": skill_type,
                            "Batting role": "N/A", "Bowling role": "N/A", "Batting position": "N/A",
                            "SR_PP": 0, "SR_MO": 0, "SR_DO": 0,
                            "Overs bowled_PP": 0, "Overs bowled_MO": 0, "Overs bowled_DO": 0,
                            "Opponent suitability score": 0, "Matchup type": "N/A",
                            "Fielding rating": 0, "Franchise/National team (for synergy calculation)": "N/A",
                            "Handedness batting": "N/A", "Handedness bowling": "N/A",
                            "Partnership runs": partnership_runs,
                            "Wickets in tandem": wickets_tandem,
                            "Matches played together": matches_together,
                            "Positional synergy index": synergy_index
                        }])

                        # Add player to existing DataFrame
                        if 'modified_df' in st.session_state:
                            st.session_state['modified_df'] = pd.concat([st.session_state['modified_df'], new_player], ignore_index=True)
                        else:
                            st.session_state['modified_df'] = pd.concat([df, new_player], ignore_index=True)

                        st.success(f"✅ {player_name} added to the XI.")

                    st.divider()

                    # 3️⃣ Recalculate
                    if st.button("🔄 Recalculate Cohesion After Changes"):
                        updated_df = st.session_state.get('modified_df', df)
                        new_score = round(updated_df["Positional synergy index"].mean(), 2)

                        display_cohesion_status(new_score)

                        st.markdown("### 📊 Updated XI Synergy Table")
                        st.dataframe(updated_df[[
                            "Player Name", "Partnership runs", "Wickets in tandem",
                            "Matches played together", "Positional synergy index"
                        ]])

                        fig_updated = px.bar(
                            updated_df, x="Player Name", y="Positional synergy index",
                            color="Primary skill type", title="Updated Synergy Index per Player"
                        )
                        st.plotly_chart(fig_updated, use_container_width=True)

                        csv_out_updated = updated_df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇ Download Updated CSV", csv_out_updated, "updated_cohesion_score.csv", "text/csv")
    else:
        st.info("📁 Please upload a CSV file with the required cohesion metrics to continue.")


        # ------------------ PRESSURE HEATMAP XI ------------------
elif selected_feature == "Pressure Heatmap XI":
    st.subheader("🔥 Pressure Heatmap XI")
    pressure_file = st.file_uploader("📂 Upload CSV with Pressure Metrics", type="csv", key="pressure_upload")

    if pressure_file:
        df = pd.read_csv(pressure_file)

        required_cols = ["Player Name", "Role", "Performance_score", "Pressure_score"]
        if all(col in df.columns for col in required_cols):
            np.random.seed(42)

            phase_options = ["Powerplay", "Middle Overs", "Death Overs"]
            situation_options = ["Chasing", "Defending", "Clutch Moments"]

            # Randomly assign phase & match situations
            df["Phase Suitability"] = np.random.choice(phase_options, size=len(df))
            df["Match Situation"] = np.random.choice(situation_options, size=len(df))

            # Categorize Pressure Zone
            df["Pressure Zone"] = pd.cut(df["Pressure_score"], bins=[0, 0.55, 0.65, 1],
                                         labels=["Low", "Medium", "High"])

            # Calculate Impact Rating
            df["Impact Rating"] = round((df["Performance_score"] * 0.6 + df["Pressure_score"] * 0.4) * 10, 2)

            st.success("✅ File processed with Pressure Heatmap logic.")
            st.markdown("Visualize how your players perform under pressure situations and phases.")

            # 📊 Heatmap Visualization
            st.markdown("### 📊 Pressure Heatmap by Situation & Zone")
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

            # 🏏 Top XI Under Pressure
            st.markdown("### 💪 Top XI Under High Pressure")
            top_xi = df[df["Pressure Zone"] == "High"].sort_values(
                by="Impact Rating", ascending=False
            ).head(11)

            st.dataframe(top_xi[[
                "Player Name", "Role", "Phase Suitability", "Match Situation", "Impact Rating"
            ]])

            # ⬇ Download Button
            csv_out = top_xi.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Pressure Heatmap XI", csv_out, "pressure_heatmap_xi.csv", "text/csv")
        else:
            st.error("❌ Missing required columns: 'Player Name', 'Role', 'Performance_score', 'Pressure_score'")
    else:
        st.info("📁 Please upload a CSV file with required pressure metrics to continue.")


        # ------------------ TACTICAL ROLE ANALYZER ------------------
elif selected_feature == "Tactical Role Analyzer":
    st.subheader("📝 Tactical Role Analyzer (With Tactical Role Assignment & Insights)")

    tactical_file = st.file_uploader("📂 Upload CSV with Tactical Metrics", type="csv", key="tactical_upload")

    if tactical_file:
        df = pd.read_csv(tactical_file)

        required_columns = [
            "Player Name", "Role", "batting_hand", "PP_SR", "MO_SR", "DO_SR",
            "SR_vs_spin", "SR_vs_pace", "bowling_style", "PP_ER", "MO_ER",
            "DO_ER", "wicket_taking_percentage", "ER_vs_LHB", "ER_vs_RHB"
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing))
        else:
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

            # Tactical Role Fit Score (Random)
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
        st.info("📁 Please upload a CSV file with tactical metrics to continue.")


        # ------------------ IMPACT-WEIGHTED CONTRIBUTION VALIDATOR ------------------
elif selected_feature == "Impact-weighted index":
    st.subheader("📊 Impact-Weighted Contribution Validator")

    impact_file = st.file_uploader("📂 Upload Unbiased XI CSV (with input columns)", type="csv", key="impact_upload")

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

    def compute_scores(input_df):
        input_df["Balls Faced"] = input_df["Balls Faced"].replace(0, 1)
        input_df["Impact Score"] = ((input_df["Runs"] / input_df["Balls Faced"]) * input_df["Match Impact Rating"]).round(2)
        input_df["Role Match Score"] = input_df["Primary Role"].map(role_score_map).fillna(7.0)

        def generate_phase_tag(phase):
            tags = {"Powerplay": "PP", "Middle": "M", "Death": "D"}
            return ", ".join([f"{t}✔" if phase == p else f"{t}❌" for p, t in tags.items()])

        input_df["Phase Balance Tag"] = input_df["Batting Phase Strength"].apply(generate_phase_tag)
        input_df["Match-up Alert"] = input_df["Opposition Match-up Rating"].apply(lambda x: "✅" if x >= 7 else "⚠")
        anchor_count = input_df["Primary Role"].value_counts().get("Anchor", 0)
        input_df["Role Balance Alert"] = ["❌ Too Many Anchors" if anchor_count > 2 else "✅ Balanced"] * len(input_df)
        input_df["Captaincy Validation"] = input_df["Captaincy Score"].apply(lambda x: "✅" if x >= 7 else "❌")
        input_df["Pitch Fit Score"] = input_df["Pitch Suitability"].map(pitch_map).fillna(7.0)

        input_df["Overall Contribution Index"] = (
            (input_df["Impact Score"] + input_df["Role Match Score"] + input_df["Pitch Fit Score"]) / 3
        ).round(2)

        input_df["Selected in Final XI"] = input_df["Overall Contribution Index"] >= 8.0
        return input_df

    if impact_file:
        df = pd.read_csv(impact_file)

        missing_cols = [col for col in impact_required_columns if col not in df.columns]
        if missing_cols:
            st.error("❌ Missing required columns:\n- " + "\n- ".join(missing_cols))
        else:
            df = df[[col for col in impact_required_columns]]
            df = compute_scores(df)

            st.subheader("📋 Evaluated Player Table with Impact Metrics")
            st.dataframe(df)

            selected_xi = df[df["Selected in Final XI"] == True]

            st.markdown("### ✅ Selected Players Based on Contribution Index")
            st.dataframe(selected_xi[[
                "Player Name", "Primary Role", "Impact Score", "Role Match Score", "Pitch Fit Score",
                "Overall Contribution Index", "Selected in Final XI"
            ]])

            if len(selected_xi) < 11:
                st.warning(f"⚠ Only {len(selected_xi)} players selected. Need {11 - len(selected_xi)} more.")
                st.markdown("### ✍ Add Remaining Players Manually")

                num_to_add = 11 - len(selected_xi)
                manual_entries = []

                for i in range(num_to_add):
                    with st.expander(f"🧍 Player {i+1} Details"):
                        player = {
                            "Player Name": st.text_input(f"Name {i+1}"),
                            "Batting Position": st.text_input(f"Batting Position {i+1}"),
                            "Primary Role": st.selectbox(f"Primary Role {i+1}", list(role_score_map.keys())),
                            "Recent Matches": st.number_input(f"Recent Matches {i+1}", 0),
                            "Runs": st.number_input(f"Runs {i+1}", 0),
                            "Balls Faced": st.number_input(f"Balls Faced {i+1}", 1),
                            "Wickets": st.number_input(f"Wickets {i+1}", 0),
                            "Overs Bowled": st.number_input(f"Overs Bowled {i+1}", 0.0),
                            "Bowling Economy": st.number_input(f"Bowling Economy {i+1}", 0.0),
                            "Batting Phase Strength": st.selectbox(f"Batting Phase Strength {i+1}", ["Powerplay", "Middle", "Death"]),
                            "Bowling Phase Strength": st.selectbox(f"Bowling Phase Strength {i+1}", ["Powerplay", "Middle", "Death"]),
                            "Match Impact Rating": st.number_input(f"Match Impact Rating {i+1}", 0.0),
                            "Fitness Status": st.selectbox(f"Fitness Status {i+1}", ["Fit", "Unfit"]),
                            "Pressure Performance": st.number_input(f"Pressure Performance {i+1}", 0.0),
                            "Pitch Suitability": st.selectbox(f"Pitch Suitability {i+1}", list(pitch_map.keys())),
                            "Opposition Match-up Rating": st.number_input(f"Opposition Match-up Rating {i+1}", 0.0),
                            "Captaincy Score": st.number_input(f"Captaincy Score {i+1}", 0.0),
                            "Format": st.selectbox(f"Format {i+1}", ["ODI", "T20", "Test"]),
                            "Avg": st.number_input(f"Avg {i+1}", 0.0),
                            "SR": st.number_input(f"SR {i+1}", 0.0),
                            "50s": st.number_input(f"50s {i+1}", 0),
                            "100s": st.number_input(f"100s {i+1}", 0),
                            "Pressure_score": st.number_input(f"Pressure_score {i+1}", 0.0),
                            "Fame_index": st.number_input(f"Fame_index {i+1}", 0.0),
                            "Endorsement_score": st.number_input(f"Endorsement_score {i+1}", 0.0),
                            "Performance_score": st.number_input(f"Performance_score {i+1}", 0.0),
                            "Bias_score": st.number_input(f"Bias_score {i+1}", 0.0),
                            "Is_Biased": st.selectbox(f"Is Biased {i+1}", ["Yes", "No"])
                        }
                        manual_entries.append(player)

                if st.button("📊 Evaluate Manually Entered Players"):
                    manual_df = pd.DataFrame(manual_entries)
                    manual_df = compute_scores(manual_df)
                    df = pd.concat([df, manual_df], ignore_index=True)
                    selected_xi = df[df["Selected in Final XI"] == True]

            if len(selected_xi) == 11:
                st.success("✅ All 11 players selected with high contribution scores.")

            csv_output = selected_xi.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Final XI", csv_output, "impact_weighted_final_xi.csv", "text/csv")

            st.markdown("### 📈 Contribution Index Visualization")
            fig = px.bar(
                df.sort_values("Overall Contribution Index", ascending=False),
                x="Player Name", y="Overall Contribution Index",
                color="Primary Role", text_auto=True,
                title="Overall Contribution Index per Player"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📁 Please upload a CSV file with required columns to begin analysis.")


        # ------------------ ROLE BALANCE AUDITOR ------------------
elif selected_feature == "Role Balance Auditor":
    st.subheader("⚖ Role Balance Auditor (With Role Distribution & Alerts)")

    role_file = st.file_uploader("📂 Upload CSV with Player Roles", type="csv", key="role_balance_upload")

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

            role_counts = df["Primary Role"].value_counts().reset_index()
            role_counts.columns = ["Primary Role", "Count"]

            # Role balance checker
            def get_balance_status(role, count):
                if role not in role_limits:
                    return "⚠ Unknown Role"
                min_r, max_r = role_limits[role]
                if count < min_r:
                    return "⚠ Too Few"
                elif count > max_r:
                    return "⚠ Too Many"
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

            # 📋 Display Data
            st.subheader("📋 Role Balance Report")
            st.dataframe(audit_df)

            # ⬇ Download
            csv_data = audit_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Role Balance CSV",
                data=csv_data,
                file_name="role_balance_audit.csv",
                mime="text/csv"
            )

            # 📊 Charts
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
    else:
        st.info("📁 Please upload a CSV file with roles to continue.")


# ------------------ RESET APP ------------------
st.markdown("---")
if st.button("🔄 Restart App"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()
