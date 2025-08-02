# --------------------------------------------------
# TrueXI App
# Author: Nihira Khare
# Date: July 2025
# --------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="TrueXI Selector", layout="wide")

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

# ------------------ STYLING ------------------
st.markdown("""
    <style>
        .stApp { background-color: #0b132b; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3, h4 { color: #ffd700; text-align: center; }
        section[data-testid="stSidebar"] { background-color: #1c2541; color: white; border-right: 2px solid #2a9d8f; }
        .css-1d391kg { color: #fcbf49 !important; }
        .css-1cpxqw2 { color: #ffffff !important; }
        .stButton > button { background-color: #2a9d8f; color: white; font-weight: bold; border-radius: 12px; padding: 10px 20px; transition: background-color 0.3s ease; }
        .stButton > button:hover { background-color: #21867a; }
        .stDownloadButton > button { background-color: #e63946; color: white; font-weight: bold; border-radius: 12px; }
        .stDownloadButton > button:hover { background-color: #d62828; }
        .stAlert { border-radius: 10px; }
        .stDataFrame { border: 2px solid #2a9d8f; border-radius: 12px }
        .stTextInput, .stSelectbox, .stSlider { border-radius: 10px !important; }
        iframe[title="cricket_header"] { display: block; margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("📊 Unbiased XI Tools")
selected_feature = st.sidebar.radio("Select Feature", [
    "Main App Flow",
    "Pressure Heatmap XI",
    "Role Balance Auditor",
    "Pitch Adaptive XI Selector"
])

# ------------------ HEADER ------------------
st.image("app logo.png", width=150)
st.markdown("<h1>🏏 TrueXI Selector App</h1>", unsafe_allow_html=True)
st.markdown("<h4>Make Data-Driven Cricket Selections Without Bias</h4>", unsafe_allow_html=True)
if lottie_cricket:
    st_lottie(lottie_cricket, height=150, key="cricket_header")

    # ------------------ UTILITY FUNCTIONS ------------------
def safe_scale(column):
    if len(np.unique(column)) > 1:
        return MinMaxScaler().fit_transform(column.values.reshape(-1, 1))
    return np.full_like(column.values, 0.5).reshape(-1, 1)

def compute_performance(row):
    role = row["Primary Role"].strip().lower()
    if role in ["batter", "wk-batter"]:
        return row["Batting Avg (scaled)"] * 0.6 + row["Batting SR (scaled)"] * 0.4
    elif role == "bowler":
        return row["Wickets (scaled)"] * 0.6 + row["Bowling Econ (scaled)"] * 0.4
    elif role == "all-rounder":
        batting = row["Batting Avg (scaled)"] * 0.3 + row["Batting SR (scaled)"] * 0.2
        bowling = row["Wickets (scaled)"] * 0.3 + row["Bowling Econ (scaled)"] * 0.2
        return batting + bowling
    return 0

def calculate_leadership_score(df):
    df = df.copy()
    df["Leadership_Score"] = 0.6 * df["Performance_score"] + 0.4 * df["Fame_score"]
    return df


# ------------------ MAIN APP FLOW ------------------
if selected_feature == "Main App Flow":

    if st.session_state.step == 0:
        uploaded_file = st.file_uploader("\U0001F4C1 Upload Final XI CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.dropna(inplace=True)
            df["Primary Role"] = df["Primary Role"].astype(str).str.strip().str.lower()
            required_columns = [
                "Player Name", "Primary Role", "Format", "Batting Avg", "Batting SR",
                "Wickets", "Bowling Economy", "Google Trends Score", "Social Media Reach"
            ]
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                st.session_state.df = df
                st.session_state.step = 1
                st.success("✅ File uploaded successfully.")

    if st.session_state.df is not None:
        df = st.session_state.df

        if st.button("Show Player Data"):
            st.subheader("\U0001F4CB Uploaded Player Data")
            format_filter = st.selectbox("Filter by Format", options=df["Format"].unique())
            role_filter = st.multiselect("Filter by Primary Role", options=df["Primary Role"].unique(), default=df["Primary Role"].unique())
            filtered_df = df[(df["Format"] == format_filter) & (df["Primary Role"].isin(role_filter))]
            st.dataframe(filtered_df)

        if st.button("Detect Biased Players"):
            st.subheader("\U0001F575️‍♂ Biased Players Detected")

            format_filter = st.selectbox("\U0001F3AF Choose Format for Bias Detection", df["Format"].unique())
            df = df[df["Format"] == format_filter].copy()

            df["Batting Avg (scaled)"] = safe_scale(df["Batting Avg"])
            df["Batting SR (scaled)"] = safe_scale(df["Batting SR"])
            df["Wickets (scaled)"] = safe_scale(df["Wickets"])
            df["Bowling Econ (scaled)"] = 1 - safe_scale(df["Bowling Economy"])

            df["Performance_score_raw"] = df.apply(compute_performance, axis=1)
            df["Performance_score"] = safe_scale(df["Performance_score_raw"])
            df["Google Trends (scaled)"] = safe_scale(df["Google Trends Score"])
            df["Social Media Reach (scaled)"] = safe_scale(df["Social Media Reach"])

            df["Fame_score"] = (
                df["Google Trends (scaled)"] * 0.5 +
                df["Social Media Reach (scaled)"] * 0.5 
            )

            df["bias_score"] = df["Fame_score"] - df["Performance_score"]

            fame_q3 = df["Fame_score"].quantile(0.75)
            perf_q1 = df["Performance_score"].quantile(0.25)
            margin = 0.05

            df["Is_Biased"] = (df["Fame_score"] > fame_q3 + margin) & (df["Performance_score"] < perf_q1 - margin)
            st.session_state.df = df

            st.dataframe(df[df["Is_Biased"]][[
                "Player Name", "Primary Role", "Fame_score", "Performance_score",
                "bias_score", "Is_Biased"
            ]])

            fig = px.scatter(df, x="Fame_score", y="Performance_score", color="Is_Biased",
                             hover_data=["Player Name", "Primary Role"],
                             title="Fame vs Performance Bias Map")
            fig.update_layout(
                xaxis_title="Fame Score",
                yaxis_title="Performance Score",
                legend_title="Bias Status"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 🔄 BEEHIVE / BEESWARM SECTION
            st.subheader("🐝 Beehive View: Performance by Primary Role")
            import random
            df["Jittered Role"] = df["Primary Role"] + df["Primary Role"].apply(lambda x: f"_{random.uniform(-0.3, 0.3):.2f}")

            # For cleaner layout, map to numerical values instead of jittered string
            role_mapping = {role: idx for idx, role in enumerate(df["Primary Role"].unique())}
            df["Primary Role_num"] = df["Primary Role"].map(role_mapping)
            df["Jittered_x"] = df["Primary Role_num"] + np.random.normal(0, 0.15, size=len(df))

            beehive_fig = px.scatter(
                df, x="Jittered_x", y="Performance_score",
                color="Primary Role", hover_data=["Player Name", "Fame_score", "Is_Biased"],
                title="Simulated Beehive Plot: Performance Score by Primary Role"
            )
            beehive_fig.update_layout(
                xaxis=dict(
                    tickvals=list(role_mapping.values()),
                    ticktext=list(role_mapping.keys()),
                    title="Primary Role"
                ),
                yaxis_title="Performance Score",
                showlegend=True
            )
            st.plotly_chart(beehive_fig, use_container_width=True)


        if st.button("Generate Final Unbiased XI"):
            st.subheader("\U0001F3C6 Final Unbiased XI")
            df = st.session_state.df
            unbiased_df = df[df["Is_Biased"] == False].copy()

            wk_batter = None
            wk_unbiased = unbiased_df[unbiased_df["Primary Role"] == "wk-batter"].copy()
            if not wk_unbiased.empty:
                wk_unbiased = calculate_leadership_score(wk_unbiased)
                wk_batter = wk_unbiased.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).head(1)
                st.info(f"✅ WK-Batter selected from unbiased list: {wk_batter.iloc[0]['Player Name']}")
            else:
                wk_all = df[df["Primary Role"] == "wk-batter"].copy()
                if not wk_all.empty:
                    wk_all = calculate_leadership_score(wk_all)
                    wk_batter = wk_all.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).head(1)
                    st.warning(f"⚠ No unbiased WK-Batter found. Selected best available: {wk_batter.iloc[0]['Player Name']}")
                else:
                    st.error("❌ No WK-Batter found in dataset.")

            remaining_pool = unbiased_df[~unbiased_df["Player Name"].isin(wk_batter["Player Name"])]
            batters = remaining_pool[remaining_pool["Primary Role"] == "batter"].nlargest(4, "Performance_score")
            bowlers = remaining_pool[remaining_pool["Primary Role"] == "bowler"].nlargest(4, "Performance_score")
            allrounders = remaining_pool[remaining_pool["Primary Role"] == "all-rounder"].nlargest(2, "Performance_score")

            final_xi = pd.concat([wk_batter, batters, bowlers, allrounders]).drop_duplicates("Player Name").head(11)
            final_xi = calculate_leadership_score(final_xi)

            final_xi = final_xi.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False)
            final_xi["Captain"] = False
            final_xi["Vice_Captain"] = False
            final_xi.iloc[0, final_xi.columns.get_loc("Captain")] = True
            final_xi.iloc[1, final_xi.columns.get_loc("Vice_Captain")] = True

            st.session_state.final_xi = final_xi
            st.dataframe(final_xi[[
                "Player Name", "Primary Role", "Performance_score", "Fame_score", "Is_Biased", "Captain", "Vice_Captain"
            ]])

            csv = final_xi[[
                "Player Name", "Primary Role", "Performance_score", "Fame_score", "Captain", "Vice_Captain"
            ]].to_csv(index=False).encode("utf-8")

            st.download_button("⬇ Download Final XI CSV", csv, "final_xi.csv", "text/csv")

            captain = final_xi[final_xi["Captain"]].iloc[0]
            vice_captain = final_xi[final_xi["Vice_Captain"]].iloc[0]

            st.success(f"🏏 Recommended Captain: {captain['Player Name']} | Leadership Score: {captain['Leadership_Score']:.2f}")
            st.info(f"🥢 Vice-Captain: {vice_captain['Player Name']} | Leadership Score: {vice_captain['Leadership_Score']:.2f}")

            if "rohit sharma" in final_xi["Player Name"].str.lower().values and captain["Player Name"].lower() != "rohit sharma":
                rohit_score = final_xi[final_xi["Player Name"].str.lower() == "rohit sharma"]["Leadership_Score"].values[0]
                st.warning(f"⚠ Rohit Sharma is the current captain, but **{captain['Player Name']}** has a higher Leadership Score ({captain['Leadership_Score']:.2f}) vs Rohit's ({rohit_score:.2f}).")

        if "final_xi" in st.session_state:
            st.markdown("---")
            st.subheader("✍ Select Future Leadership Manually")
            with st.form("manual_leadership_form"):
                manual_candidates = st.multiselect(
                    "Select at least 2 players from the Unbiased XI for custom captain & vice-captain evaluation:",
                    options=st.session_state.final_xi["Player Name"].tolist()
                )
                submitted = st.form_submit_button("\U0001F9E0 Calculate Leadership")

            if submitted:
                if len(manual_candidates) >= 2:
                    manual_df = st.session_state.final_xi[
                        st.session_state.final_xi["Player Name"].isin(manual_candidates)
                    ].copy()

                    manual_df = calculate_leadership_score(manual_df)
                    manual_df = manual_df.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False)
                    manual_df["Captain"] = False
                    manual_df["Vice_Captain"] = False
                    manual_df.iloc[0, manual_df.columns.get_loc("Captain")] = True
                    manual_df.iloc[1, manual_df.columns.get_loc("Vice_Captain")] = True

                    st.success(f"🥢 Manually Selected Captain: {manual_df.iloc[0]['Player Name']} | Leadership Score: {manual_df.iloc[0]['Leadership_Score']:.2f}")
                    st.info(f"🎖 Manually Selected Vice-Captain: {manual_df.iloc[1]['Player Name']} | Leadership Score: {manual_df.iloc[1]['Leadership_Score']:.2f}")

                    st.dataframe(manual_df[[
                        "Player Name", "Primary Role", "Performance_score", "Fame_score",
                        "Leadership_Score", "Captain", "Vice_Captain"
                    ]])

                    manual_csv = manual_df[[
                        "Player Name", "Primary Role", "Performance_score", "Fame_score", "Captain", "Vice_Captain"
                    ]].to_csv(index=False).encode("utf-8")

                    st.download_button("⬇ Download Manual Captain-Vice CSV", manual_csv, "manual_captain_vice.csv", "text/csv")
                else:
                    st.warning("👥 Please select at least 2 players to perform leadership calculation.")

    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)


# ------------------ PRESSURE HEATMAP XI ------------------
elif selected_feature == "Pressure Heatmap XI":
    st.subheader("🔥 Pressure Heatmap XI")
    pressure_file = st.file_uploader("📂 Upload CSV with Pressure Metrics", type="csv", key="pressure_upload")

    if pressure_file:
        df = pd.read_csv(pressure_file)

        st.write("📊 Uploaded Data Preview:")
        st.dataframe(df.head())

        required_cols = ["Player Name", "Primary Role", "Death Overs Perf", "Clutch Game Perf", "Pressure Fielding", "Win Clutch Impact"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        df.fillna(0, inplace=True)

        # 🎯 Role-specific pressure score calculation
        def calculate_pressure_score(row):
            role = row["Primary Role"].lower()
            if role == "bowler":
                return round(
                    row["Death Overs Perf"] * 0.4 +
                    row["Clutch Game Perf"] * 0.3 +
                    row["Pressure Fielding"] * 0.2 +
                    row["Win Clutch Impact"] * 0.1, 3)
            elif role in ["finisher"]:
                return round(
                    row["Death Overs Perf"] * 0.4 +
                    row["Clutch Game Perf"] * 0.3 +
                    row["Pressure Fielding"] * 0.2 +
                    row["Win Clutch Impact"] * 0.1, 3)
            elif role in ["anchor", "opener"]:
                return round(
                    row["Clutch Game Perf"] * 0.4 +
                    row["Pressure Fielding"] * 0.3 +
                    row["Win Clutch Impact"] * 0.3, 3)
            elif role in ["floater"]:
                return round(
                    row["Clutch Game Perf"] * 0.35 +
                    row["Death Overs Perf"] * 0.25 +
                    row["Pressure Fielding"] * 0.25 +
                    row["Win Clutch Impact"] * 0.15, 3)
            else:  # All-rounder or undefined
                return round(
                    row["Death Overs Perf"] * 0.3 +
                    row["Clutch Game Perf"] * 0.3 +
                    row["Pressure Fielding"] * 0.2 +
                    row["Win Clutch Impact"] * 0.2, 3)

        df["Pressure_score"] = df.apply(calculate_pressure_score, axis=1)

        def assign_phase_suitability(row):
            role = row["Primary Role"].lower()
            score = row["Pressure_score"]
            if role == "bowler":
                if score >= 0.8:
                    return "Death Overs"
                elif score >= 0.6:
                    return "Middle Overs"
                else:
                    return "Powerplay"
            else:
                if score >= 0.8:
                    return "Powerplay"
                elif score >= 0.6:
                    return "Middle Overs"
                else:
                    return "Death Overs"

        def assign_match_situation(row):
            role = row["Primary Role"].lower()
            score = row["Pressure_score"]
            if score >= 0.75:
                return "Clutch Moments"
            elif role == "bowler":
                return "Defending" if score >= 0.6 else "Support Role"
            elif role in ["batter", "all-rounder"]:
                return "Chasing" if score >= 0.6 else "Anchor / Setup"
            else:
                return "Flexible"

        df["Phase Suitability"] = df.apply(assign_phase_suitability, axis=1)
        df["Match Situation"] = df.apply(assign_match_situation, axis=1)
        df["Pressure Zone"] = pd.cut(df["Pressure_score"], bins=[0, 0.55, 0.65, 1], labels=["Low", "Medium", "High"])
        df["Impact Rating"] = round(df["Pressure_score"] * 10, 2)

        st.success("✅ File processed with updated Pressure Heatmap logic.")

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

        st.markdown("### 🐝 Beehive View: Impact Rating by Role")
        import numpy as np
        role_map = {r: i for i, r in enumerate(df["Primary Role"].unique())}
        df["Jitter_x"] = df["Primary Role"].map(role_map) + np.random.normal(0, 0.15, size=len(df))

        fig_bee = px.scatter(
            df,
            x="Jitter_x",
            y="Impact Rating",
            color="Pressure Zone",
            hover_data=["Player Name", "Primary Role", "Match Situation"],
            title="Beehive Plot: Impact Rating by Role",
        )
        fig_bee.update_layout(
            xaxis=dict(
                tickvals=list(role_map.values()),
                ticktext=list(role_map.keys()),
                title="Role"
            ),
            yaxis_title="Impact Rating",
            showlegend=True
        )
        st.plotly_chart(fig_bee, use_container_width=True)

        st.markdown("### 💪 Top XI Under High Pressure")
        top_xi = df[df["Pressure Zone"] == "High"].sort_values(by="Impact Rating", ascending=False).head(11)
        st.dataframe(top_xi[["Player Name", "Primary Role", "Phase Suitability", "Match Situation", "Impact Rating"]])

        st.markdown("### ❌ Players Not Selected in Top XI (High Pressure)")
        picked_names = top_xi["Player Name"].tolist()
        unpicked_df = df[(df["Pressure Zone"] == "High") & (~df["Player Name"].isin(picked_names))]

        if not unpicked_df.empty:
            st.dataframe(unpicked_df[["Player Name", "Primary Role", "Impact Rating"]])
        else:
            st.info("✅ All High Pressure players are in the Top XI.")

        st.markdown("### ✍️ Add Manual Player(s) to XI")
        manual_players = []
        manual_validated = False

        with st.expander("➕ Add Player Manually"):
            num_manual = st.number_input("How many manual players do you want to add?", min_value=0, max_value=20, step=1)

            for i in range(num_manual):
                st.markdown(f"#### Player {i+1}")
                name = st.text_input(f"Player Name {i+1}", key=f"name_{i}")
                role = st.selectbox(f"Primary Role {i+1}", ["Batter", "Bowler", "All-rounder", "Wicketkeeper", "Anchor", "Opener", "Finisher", "Floater"], key=f"role_{i}")
                death_perf = st.number_input("Death Overs Perf", 0.0, 1.0, 0.0, 0.01, key=f"death_{i}")
                clutch_perf = st.number_input("Clutch Game Perf", 0.0, 1.0, 0.0, 0.01, key=f"clutch_{i}")
                fielding = st.number_input("Pressure Fielding", 0.0, 1.0, 0.0, 0.01, key=f"field_{i}")
                win_impact = st.number_input("Win Clutch Impact", 0.0, 1.0, 0.0, 0.01, key=f"win_{i}")

                manual_row = {
                    "Primary Role": role.lower(),
                    "Death Overs Perf": death_perf,
                    "Clutch Game Perf": clutch_perf,
                    "Pressure Fielding": fielding,
                    "Win Clutch Impact": win_impact
                }
                pressure_score = calculate_pressure_score(manual_row)
                impact_rating = round(pressure_score * 10, 2)

                phase = "Death Overs" if pressure_score >= 0.8 else "Middle Overs" if pressure_score >= 0.6 else "Powerplay"
                match_sit = "Clutch Moments" if pressure_score >= 0.75 else ("Defending" if role.lower() == "bowler" and pressure_score >= 0.6 else "Chasing")

                manual_players.append({
                    "Player Name": name,
                    "Primary Role": role,
                    "Pressure_score": pressure_score,
                    "Phase Suitability": phase,
                    "Match Situation": match_sit,
                    "Pressure Zone": "High",
                    "Impact Rating": impact_rating
                })

            if num_manual > 0 and st.button("✅ Calculate Pressure Validation"):
                validated_players = []
                for p in manual_players:
                    is_valid = (
                        p["Pressure_score"] >= 0.7 and
                        p["Impact Rating"] >= 6.5 and
                        p["Pressure Zone"] == "High"
                    )
                    status = "✅ Performs Under Pressure" if is_valid else "❌ Does Not Perform Under Pressure"
                    validated_players.append({
                        **p,
                        "Pressure Validation": status
                    })

                st.markdown("### 🔍 Manual Players Pressure Validation")
                st.dataframe(pd.DataFrame(validated_players)[[ "Player Name", "Primary Role", "Pressure_score", "Impact Rating", "Pressure Validation" ]])
                manual_players = validated_players
                manual_validated = True

        combined_df = pd.concat([top_xi, pd.DataFrame(manual_players)], ignore_index=True)

        st.markdown("### 📋 Final Pressure XI")
        st.dataframe(combined_df[["Player Name", "Primary Role", "Phase Suitability", "Match Situation", "Impact Rating"]])

        csv_out = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Final Pressure Heatmap XI", csv_out, "final_pressure_heatmap_xi.csv", "text/csv")

    else:
        st.info("📁 Please upload a CSV file with required pressure metrics to continue.")

    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)

# ------------------ ROLE BALANCE AUDITOR ------------------
elif selected_feature == "Role Balance Auditor":
    st.subheader("⚖ Role Balance Auditor (With Role Distribution & Alerts)")

    role_file = st.file_uploader("📂 Upload CSV with Player Roles", type="csv", key="role_balance_upload")

    if role_file:
        df = pd.read_csv(role_file)

        required_columns = ["Player Name", "Role", "Batting Position", "Format"]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns.")

            # Recommended min and max for roles
            role_limits = {
                "Opener": (2, 3),
                "Anchor": (1, 2),
                "Floater": (1, 2),
                "Finisher": (1, 2),
                "Wk": (1, 1),
                "All-rounder": (1, 3),
                "Spinner": (1, 2),
                "Fast Bowler": (2, 3),
                "Death Specialist": (1, 2)
            }

            role_counts = df["Role"].value_counts().reset_index()
            role_counts.columns = ["Role", "Count"]

            # Role balance checker with emoji
            def get_balance_status(role, count):
                if role not in role_limits:
                    return "⚠ Unknown Role"
                min_r, max_r = role_limits[role]
                if count < min_r:
                    return "🟠 Too Few"
                elif count > max_r:
                    return "🔴 Too Many"
                else:
                    return "🟢 Balanced"

            role_counts["Balance Status"] = role_counts.apply(
                lambda row: get_balance_status(row["Role"], row["Count"]), axis=1
            )

            # Merge with player data
            audit_df = df.merge(role_counts, on="Role", how="left")

            # Reorder columns
            audit_df = audit_df[[
                "Player Name", "Role", "Batting Position", "Format", "Count", "Balance Status"
            ]]

            # 📋 Display Data
            st.subheader("📋 Role Balance Report")
            st.dataframe(audit_df)

            # 🔍 Suggestions (optional enhancement)
            imbalanced_roles = role_counts[role_counts["Balance Status"] != "🟢 Balanced"]
            if not imbalanced_roles.empty:
                st.warning("🔍 Suggestions to improve balance:")
                for _, row in imbalanced_roles.iterrows():
                    role = row["Role"]
                    count = row["Count"]
                    min_r, max_r = role_limits.get(role, (0, 0))
                    if count < min_r:
                        st.markdown(f"- ➕ Consider *adding* more players for role: {role} (Current: {count}, Minimum Required: {min_r})")
                    elif count > max_r:
                        st.markdown(f"- ➖ Consider *removing* some players from role: {role} (Current: {count}, Maximum Allowed: {max_r})")

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
                role_counts, names="Role", values="Count",
                title="Role Distribution in Current XI"
            )
            st.plotly_chart(pie_chart, use_container_width=True)

            bar_chart = px.bar(
                role_counts, x="Role", y="Count", color="Balance Status", text_auto=True,
                title="Role Count with Balance Status"
            )
            st.plotly_chart(bar_chart, use_container_width=True)

            # 🐝 Beehive (Strip) Plot
            st.subheader("🐝 Beehive Plot (Role vs Batting Position by Format)")
            beehive = px.strip(
                df,
                x="Role",
                y="Batting Position",
                color="Format",
                hover_data=["Player Name"],
                stripmode="overlay",
                title="Beehive Plot of Player Roles and Batting Positions"
            )
            beehive.update_traces(jitter=0.35, marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(beehive, use_container_width=True)

        else:
            missing = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing))
    else:
        st.info("📁 Please upload a CSV file with roles to continue.")

    # --- Signature Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)


# ------------------ PITCH ADAPTIVE XI SELECTOR ------------------
elif selected_feature == "Pitch Adaptive XI Selector":
    st.subheader("🏟️ Pitch Adaptive XI Selector")

    uploaded_adaptive_file = st.file_uploader("📁 Upload Final XI CSV", type="csv", key="pitch_adaptive_upload")

    if uploaded_adaptive_file:
        df = pd.read_csv(uploaded_adaptive_file)

        if all(col in df.columns for col in ["Player Name", "Role", "Format"]):

            all_venues = {
                "Wankhede Stadium": "Red Soil",
                "Brabourne Stadium": "Red Soil",
                "MA Chidambaram Stadium": "Red Soil",
                "M. Chinnaswamy Stadium": "Red Soil",
                "Sawai Mansingh Stadium": "Red Soil",
                "Narendra Modi Stadium": "Black Soil",
                "Ekana Stadium": "Black Soil",
                "Arun Jaitley Stadium": "Black Soil",
                "Rajiv Gandhi Intl Stadium": "Black Soil",
                "Barsapara Stadium": "Black Soil",
                "Holkar Stadium": "Black Soil",
                "Eden Gardens": "Red Soil",
            }

            selected_venue = st.selectbox("📍 Select Match Venue", list(all_venues.keys()))
            pitch_type = all_venues[selected_venue]

            match_time = st.selectbox("🕐 Match Time", ["Day", "Night"])

            def recommend_toss_decision(pitch_type, match_time):
                if pitch_type == "Red Soil" and match_time == "Day":
                    return "Bat", "Red soil dries out under sunlight. Spinners get more turn later, making it harder to bat second."
                elif pitch_type == "Red Soil" and match_time == "Night":
                    return "Field", "Dew neutralizes spin at night on red soil. Easier to chase."
                elif pitch_type == "Black Soil" and match_time == "Day":
                    return "Field", "Black soil retains moisture early. Bowling first helps, then pitch slows for batting."
                elif pitch_type == "Black Soil" and match_time == "Night":
                    return "Bat", "Black soil gets harder and bouncier at night. Dew is less effective, better to set target."

            suggested_toss, toss_reason = recommend_toss_decision(pitch_type, match_time)

            st.markdown(f"🧱 **Pitch Type:** `{pitch_type}`")
            st.markdown(f"🕐 **Match Time:** `{match_time}`")
            st.markdown(f"🧽 **Toss Recommendation:** The captain should **opt to `{suggested_toss}` first** if they win the toss.")
            st.info(f"📌 **Reason:** {toss_reason}")

            def classify_player(row):
                role = row["Role"].lower()

                if pitch_type == "Red Soil" and match_time == "Day":
                    if any(x in role for x in ["spinner", "spin all-rounder", "anchor"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["opener", "floater"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"

                elif pitch_type == "Red Soil" and match_time == "Night":
                    if any(x in role for x in ["fast", "death", "finisher", "seamer", "floater"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["opener", "spinner", "spin all-rounder"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"

                elif pitch_type == "Black Soil" and match_time == "Day":
                    if any(x in role for x in ["fast", "seamer", "anchor", "pace-all-rounder"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["spinner", "opener"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"

                elif pitch_type == "Black Soil" and match_time == "Night":
                    if any(x in role for x in ["fast", "death", "finisher", "opener"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["spinner", "floater"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"

            df["Pitch Adaptiveness"] = df.apply(classify_player, axis=1)

            role_priority = {
                "opener": 0,
                "anchor": 0,
                "floater": 0,
                "finisher": 0,
                "spinner": 1,
                "fast": 1,
                "seamer": 1,
                "death": 1,
                "spin all-rounder": 2,
                "pace-all-rounder": 2,
                "all-rounder": 2,
            }

            def get_sort_priority(role):
                for keyword, priority in role_priority.items():
                    if keyword in role.lower():
                        return priority
                return 3

            df["Sort Priority"] = df["Role"].apply(get_sort_priority)
            df = df.sort_values(by="Sort Priority").drop(columns=["Sort Priority"])

            st.subheader("📋 Adaptive Classification")
            st.dataframe(df)

            import plotly.express as px
            st.subheader("🐝 Beehive View of Player Adaptiveness")
            fig = px.strip(
                df,
                x="Pitch Adaptiveness",
                y="Role",
                color="Pitch Adaptiveness",
                hover_name="Player Name",
                stripmode="overlay",
                title="Beehive Plot: Player Roles vs Pitch Suitability",
                height=500,
            )
            fig.update_traces(jitter=0.6, marker_size=12)
            st.plotly_chart(fig, use_container_width=True)

            not_ideal_players = df[df["Pitch Adaptiveness"] == "❌ Not Ideal"]

            if not not_ideal_players.empty:
                st.warning("❌ Some players are not ideal for selected pitch. Please suggest replacements.")
                replacements = {}
                for idx, row in not_ideal_players.iterrows():
                    player_name = row["Player Name"]
                    st.markdown(f"🔁 Replace `{player_name}`:")
                    new_name = st.text_input(f"New Player Name for {player_name}", key=f"replace_{idx}")
                    new_role = st.text_input(f"Role for {new_name}", key=f"role_{idx}")
                    new_format = st.selectbox(f"Format for {new_name}", ["ODI", "T20"], key=f"format_{idx}")
                    if new_name and new_role:
                        new_row = {
                            "Player Name": new_name,
                            "Role": new_role,
                            "Format": new_format
                        }
                        replacements[idx] = new_row

                if replacements:
                    for idx, new_data in replacements.items():
                        new_data["Pitch Adaptiveness"] = classify_player(new_data)
                        df.loc[idx] = new_data

            st.markdown("✅ Final Pitch Adaptive XI")
            st.dataframe(df)

            final_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Final Pitch Adaptive XI CSV",
                data=final_csv,
                file_name="final_pitch_adaptive_xi.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ Required columns missing: ['Player Name', 'Role', 'Format']")
    else:
        st.info("📂 Please upload the Final XI CSV to proceed.")

    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)