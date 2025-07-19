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
    "Pressure Heatmap XI",  
    "Role Balance Auditor",
    "Pitch Adaptive XI"
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
                if row["Role"] in ["Batter", "WK-Batter"]:
                    return row["Batting Avg (scaled)"] * 0.6 + row["Batting SR (scaled)"] * 0.4
                elif row["Role"] == "Bowler":
                    return row["Wickets (scaled)"] * 0.6 + row["Bowling Econ (scaled)"] * 0.4
                elif row["Role"] == "All-rounder":
                    batting = row["Batting Avg (scaled)"] * 0.5 + row["Batting SR (scaled)"] * 0.2
                    bowling = row["Wickets (scaled)"] * 0.2 + row["Bowling Econ (scaled)"] * 0.1
                    return batting + bowling
                return 0

            df["Performance_score_raw"] = df.apply(compute_performance, axis=1)
            df["Performance_score"] = scaler.fit_transform(df[["Performance_score_raw"]])
            df["Fame_score"] = scaler.fit_transform(df[["Fame_index"]]) * 0.6 + scaler.fit_transform(df[["Endorsement_score"]]) * 0.4

            df["bias_score"] = df["Fame_score"] - df["Performance_score"]

             # 💡 NEW: Fair Bias Detection Logic with Margin
            fame_threshold = df["Fame_score"].quantile(0.75)
            performance_threshold = df["Performance_score"].quantile(0.25)
            margin = 0.05  # ← 5% buffer on both sides


            fame_threshold = df["Fame_score"].quantile(0.75)
            performance_threshold = df["Performance_score"].quantile(0.25)
            df["Is_Biased"] = (
           (df["Fame_score"] > fame_threshold + margin) &
           (df["Performance_score"] < performance_threshold - margin)
)

            st.session_state.df = df
            st.dataframe(df[df["Is_Biased"]][["Player Name", "Role", "Fame_score", "Performance_score", "bias_score", "Is_Biased"]])

            fig = px.scatter(df, x="Fame_score", y="Performance_score", color="Is_Biased",
                             hover_data=["Player Name", "Role"], title="Fame vs Performance Bias Map")
            st.plotly_chart(fig, use_container_width=True)

        # ------------------ Generate Final Unbiased XI ------------------
        if st.button("Generate Final Unbiased XI"):
            st.subheader("🏆 Final Unbiased XI")
            df = st.session_state.df
            unbiased_df = df[df["Is_Biased"] == False]

            wk_batter = None  # ← FIX: Always define this variable

            # Step 1: Try Unbiased WK-Batter
            wk_unbiased = unbiased_df[unbiased_df["Role"] == "Wk-Batter"].copy()
            if not wk_unbiased.empty:
                wk_unbiased["Leadership_Score"] = 0.6 * wk_unbiased["Performance_score"] + 0.4 * wk_unbiased["Fame_score"]
                wk_batter = wk_unbiased.nlargest(1, "Leadership_Score")
                st.info(f"✅ WK-Batter selected from unbiased list: {wk_batter.iloc[0]['Player Name']}")
            else:
                # Fallback: Select from original uploaded dataset (even if biased)
                wk_all = df[df["Role"] == "Wk-Batter"].copy()
                if not wk_all.empty:
                    wk_all["Leadership_Score"] = 0.6 * wk_all["Performance_score"] + 0.4 * wk_all["Fame_score"]
                    wk_batter = wk_all.nlargest(1, "Leadership_Score")
                    st.warning(f"⚠️ No unbiased WK-Batter found. Selected best available from full dataset: {wk_batter.iloc[0]['Player Name']}")
                else:
                    st.error("❌ No WK-Batter found at all in the dataset. Please include at least one.")
                      # Avoid running below code

            # Step 2: Select remaining 10 players (excluding selected WK-Batter)
            remaining_pool = unbiased_df[~unbiased_df["Player Name"].isin(wk_batter["Player Name"])]

            batters = remaining_pool[remaining_pool["Role"] == "Batter"].nlargest(4, "Performance_score")
            bowlers = remaining_pool[remaining_pool["Role"] == "Bowler"].nlargest(4, "Performance_score")
            allrounders = remaining_pool[remaining_pool["Role"] == "All-rounder"].nlargest(2, "Performance_score")

            final_xi = pd.concat([wk_batter, batters, bowlers, allrounders])
            final_xi = final_xi.drop_duplicates(subset="Player Name").head(11)
            st.session_state.final_xi = final_xi

            final_xi["Leadership_Score"] = 0.6 * final_xi["Performance_score"] + 0.4 * final_xi["Fame_score"]

            captain = final_xi.loc[final_xi["Leadership_Score"].idxmax()]
            final_xi["Captain"] = final_xi["Player Name"] == captain["Player Name"]

            remaining = final_xi[final_xi["Player Name"] != captain["Player Name"]]
            vice_captain = remaining.loc[remaining["Leadership_Score"].idxmax()]
            final_xi["Vice_Captain"] = final_xi["Player Name"] == vice_captain["Player Name"]

            st.dataframe(final_xi[["Player Name", "Role", "Performance_score", "Fame_score", "Is_Biased", "Captain", "Vice_Captain"]])

            csv = final_xi.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Final XI CSV", csv, "final_xi.csv", "text/csv")

            st.success(f"🏏 **Recommended Captain:** {captain['Player Name']} | Leadership Score: {captain['Leadership_Score']:.2f}")
            st.info(f"🥢 **Vice-Captain:** {vice_captain['Player Name']} | Leadership Score: {vice_captain['Leadership_Score']:.2f}")

            if "Rohit Sharma" in final_xi["Player Name"].values and captain["Player Name"] != "Rohit Sharma":
                rohit_score = final_xi[final_xi["Player Name"] == "Rohit Sharma"]["Leadership_Score"].values[0]
               
                st.warning(f"⚠️ Rohit Sharma is the **current captain**, but based on data, **{captain['Player Name']}** has a higher Leadership Score ({captain['Leadership_Score']:.2f}) vs Rohit's ({rohit_score:.2f}).")

        # ------------------ Manual Leadership ------------------
        if "final_xi" in st.session_state:
            st.markdown("---")
            st.subheader("✍️ Select Future Leadership Manually")

            with st.form("manual_leadership_form"):
                manual_candidates = st.multiselect(
                    "Select at least 2 players from the Unbiased XI for custom captain & vice-captain evaluation:",
                    options=st.session_state.final_xi["Player Name"].tolist()
                )
                submitted = st.form_submit_button("🧠 Calculate Leadership")

            if submitted:
                if len(manual_candidates) >= 2:
                    manual_df = st.session_state.final_xi[st.session_state.final_xi["Player Name"].isin(manual_candidates)].copy()
                    manual_df["Leadership_Score"] = 0.6 * manual_df["Performance_score"] + 0.4 * manual_df["Fame_score"]

                    manual_captain = manual_df.loc[manual_df["Leadership_Score"].idxmax()]
                    manual_df["Captain"] = manual_df["Player Name"] == manual_captain["Player Name"]

                    remaining_manual = manual_df[manual_df["Player Name"] != manual_captain["Player Name"]]
                    manual_vice_captain = remaining_manual.loc[remaining_manual["Leadership_Score"].idxmax()]
                    manual_df["Vice_Captain"] = manual_df["Player Name"] == manual_vice_captain["Player Name"]

                    st.success(f"🥢 **Manually Selected Captain:** {manual_captain['Player Name']} | Leadership Score: {manual_captain['Leadership_Score']:.2f}")
                    st.info(f"🎖 **Manually Selected Vice-Captain:** {manual_vice_captain['Player Name']} | Leadership Score: {manual_vice_captain['Leadership_Score']:.2f}")

                    st.dataframe(manual_df[[
                        "Player Name", "Role", "Performance_score", "Fame_score",
                        "Leadership_Score", "Captain", "Vice_Captain"
                    ]])
                     # ✅ Download button for manually selected leadership
                    manual_csv = manual_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Manual Captain-Vice CSV", manual_csv, "manual_captain_vice.csv", "text/csv")
                else:
                    st.warning("👥 Please select at least 2 players to perform leadership calculation.")

                # --- Signature Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)
    
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
    # --- Signature Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)
  
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
    # --- Signature Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)


# ------------------ PITCH ADAPTIVE XI ------------------
elif selected_feature == "Pitch Adaptive XI":
    st.subheader("🌍 Pitch Adaptive XI Selector")
    pitch_file = st.file_uploader("📂 Upload CSV with Player Stats & Roles", type="csv", key="pitch_adaptive_upload")

    pitch_type = st.selectbox("🧱 Select Pitch Type", ["Red Soil", "Black Soil"])
    match_time = st.selectbox("🕰️ Select Match Timing", ["Day", "Night"])

    if pitch_file:
        df = pd.read_csv(pitch_file)
        st.dataframe(df.head(), use_container_width=True)

        # Define pitch strategy logic
        if pitch_type == "Red Soil" and match_time == "Day":
            st.info("Red Soil + Day Match: Prioritize bounce-friendly pacers and spinners for later overs")
            pace_filter = df[(df["Bowling Type"].str.contains("Pace")) & (df["Bounce"] >= 7)]
            spin_filter = df[(df["Bowling Type"].str.contains("Spin")) & (df["Turn"] >= 6)]
        elif pitch_type == "Red Soil" and match_time == "Night":
            st.info("Red Soil + Night Match: Dew can reduce spin. Favor batting depth and swing bowlers")
            pace_filter = df[(df["Bowling Type"].str.contains("Pace")) & (df["Swing"] >= 6)]
            spin_filter = df[(df["Bowling Type"].str.contains("Spin")) & (df["Turn"] >= 5)]
        elif pitch_type == "Black Soil" and match_time == "Day":
            st.info("Black Soil + Day Match: Slow surface favors spin early and slower variations")
            pace_filter = df[(df["Bowling Type"].str.contains("Pace")) & (df["Variation"] >= 6)]
            spin_filter = df[(df["Bowling Type"].str.contains("Spin")) & (df["Turn"] >= 7)]
        elif pitch_type == "Black Soil" and match_time == "Night":
            st.info("Black Soil + Night Match: Dew + slow pitch favors defensive bowling & strong batting")
            pace_filter = df[(df["Bowling Type"].str.contains("Pace")) & (df["Control"] >= 6)]
            spin_filter = df[(df["Bowling Type"].str.contains("Spin")) & (df["Control"] >= 6)]

        # Top performers by role
        pacers = pace_filter.sort_values(by=["Economy", "Wickets"], ascending=[True, False]).head(3)
        spinners = spin_filter.sort_values(by=["Economy", "Wickets"], ascending=[True, False]).head(2)
        batters = df[df["Role"] == "Batter"].sort_values(by="Avg", ascending=False).head(5)
        keeper = df[df["Role"] == "Wicket Keeper"].sort_values(by="Avg", ascending=False).head(1)

        # Form final XI
        pitch_xi = pd.concat([pacers, spinners, batters, keeper]).drop_duplicates().head(11)
        st.success("✅ Pitch Adaptive XI Ready")
        st.dataframe(pitch_xi, use_container_width=True)

        # Download option
        csv_pitch = pitch_xi.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Pitch Adaptive XI", data=csv_pitch, file_name='Pitch_Adaptive_XI.csv', mime='text/csv')
