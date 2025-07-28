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
    "Pitch Adaptive XI Auditor"
])

# ------------------ HEADER ------------------
st.image("app logo.jpg", width=150)
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
    role = row["Role"].strip().lower()
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
            df["Role"] = df["Role"].astype(str).str.strip().str.lower()
            required_columns = [
                "Player Name", "Role", "Format", "Batting Avg", "Batting SR",
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
            role_filter = st.multiselect("Filter by Role", options=df["Role"].unique(), default=df["Role"].unique())
            filtered_df = df[(df["Format"] == format_filter) & (df["Role"].isin(role_filter))]
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
                df["Google Trends (scaled)"] * 0.35 +
                df["Social Media Reach (scaled)"] * 0.35 +
                df["Performance_score"] * 0.30
            )
            df["bias_score"] = df["Fame_score"] - df["Performance_score"]

            fame_q3 = df["Fame_score"].quantile(0.75)
            perf_q1 = df["Performance_score"].quantile(0.25)
            margin = 0.05

            df["Is_Biased"] = (df["Fame_score"] > fame_q3 + margin) & (df["Performance_score"] < perf_q1 - margin)
            st.session_state.df = df

            st.dataframe(df[df["Is_Biased"]][[
                "Player Name", "Role", "Fame_score", "Performance_score",
                "bias_score", "Is_Biased"
            ]])

            fig = px.scatter(df, x="Fame_score", y="Performance_score", color="Is_Biased",
                             hover_data=["Player Name", "Role"],
                             title="Fame vs Performance Bias Map")
            fig.update_layout(
                xaxis_title="Fame Score",
                yaxis_title="Performance Score",
                legend_title="Bias Status"
            )
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Generate Final Unbiased XI"):
            st.subheader("\U0001F3C6 Final Unbiased XI")
            df = st.session_state.df
            unbiased_df = df[df["Is_Biased"] == False].copy()

            wk_batter = None
            wk_unbiased = unbiased_df[unbiased_df["Role"] == "wk-batter"].copy()
            if not wk_unbiased.empty:
                wk_unbiased = calculate_leadership_score(wk_unbiased)
                wk_batter = wk_unbiased.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).head(1)
                st.info(f"✅ WK-Batter selected from unbiased list: {wk_batter.iloc[0]['Player Name']}")
            else:
                wk_all = df[df["Role"] == "wk-batter"].copy()
                if not wk_all.empty:
                    wk_all = calculate_leadership_score(wk_all)
                    wk_batter = wk_all.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).head(1)
                    st.warning(f"⚠ No unbiased WK-Batter found. Selected best available: {wk_batter.iloc[0]['Player Name']}")
                else:
                    st.error("❌ No WK-Batter found in dataset.")

            remaining_pool = unbiased_df[~unbiased_df["Player Name"].isin(wk_batter["Player Name"])]
            batters = remaining_pool[remaining_pool["Role"] == "batter"].nlargest(4, "Performance_score")
            bowlers = remaining_pool[remaining_pool["Role"] == "bowler"].nlargest(4, "Performance_score")
            allrounders = remaining_pool[remaining_pool["Role"] == "all-rounder"].nlargest(2, "Performance_score")

            final_xi = pd.concat([wk_batter, batters, bowlers, allrounders]).drop_duplicates("Player Name").head(11)
            final_xi = calculate_leadership_score(final_xi)

            final_xi = final_xi.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False)
            final_xi["Captain"] = False
            final_xi["Vice_Captain"] = False
            final_xi.iloc[0, final_xi.columns.get_loc("Captain")] = True
            final_xi.iloc[1, final_xi.columns.get_loc("Vice_Captain")] = True

            st.session_state.final_xi = final_xi
            st.dataframe(final_xi[[
                "Player Name", "Role", "Performance_score", "Fame_score", "Is_Biased", "Captain", "Vice_Captain"
            ]])

            csv = final_xi[[
                "Player Name", "Role", "Performance_score", "Fame_score", "Captain", "Vice_Captain"
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
                        "Player Name", "Role", "Performance_score", "Fame_score",
                        "Leadership_Score", "Captain", "Vice_Captain"
                    ]])

                    manual_csv = manual_df[[
                        "Player Name", "Role", "Performance_score", "Fame_score", "Captain", "Vice_Captain"
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
        df.dropna(inplace=True)

        # ---------------------- Check for Performance_score column ----------------------
        if "Performance_score" not in df.columns:
            st.error("❌ 'Performance_score' column is missing in your uploaded CSV.")
            st.info("🔁 Please go to 'Main App Flow' first, upload your Final XI, and let the app calculate 'Performance_score'. Then save and re-upload here.")
            st.stop()


        required_cols_base = ["Player Name", "Role", "Performance_score"]
        missing_cols = [col for col in required_cols_base if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            if "Pressure_score" not in df.columns:
                if st.checkbox("⚙️ Auto-generate Pressure Score (Performance × 0.97)"):
                    df["Pressure_score"] = df["Performance_score"] * 0.97
                    st.success("✅ 'Pressure_score' generated from Performance_score.")
                else:
                    st.warning("⚠️ 'Pressure_score' column is missing. Please check the box to auto-generate or upload a complete CSV.")
                    st.stop()

            def assign_phase_suitability(row):
                if row["Performance_score"] >= 0.8:
                    return "Death Overs"
                elif row["Performance_score"] >= 0.6:
                    return "Middle Overs"
                else:
                    return "Powerplay"

            def assign_match_situation(row):
                if row["Role"] == "Bowler" and row["Performance_score"] >= 0.7:
                    return "Defending"
                elif row["Performance_score"] >= 0.75:
                    return "Clutch Moments"
                else:
                    return "Chasing"

            df["Phase Suitability"] = df.apply(assign_phase_suitability, axis=1)
            df["Match Situation"] = df.apply(assign_match_situation, axis=1)
            df["Pressure Zone"] = pd.cut(df["Pressure_score"], bins=[0, 0.55, 0.65, 1], labels=["Low", "Medium", "High"])
            df["Impact Rating"] = round((df["Performance_score"] * 0.6 + df["Pressure_score"] * 0.4) * 10, 2)

            st.success("✅ File processed with Pressure Heatmap logic.")
            st.markdown("Visualize how your players perform under pressure situations and phases.")

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

            st.markdown("### 💪 Top XI Under High Pressure")
            top_xi = df[df["Pressure Zone"] == "High"].sort_values(by="Impact Rating", ascending=False).head(11)
            st.dataframe(top_xi[["Player Name", "Role", "Phase Suitability", "Match Situation", "Impact Rating"]])

            st.markdown("### ❌ Players Not Selected in Top XI (High Pressure)")
            picked_names = top_xi["Player Name"].tolist()
            unpicked_df = df[(df["Pressure Zone"] == "High") & (~df["Player Name"].isin(picked_names))]

            if not unpicked_df.empty:
                st.dataframe(unpicked_df[["Player Name", "Role", "Impact Rating"]])
            else:
                st.info("✅ All High Pressure players are in the Top XI.")

            st.markdown("### ✍️ Add Manual Player(s) to XI")
            manual_players = []
            manual_validated = False

            with st.expander("➕ Add Player Manually"):
                num_manual = st.number_input("How many manual players do you want to add?", min_value=0, max_value=5, step=1)

                for i in range(num_manual):
                    st.markdown(f"#### Player {i+1}")
                    name = st.text_input(f"Player Name {i+1}", key=f"name_{i}")
                    role = st.selectbox(f"Role {i+1}", ["Batter", "Bowler", "All-rounder", "Wicketkeeper"], key=f"role_{i}")
                    perf_score = st.number_input(f"Performance Score {i+1}", min_value=0.0, max_value=1.0, step=0.01, key=f"perf_{i}")

                    pressure_score = round(perf_score * 0.97, 3)
                    impact_rating = round((perf_score * 0.6 + pressure_score * 0.4) * 10, 2)

                    phase = "Death Overs" if perf_score >= 0.8 else "Middle Overs" if perf_score >= 0.6 else "Powerplay"
                    match_sit = "Defending" if role == "Bowler" and perf_score >= 0.7 else "Clutch Moments" if perf_score >= 0.75 else "Chasing"

                    manual_players.append({
                        "Player Name": name,
                        "Role": role,
                        "Performance_score": perf_score,
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
                            p["Performance_score"] >= 0.7 and
                            p["Impact Rating"] >= 6.5 and
                            p["Pressure Zone"] == "High"
                        )
                        status = "✅ Performs Under Pressure" if is_valid else "❌ Does Not Perform Under Pressure"
                        validated_players.append({
                            **p,
                            "Pressure Validation": status
                        })

                    st.markdown("### 🔍 Manual Players Pressure Validation")
                    st.dataframe(pd.DataFrame(validated_players)[[
                        "Player Name", "Role", "Performance_score", "Impact Rating", "Pressure Validation"
                    ]])
                    manual_players = validated_players
                    manual_validated = True

            combined_df = pd.concat([top_xi, pd.DataFrame(manual_players)], ignore_index=True)

            st.markdown("### 📋 Final Pressure XI")
            st.dataframe(combined_df[["Player Name", "Role", "Phase Suitability", "Match Situation", "Impact Rating"]])

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

# ------------------ PITCH ADAPTIVE XI AUDITOR ------------------
elif selected_feature == "Pitch Adaptive XI Auditor":

    import pandas as pd
    import streamlit as st

    st.subheader("🌱 Pitch Adaptive XI Auditor (Upload-Based)")

    uploaded_file = st.file_uploader("📁 Upload Unbiased XI CSV", type="csv")

    # Venue modifiers
    venue_modifiers = {
        "Red Soil": {
            "Day": {"opener": 1.0, "anchor": 1.0, "spinner": 1.1, "pacer": 0.9},
            "Night": {"opener": 0.9, "anchor": 0.9, "spinner": 1.2, "pacer": 1.0}
        },
        "Black Soil": {
            "Day": {"opener": 1.0, "anchor": 1.0, "spinner": 0.9, "pacer": 1.2},
            "Night": {"opener": 0.8, "anchor": 0.8, "spinner": 1.0, "pacer": 1.3}
        }
    }

    # Base scores per role
    base_scores = {
        "opener": 7,
        "anchor": 7,
        "spinner": 7,
        "pacer": 7,
        "finisher": 6
    }

    # Adaptability calculator
    def calculate_adaptability(row):
        score = 0
        if row.get('avg_vs_spin', 0) > 35:
            score += 2
        if row.get('avg_vs_pace', 0) > 35:
            score += 2
        if row.get('roles', 0) >= 3:
            score += 2
        if row.get('venues_played', 0) >= 5:
            score += 2
        if row.get('clutch_score', 0) > 7:
            score += 2
        return min(score, 10)

    # Pitch score and suitability calculation (Skill removed)
    def calculate_pitch_score_and_suitability(row, soil, timing):
        role = row["Primary Role"].strip().lower()
        adaptability = row.get("Adaptability Score", 5)

        base_score = base_scores.get(role, 6)
        role_modifier = venue_modifiers[soil][timing].get(role, 1.0)

        final_score = (base_score * role_modifier) + (0.2 * adaptability)
        threshold = 7.5 if soil == "Black Soil" and timing == "Night" else 7.0
        suitability = "✅ Best Suited" if final_score >= threshold else "❌ Not Best Suited"

        return pd.Series([round(final_score, 2), suitability])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Adaptability Score" not in df.columns:
            df["Adaptability Score"] = df.apply(lambda row: calculate_adaptability(row), axis=1)

        soil = st.selectbox("🫑 Choose Soil Type", ["Red Soil", "Black Soil"])
        timing = st.selectbox("⏰ Choose Match Timing", ["Day", "Night"])

        df[["Pitch Score", "Pitch Suitability"]] = df.apply(
            lambda row: calculate_pitch_score_and_suitability(row, soil, timing), axis=1
        )

        st.subheader("📊 Unbiased XI with Pitch Score & Suitability")
        st.dataframe(df)

        csv_original = df.to_csv(index=False).encode("utf-8")
        st.download_button("📅 Download Original CSV", data=csv_original, file_name="Pitch_Adaptive_XI_Original.csv", mime="text/csv")

        # --- Manual Replacement Logic ---
        unsuitable_players = df[df["Pitch Suitability"] == "❌ Not Best Suited"]
        replacements_made = False

        if not unsuitable_players.empty:
            st.warning("⚠ Some players are not best suited. Replace below if needed:")

            for idx, row in unsuitable_players.iterrows():
                with st.expander(f"🔁 Replace {row['Player Name']} ({row['Primary Role']})"):
                    new_name = st.text_input("Name", key=f"name_{idx}")
                    new_role = st.selectbox("Primary Role", options=list(base_scores.keys()), key=f"role_{idx}")
                    new_adapt = st.slider("Adaptability Score", 1, 10, 5, key=f"adapt_{idx}")
                    if st.button("➕ Add Replacement", key=f"add_{idx}") and new_name:
                        new_row = pd.DataFrame([{
                            "Player Name": new_name,
                            "Primary Role": new_role.title(),
                            "Adaptability Score": new_adapt
                        }])
                        new_row[["Pitch Score", "Pitch Suitability"]] = new_row.apply(
                            lambda r: calculate_pitch_score_and_suitability(r, soil, timing), axis=1
                        )
                        df.loc[idx] = new_row.iloc[0]
                        st.success(f"{new_name} added as replacement for {row['Player Name']}")
                        replacements_made = True

        # Final Updated XI
        st.subheader("✅ Final Pitch Adaptive XI")
        st.dataframe(df)

        csv_final = df.to_csv(index=False).encode("utf-8")
        st.download_button("📅 Download Final Updated CSV", data=csv_final, file_name="Pitch_Adaptive_XI_Updated.csv", mime="text/csv")

        # Chart
        st.subheader("📈 Pitch Score Bar Chart")
        st.bar_chart(df.set_index("Player Name")["Pitch Score"])

    else:
        st.info("Please upload your Unbiased XI CSV file to begin.")

    # --- Signature Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)
