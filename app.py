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
    "Subscription Plans",
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

    # -------------- SUBSCRIPTION PLANS PAGE ----------------
if selected_feature == "Subscription Plans":
    st.title("📦 Subscription Plans – TrueXI Selector")
    st.markdown("Choose a plan below and contact us to activate it!")

    plans = {
        "🟢 XIStart Engine (₹199/month)": "✅ View-only Unbiased XI",
        "🟡 PressurePulse (₹399/month)": "✅ Heatmap XI + Download + All XIStart Engine features",
        "🟠 RoleMatrix Pro (₹699/month)": "✅ Role Balance Auditor + All PressurePulse features",
        "🔵 GameSense Elite (₹999/month)": "✅ Pitch Adaptive XI + All Pro features"
    }

    plan_choice = st.selectbox("Select a Plan", list(plans.keys()))
    st.markdown(f"**Features:** {plans[plan_choice]}")

    st.markdown("### 📞 Contact to Subscribe")

    st.info("""
    To activate your selected plan, please contact us:

    📧 Email: **nihirakhare12@gmail.com**  
    📱 WhatsApp/Call: **+91-7897138303**

    🕐 We’ll verify payment and manually activate your access.
    """)

# ------------------ MAIN APP FLOW ------------------
elif selected_feature == "Main App Flow":

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
        df.dropna(inplace=True)

        required_cols = ["Player Name", "Role", "Death Overs Perf", "Clutch Game Perf", "Pressure Fielding", "Win Clutch Impact"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # ✅ Calculate Pressure Score from weighted components
        df["Pressure_score"] = (
            df["Death Overs Perf"] * 0.4 +
            df["Clutch Game Perf"] * 0.3 +
            df["Pressure Fielding"] * 0.2 +
            df["Win Clutch Impact"] * 0.1
        )

        def assign_phase_suitability(row):
            role = row["Role"].lower()
            score = row["Pressure_score"]
            if role == "bowler":
                if score >= 0.8:
                    return "Death Overs"
                elif score >= 0.6:
                    return "Middle Overs"
                else:
                    return "Powerplay"
            else: # For batter and all-rounder
                if score >= 0.8:
                    return "Powerplay"
                elif score >= 0.6:
                    return "Middle Overs"
                else:
                    return "Death Overs"

        def assign_match_situation(row):
            if row["Pressure_score"] >= 0.75:
                return "Clutch Moments"
            elif row["Role"].lower() == "bowler":
                return "Defending" if row["Pressure_score"] >= 0.6 else "Support Role"
            elif row["Role"].lower() in ["batter", "all-rounder"]:
                return "Chasing" if row["Pressure_score"] >= 0.6 else "Anchor / Setup"
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

        # ✅ Beehive Plot
        st.markdown("### 🐝 Beehive View: Impact Rating by Role")
        import numpy as np
        role_map = {r: i for i, r in enumerate(df["Role"].unique())}
        df["Jitter_x"] = df["Role"].map(role_map) + np.random.normal(0, 0.15, size=len(df))

        fig_bee = px.scatter(
            df,
            x="Jitter_x",
            y="Impact Rating",
            color="Pressure Zone",
            hover_data=["Player Name", "Role", "Match Situation"],
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
            num_manual = st.number_input("How many manual players do you want to add?", min_value=0, max_value=20, step=1)

            for i in range(num_manual):
                st.markdown(f"#### Player {i+1}")
                name = st.text_input(f"Player Name {i+1}", key=f"name_{i}")
                role = st.selectbox(f"Role {i+1}", ["Batter", "Bowler", "All-rounder", "Wicketkeeper"], key=f"role_{i}")
                death_perf = st.number_input("Death Overs Perf", 0.0, 1.0, 0.0, 0.01, key=f"death_{i}")
                clutch_perf = st.number_input("Clutch Game Perf", 0.0, 1.0, 0.0, 0.01, key=f"clutch_{i}")
                fielding = st.number_input("Pressure Fielding", 0.0, 1.0, 0.0, 0.01, key=f"field_{i}")
                win_impact = st.number_input("Win Clutch Impact", 0.0, 1.0, 0.0, 0.01, key=f"win_{i}")

                pressure_score = round(
                    death_perf * 0.4 + clutch_perf * 0.3 + fielding * 0.2 + win_impact * 0.1, 3
                )
                impact_rating = round(pressure_score * 10, 2)

                phase = "Death Overs" if pressure_score >= 0.8 else "Middle Overs" if pressure_score >= 0.6 else "Powerplay"
                match_sit = "Clutch Moments" if pressure_score >= 0.75 else ("Defending" if role == "Bowler" and pressure_score >= 0.6 else "Chasing")

                manual_players.append({
                    "Player Name": name,
                    "Role": role,
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
                st.dataframe(pd.DataFrame(validated_players)[[ "Player Name", "Role", "Pressure_score", "Impact Rating", "Pressure Validation" ]])
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
    st.subheader("⚖️ Role Balance Auditor")

    uploaded_file = st.file_uploader("📁 Upload Final XI CSV", type="csv", key="role_balance_upload")

    if uploaded_file:
        import plotly.express as px

        df = pd.read_csv(uploaded_file)
        df.dropna(subset=["Player Name", "Role", "Format"], inplace=True)

        format_selected = df["Format"].iloc[0].strip().upper()
        st.write(f"📌 Format Detected: **{format_selected}**")

        role_limits = {
            "ODI": {
                "Opener": (1, 2), "Top Order": (1, 2), "Middle Order": (2, 3),
                "Finisher": (1, 2), "Wicketkeeper": (1, 1), "All-Rounder": (1, 3),
                "Spinner": (1, 2), "Pacer": (2, 4), "Death Specialist": (1, 2),
                "Powerplay Specialist": (1, 2), "Anchor": (1, 2), "Aggressor": (1, 2),
                "Captain": (1, 1), "Vice-Captain": (1, 1)
            },
            "T20": {
                "Opener": (1, 2), "Top Order": (1, 2), "Middle Order": (1, 2),
                "Finisher": (2, 3), "Wicketkeeper": (1, 1), "All-Rounder": (1, 3),
                "Spinner": (1, 2), "Pacer": (2, 4), "Death Specialist": (2, 3),
                "Powerplay Specialist": (1, 2), "Anchor": (0, 1), "Aggressor": (2, 3),
                "Captain": (1, 1), "Vice-Captain": (1, 1)
            }
        }

        selected_limits = role_limits.get(format_selected, {})

        def recommend_role(row):
            pos = row.get("Batting Position", None)
            sr = row.get("Strike Rate", None)
            avg = row.get("Batting Average", None)
            econ = row.get("Economy", None)
            wk = row.get("Is Wicketkeeper", "No")

            if isinstance(wk, str) and wk.lower() == "yes":
                return "Wicketkeeper"
            try:
                pos = int(pos)
                sr = float(sr) if sr is not None else 0
                avg = float(avg) if avg is not None else 0
                econ = float(econ) if econ is not None else None
            except:
                return "Unknown"

            if pos in [1, 2]:
                return "Opener" if sr >= 120 else "Anchor"
            elif pos in [3, 4]:
                return "Top Order" if avg >= 35 else "Aggressor"
            elif pos in [5, 6]:
                return "Finisher" if sr >= 130 else "Middle Order"
            elif pos >= 7:
                return "All-Rounder" if avg > 20 else "Death Specialist"
            return "Unknown"

        df["Recommended Role"] = df.apply(recommend_role, axis=1)

        # Actual Role Counts
        role_counts = df["Role"].value_counts().to_dict()

        # Role Audit Results
        audit_results = []
        for role, (min_required, max_allowed) in selected_limits.items():
            actual_count = role_counts.get(role, 0)
            status = "✅ Balanced"
            if actual_count < min_required:
                status = f"❌ Too Few (Min: {min_required})"
            elif actual_count > max_allowed:
                status = f"⚠️ Too Many (Max: {max_allowed})"
            audit_results.append({
                "Role": role,
                "Actual Count": actual_count,
                "Min Required": min_required,
                "Max Allowed": max_allowed,
                "Status": status
            })

        audit_df = pd.DataFrame(audit_results)

        # Role Match
        df["Role Match"] = df.apply(
            lambda row: "✅" if row["Role"] == row["Recommended Role"] else "❌", axis=1
        )

        st.markdown("### 📋 Role Audit Summary")
        st.dataframe(audit_df, use_container_width=True)

        st.markdown("### 🧠 Recommended Roles vs Actual")
        st.dataframe(df[["Player Name", "Role", "Recommended Role", "Role Match"]], use_container_width=True)

        st.markdown("### 📊 Role Distribution Chart")
        role_chart = pd.DataFrame(role_counts.items(), columns=["Role", "Count"])
        st.bar_chart(role_chart.set_index("Role"))

        # 🐝 Beehive Plot (Recommended Role vs Batting Avg)
        st.markdown("### 🐝 Beehive Plot: Role vs Batting Avg")
        if "Batting Average" in df.columns:
            fig = px.strip(
                df,
                x="Recommended Role",
                y="Batting Average",
                color="Recommended Role",
                stripmode="overlay",
                hover_data=["Player Name"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # 🧩 Role Category Balance
        role_to_category = {
            "Opener": "Batter", "Top Order": "Batter", "Middle Order": "Batter",
            "Finisher": "Batter", "Anchor": "Batter", "Aggressor": "Batter",
            "Pacer": "Bowler", "Spinner": "Bowler", "Death Specialist": "Bowler",
            "Powerplay Specialist": "Bowler", "All-Rounder": "All-Rounder",
            "Wicketkeeper": "Wicketkeeper"
        }

        df["Category"] = df["Recommended Role"].map(role_to_category)
        cat_counts = df["Category"].value_counts()

        st.markdown("### 🧩 Player Type Distribution (Category Balance)")
        st.dataframe(cat_counts.reset_index().rename(columns={"index": "Category", "Category": "Count"}))

        st.bar_chart(cat_counts)

    else:
        st.info("Please upload a Final XI CSV to continue.")

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
            st.markdown(f"🧭 **Toss Recommendation:** The captain should **opt to `{suggested_toss}` first** if they win the toss.")
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

            # ---------- BEEHIVE / BEESWARM STYLE STRIP PLOT ----------
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

            # ---------- REPLACEMENT LOGIC ----------
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

    # --- Signature Footer ---
    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)

