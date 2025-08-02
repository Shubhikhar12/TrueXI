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


    st.markdown(
    """
    <hr style="margin-top: 50px;"/>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2025 <b>TrueXI</b>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------ PRESSURE HEATMAP XI ------------------
elif selected_feature == "Pressure Heatmap XI":
    st.subheader("🔥 Pressure Heatmap XI")

    uploaded_file = st.file_uploader("📂 Upload Combined CSV for All Roles", type="csv", key="pressure_upload")

    expected_roles = ["opener", "anchor", "floater", "finisher", "wicketkeeper", "all-rounder", "fast bowler", "spinner"]

    role_score_logic = {
        "opener": lambda row: (row['Powerplay SR'] * 0.4) + (row['Bat Avg vs Top Teams'] * 0.3) + ((100 - row['Dot Ball % in PP']) * 0.3),
        "anchor": lambda row: (row['Powerplay SR'] * 0.4) + (row['Bat Avg vs Top Teams'] * 0.3) + ((100 - row['Dot Ball % in PP']) * 0.3),
        "floater": lambda row: (row['Middle Over SR'] * 0.4) + (row['Boundary %'] * 0.3) + (row['Avg after Early Wkts'] * 0.3),
        "finisher": lambda row: (row['Death Over SR'] * 0.4) + (row['Death Bat Avg'] * 0.3) + ((100 - row['Dot Ball % Death']) * 0.3),
        "wicketkeeper": lambda row: (row['Powerplay SR'] * 0.3) + ((100 - row['Dot Ball % in PP']) * 0.2) + (row['Bat Avg vs Top Teams'] * 0.2) + (row['Dismissals/Innings'] * 0.2) - (row['Error Rate'] * 0.1),
        "all-rounder": lambda row: ((row['Bat Pressure Score'] * 0.5) + (row['Bowl Pressure Score'] * 0.5)),
        "fast bowler": lambda row: ((100 - row['Death Econ']) * 0.3) + (row['Dot% Death'] * 0.3) + (row['Wkts in Death'] * 0.4),
        "spinner": lambda row: ((100 - row['Mid Overs Econ']) * 0.4) + (row['Wkts Middle'] * 0.4) + (row['Dot% Middle'] * 0.2)
    }

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Convert relevant columns to numeric
        columns_to_convert = [
            'Powerplay SR', 'Bat Avg vs Top Teams', 'Dot Ball % in PP', 'Middle Over SR',
            'Boundary %', 'Avg after Early Wkts', 'Death Over SR', 'Death Bat Avg', 'Dot Ball % Death',
            'Dismissals/Innings', 'Error Rate', 'Bat Pressure Score', 'Bowl Pressure Score',
            'Death Econ', 'Dot% Death', 'Wkts in Death', 'Mid Overs Econ', 'Wkts Middle', 'Dot% Middle'
        ]
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'Role' not in df.columns:
            st.error("❌ 'Role' column is missing in uploaded CSV.")
        else:
            df['Pressure Score'] = 0.0  # Initialize Pressure Score

            # Calculate Pressure Scores
            for idx, row in df.iterrows():
                role = str(row['Role']).lower()
                if role in role_score_logic:
                    try:
                        df.at[idx, 'Pressure Score'] = role_score_logic[role](row)
                    except Exception as e:
                        st.warning(f"⚠️ Error calculating pressure score for {row['Player Name']} ({role}): {e}")
                else:
                    st.warning(f"⚠️ Unknown role '{role}' for player {row['Player Name']}")

            df['Performs Under Pressure'] = df['Pressure Score'].apply(lambda x: '✅' if x >= 75 else '❌')

            st.markdown("### 📋 All Players Pressure Scores")
            st.dataframe(df)

            # ---------------- Manual Player Section ----------------
            st.markdown("### ➕ Add Manual Players (Replaces ❌ Players)")
            manual_players = []
            with st.expander("Add Manual Players"):
                num_manual = st.number_input("Number of Manual Players to Add", min_value=0, max_value=10, step=1, key="manual_add_count")
                
                for i in range(num_manual):
                    st.markdown(f"---\n#### Manual Player {i+1}")
                    name = st.text_input("Player Name", key=f"manual_name_{i}")
                    role = st.selectbox("Role", expected_roles, key=f"manual_role_{i}")

                    inputs = {}
                    if role in ["opener", "anchor"]:
                        inputs['Powerplay SR'] = st.number_input("Powerplay SR", key=f"ppsr_{i}")
                        inputs['Bat Avg vs Top Teams'] = st.number_input("Bat Avg vs Top Teams", key=f"batavg_{i}")
                        inputs['Dot Ball % in PP'] = st.number_input("Dot Ball % in PP", key=f"dotpp_{i}")
                    elif role == "floater":
                        inputs['Middle Over SR'] = st.number_input("Middle Over SR", key=f"mosr_{i}")
                        inputs['Boundary %'] = st.number_input("Boundary %", key=f"bnd_{i}")
                        inputs['Avg after Early Wkts'] = st.number_input("Avg after Early Wkts", key=f"earlywkts_{i}")
                    elif role == "finisher":
                        inputs['Death Over SR'] = st.number_input("Death Over SR", key=f"dosr_{i}")
                        inputs['Death Bat Avg'] = st.number_input("Death Bat Avg", key=f"davg_{i}")
                        inputs['Dot Ball % Death'] = st.number_input("Dot Ball % Death", key=f"dotdeath_{i}")
                    elif role == "wicketkeeper":
                        inputs['Powerplay SR'] = st.number_input("Powerplay SR", key=f"wksr_{i}")
                        inputs['Dot Ball % in PP'] = st.number_input("Dot Ball % in PP", key=f"wkdot_{i}")
                        inputs['Bat Avg vs Top Teams'] = st.number_input("Bat Avg vs Top Teams", key=f"wkavg_{i}")
                        inputs['Dismissals/Innings'] = st.number_input("Dismissals/Innings", key=f"wkdis_{i}")
                        inputs['Error Rate'] = st.number_input("Error Rate", key=f"wkerr_{i}")
                    elif role == "all-rounder":
                        inputs['Bat Pressure Score'] = st.number_input("Bat Pressure Score", key=f"bps_{i}")
                        inputs['Bowl Pressure Score'] = st.number_input("Bowl Pressure Score", key=f"bws_{i}")
                    elif role == "fast bowler":
                        inputs['Death Econ'] = st.number_input("Death Econ", key=f"econ_{i}")
                        inputs['Dot% Death'] = st.number_input("Dot% Death", key=f"dotdeathp_{i}")
                        inputs['Wkts in Death'] = st.number_input("Wkts in Death", key=f"wktsdeath_{i}")
                    elif role == "spinner":
                        inputs['Mid Overs Econ'] = st.number_input("Mid Overs Econ", key=f"spinecon_{i}")
                        inputs['Wkts Middle'] = st.number_input("Wkts Middle", key=f"spinwkts_{i}")
                        inputs['Dot% Middle'] = st.number_input("Dot% Middle", key=f"spindot_{i}")

                    if st.button(f"Calculate Score for {name}", key=f"calc_button_{i}"):
                        try:
                            temp_row = pd.Series(inputs)
                            pressure_score = role_score_logic[role](temp_row)
                            performs = '✅' if pressure_score >= 75 else '❌'
                            st.success(f"Pressure Score: {pressure_score:.2f} | Performs Under Pressure: {performs}")

                            manual_players.append({
                                "Player Name": name,
                                "Role": role,
                                "Pressure Score": round(pressure_score, 2),
                                "Performs Under Pressure": performs
                            })
                        except Exception as e:
                            st.error(f"Calculation Error: {e}")

            # Replace ❌ Players with Manual Players
            underperforming_indices = df[df['Performs Under Pressure'] == '❌'].index.tolist()
            for idx, manual_player in zip(underperforming_indices, manual_players):
                df.at[idx, 'Player Name'] = manual_player['Player Name']
                df.at[idx, 'Role'] = manual_player['Role']
                df.at[idx, 'Pressure Score'] = manual_player['Pressure Score']
                df.at[idx, 'Performs Under Pressure'] = manual_player['Performs Under Pressure']

            # Sort & Select Top 11
            final_xi = df.sort_values(by="Pressure Score", ascending=False).head(11)
            st.markdown("### ✅ Final Pressure XI Candidates")
            st.dataframe(final_xi)

            # BeeHive (Swarm) Plot
            st.markdown("### 🐝 Beehive Plot - Pressure Score Distribution")
            fig = px.strip(df, x='Role', y='Pressure Score', color='Performs Under Pressure', stripmode='overlay', title="Pressure Score Beehive Plot", template="plotly_dark")
            st.plotly_chart(fig)

            st.download_button("⬇ Download Final XI CSV", final_xi.to_csv(index=False), file_name="pressure_xi.csv")

    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)

    st.markdown(
    """
    <hr style="margin-top: 50px;"/>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2025 <b>TrueXI</b>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)


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

    st.markdown(
    """
    <hr style="margin-top: 50px;"/>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2025 <b>TrueXI</b>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

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
                    return "Bat", "Red soil crumbles and deteriorates more in day games. The dry surface assists spinners early, and chasing becomes harder as the pitch breaks down."
                elif pitch_type == "Red Soil" and match_time == "Night":
                    return "Field", "At night, batting becomes easier early on before turn and variable bounce develop. Dew also reduces spin effectiveness, making chasing more favorable."
                elif pitch_type == "Black Soil" and match_time == "Day":
                    return "Bat", "Black soil holds moisture longer but slows down as the match progresses. Spin starts to grip more, and stroke-making becomes harder in the second innings."
                elif pitch_type == "Black Soil" and match_time == "Night":
                    return "Field", "Dew at night reduces spin and helps the ball skid under lights. The pitch stays better for batting in the second innings, making chasing the safer option."

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

    st.markdown(
    """
    <hr style="margin-top: 50px;"/>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2025 <b>TrueXI</b>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
