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
    "Actual Performance Indicator",
    "Role Balance Auditor",
    "Pitch Adaptive XI Selector"
])

# ------------------ HEADER ------------------
st.image("app logo.png", width=150)
st.markdown("<h1>🏏 True XI</h1>", unsafe_allow_html=True)
st.markdown("<h4>Make Data-Driven Cricket Selections Without Bias</h4>", unsafe_allow_html=True)
if lottie_cricket:
    st_lottie(lottie_cricket, height=150, key="cricket_header")

# ------------------ UTILITY FUNCTIONS ------------------
def safe_scale(column):
    if len(np.unique(column)) > 1:
        return MinMaxScaler().fit_transform(column.values.reshape(-1, 1))
    return np.full_like(column.values, 0.5).reshape(-1, 1)

# ✅ Convert "90M", "500K" etc. to numbers
def convert_social_media_value(val):
    if isinstance(val, str):
        val = val.strip().upper()
        if val.endswith("M"):
            return float(val[:-1]) * 1_000_000
        elif val.endswith("K"):
            return float(val[:-1]) * 1_000
        elif val.endswith("B"):
            return float(val[:-1]) * 1_000_000_000
        else:
            try:
                return float(val)
            except:
                return np.nan
    return val

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

                 # 🔹 NEW — If Twitter/Instagram columns exist, sum them into Social Media Reach
            if "Twitter Followers" in df.columns and "Instagram Followers" in df.columns:
                df["Twitter Followers"] = df["Twitter Followers"].apply(convert_social_media_value)
                df["Instagram Followers"] = df["Instagram Followers"].apply(convert_social_media_value)
                df["Social Media Reach"] = df["Twitter Followers"] + df["Instagram Followers"]

            # 🔹 Still handle case where Social Media Reach is already present
            if "Social Media Reach" in df.columns:
                df["Social Media Reach"] = df["Social Media Reach"].apply(convert_social_media_value)


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

            fig = px.scatter(
                df, x="Fame_score", y="Performance_score", color="Is_Biased",
                hover_data=["Player Name", "Primary Role"],
                title="Fame vs Performance Bias Map"
            )
            
            fig.update_layout(
                xaxis_title="Fame Score",
                yaxis_title="Performance Score",
                legend_title="Bias Status",
                plot_bgcolor="#0b132b",    # Matches app background
                paper_bgcolor="#0b132b",   # Matches app background
                font=dict(color="white")   # Ensures text is visible
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
                plot_bgcolor="#0b132b",     # Matches app background
                paper_bgcolor="#0b132b",    # Matches app background
                font=dict(color="white"),   # White text for visibility
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

# ------------------ ACTUAL PERFORMANCE INDICATOR ------------------
elif selected_feature == "Actual Performance Indicator":
    st.subheader("📈 Actual Performance Indicator (API) - With Format-Based Scoring")

    api_file = st.file_uploader("📂 Upload CSV with Player Stats", type="csv", key="api_upload")

    if api_file:
        df = pd.read_csv(api_file)

        required_columns = [
            "Player Name", "Format", "Runs", "Strike_Rate", "Fours", "Sixes", "Wickets",
            "Dot_Balls", "Maidens", "Runs_Conceded", "Catches", "Run_Outs", "Stumpings", "Balls_Faced"
        ]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns.")

            # -------------------- API Calculation Logic --------------------
            def calculate_scores(row):
                fmt = row["Format"]

                # Batting
                if fmt == "T20":
                    batting = row["Runs"] + (row["Strike_Rate"] / 2.5) + (row["Fours"] * 1) + (row["Sixes"] * 1.5)
                    bowling = (row["Wickets"] * 25) + (row["Dot_Balls"] * 1.2) + (row["Maidens"] * 12) - (row["Runs_Conceded"] / 2)
                elif fmt == "ODI":
                    batting = row["Runs"] + (row["Strike_Rate"] / 3) + (row["Fours"] * 1) + (row["Sixes"] * 1.2)
                    bowling = (row["Wickets"] * 25) + (row["Dot_Balls"] * 1) + (row["Maidens"] * 12) - (row["Runs_Conceded"] / 2)
                elif fmt == "Test":
                    batting = row["Runs"] + (row["Balls_Faced"] / 2)
                    bowling = (row["Wickets"] * 25) + (row["Maidens"] * 15) - (row["Runs_Conceded"] / 2)
                else:
                    batting = bowling = 0

                # Fielding (same for all formats)
                fielding = (row["Catches"] * 10) + (row["Run_Outs"] * 12) + (row["Stumpings"] * 15)

                # API formula
                api = (batting * 0.4) + (bowling * 0.4) + (fielding * 0.2)

                return pd.Series([batting, bowling, fielding, api])

            df[["Batting Score", "Bowling Score", "Fielding Score", "API"]] = df.apply(calculate_scores, axis=1)

            # -------------------- Remarks Logic --------------------
            def get_remark(api_score, max_api):
                if api_score == max_api:
                    return "🏆 Top Performer"
                elif api_score >= 0.8 * max_api:
                    return "🔥 Excellent"
                elif api_score >= 0.6 * max_api:
                    return "✅ Good"
                elif api_score >= 0.4 * max_api:
                    return "⚠️ Average"
                else:
                    return "🔻 Needs Improvement"

            df_sorted = df.sort_values(by="API", ascending=False).reset_index(drop=True)
            max_api = df_sorted["API"].max()
            df_sorted["Remarks"] = df_sorted["API"].apply(lambda x: get_remark(x, max_api))

            # 📋 API Report
            st.subheader("📋 API Performance Report")
            st.dataframe(df_sorted)

            # ⬇ CSV Download
            csv_data = df_sorted.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download API Report CSV",
                data=csv_data,
                file_name="api_performance_report.csv",
                mime="text/csv"
            )

            # 📊 Bar Chart
            import plotly.express as px
            bar_fig = px.bar(
                df_sorted, x="Player Name", y="API", color="Remarks", text_auto=True,
                title="API Score by Player"
            )
            bar_fig.update_layout(
                plot_bgcolor="#0b132b",
                paper_bgcolor="#0b132b",
                font=dict(color="white"),
                xaxis_title="Player Name",
                yaxis_title="API",
                legend_title="Remarks"
            )

            st.plotly_chart(bar_fig, use_container_width=True)

            # 🐝 Beehive Plot (Format-wise API Spread)
            st.subheader("🐝 Beehive Plot (API vs Format)")
            bee_plot = px.strip(
                df_sorted,
                x="Format",
                y="API",
                hover_data=["Player Name", "Remarks"],
                color="Remarks",
                stripmode="overlay",
                title="Beehive Plot of API Across Formats"
            )
            bee_plot.update_traces(jitter=0.35, marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            bee_plot.update_layout(
                plot_bgcolor="#0b132b",
                paper_bgcolor="#0b132b",
                font=dict(color="white"),
                xaxis_title="Format",
                yaxis_title="API",
                legend_title="Remarks"
            )
            st.plotly_chart(bee_plot, use_container_width=True)

            # 🔍 Conclusion
            st.subheader("🔍 Conclusion & Insights")
            top_player = df_sorted.iloc[0]
            least_player = df_sorted.iloc[-1]
            st.success(f"🏅 **Top Performer:** {top_player['Player Name']} with an API of `{top_player['API']:.2f}` in **{top_player['Format']}** format.")
            st.error(f"📉 **Least Performer:** {least_player['Player Name']} with an API of `{least_player['API']:.2f}`.")

            st.markdown("### 📝 Remarks Summary:")
            summary_counts = df_sorted["Remarks"].value_counts().to_dict()
            for remark, count in summary_counts.items():
                st.markdown(f"- **{remark}**: {count} player(s)")

        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing_cols))

    else:
        st.info("📁 Please upload a CSV file to proceed with API calculation.")

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

# ------------------ BOWLING CHANGE EFFICIENCY TRACKER ------------------
elif selected_feature == "Bowling Change Efficiency Tracker":
    st.subheader("🎯 Bowling Change Efficiency Tracker (With Impact Analysis & Alerts)")

    bowling_file = st.file_uploader("📂 Upload CSV with Bowling Change Data", type="csv", key="bowling_change_upload")

    if bowling_file:
        df = pd.read_csv(bowling_file)

        required_columns = ["Over Number", "Bowler Name", "New Bowler?", "Wickets in Next Over",
                            "Runs Conceded in Next Over", "Match Format"]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns.")

            # Define success criteria (basic)
            def is_successful_change(wickets, runs, fmt):
                if wickets > 0:
                    return True
                # Economy thresholds by format
                thresholds = {"T20": 6, "ODI": 4.5, "Test": 3}
                return runs <= thresholds.get(fmt, 6)

            df["Successful Change"] = df.apply(
                lambda row: is_successful_change(row["Wickets in Next Over"], row["Runs Conceded in Next Over"], row["Match Format"]),
                axis=1
            )

            # Filter only bowling changes
            changes_df = df[df["New Bowler?"] == True]

            # Calculate efficiency
            total_changes = len(changes_df)
            successful_changes = changes_df["Successful Change"].sum()
            bce_percentage = (successful_changes / total_changes * 100) if total_changes > 0 else 0

            # 📋 Display Table
            audit_df = changes_df[[
                "Over Number", "Bowler Name", "Wickets in Next Over",
                "Runs Conceded in Next Over", "Successful Change", "Match Format"
            ]]

            st.subheader("📋 Bowling Change Impact Report")
            st.dataframe(audit_df)

            # 🔍 Suggestions
            if bce_percentage < 50:
                st.warning(f"⚠ Bowling Change Efficiency is low ({bce_percentage:.1f}%). Consider reviewing change strategies.")
            else:
                st.success(f"✅ Bowling Change Efficiency is strong ({bce_percentage:.1f}%).")

            # ⬇ Download
            csv_data = audit_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Bowling Change Report",
                data=csv_data,
                file_name="bowling_change_efficiency.csv",
                mime="text/csv"
            )

            # 📊 Pie chart for success/failure
            success_counts = audit_df["Successful Change"].value_counts().reset_index()
            success_counts.columns = ["Successful Change", "Count"]
            success_counts["Successful Change"] = success_counts["Successful Change"].map({True: "Successful", False: "Unsuccessful"})

            pie_chart = px.pie(
                success_counts,
                names="Successful Change",
                values="Count",
                title="Bowling Change Success Rate"
            )
            pie_chart.update_layout(
                paper_bgcolor="#0b132b",
                plot_bgcolor="#0b132b",
                title_font_color="white",
                font=dict(color="white")
            )
            st.plotly_chart(pie_chart, use_container_width=True)

            # 📊 Bar chart: Average wickets & runs after change by format
            avg_stats = changes_df.groupby("Match Format").agg({
                "Wickets in Next Over": "mean",
                "Runs Conceded in Next Over": "mean"
            }).reset_index()

            bar_chart = px.bar(
                avg_stats,
                x="Match Format",
                y=["Wickets in Next Over", "Runs Conceded in Next Over"],
                barmode="group",
                title="Average Wickets & Runs After Bowling Change"
            )
            bar_chart.update_layout(
                paper_bgcolor="#0b132b",
                plot_bgcolor="#0b132b",
                font_color="white"
            )
            st.plotly_chart(bar_chart, use_container_width=True)

            # 🐝 Beehive plot: Over number vs Wickets
            st.subheader("🐝 Beehive Plot (Over Number vs Wickets After Change)")
            beehive = px.strip(
                changes_df,
                x="Over Number",
                y="Wickets in Next Over",
                color="Match Format",
                hover_data=["Bowler Name"],
                stripmode="overlay",
                title="Impact of Bowling Change by Over"
            )
            beehive.update_traces(
                jitter=0.35,
                marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))
            )
            beehive.update_layout(
                paper_bgcolor='#0b132b',
                plot_bgcolor='#0b132b',
                title_font=dict(color='white'),
                xaxis_title_font=dict(color='white'),
                yaxis_title_font=dict(color='white'),
                legend_title_font=dict(color='white'),
                legend_font=dict(color='white')
            )
            st.plotly_chart(beehive, use_container_width=True)

            # 🔥 NEW VISUAL 1: Impact Heatmap
            st.subheader("🔥 Impact Heatmap (Over vs Runs Conceded)")
            heatmap_data = changes_df.pivot_table(
                index="Over Number",
                columns="Match Format",
                values="Runs Conceded in Next Over",
                aggfunc="mean"
            )
            heatmap = px.imshow(
                heatmap_data,
                color_continuous_scale="RdYlGn_r",
                title="Average Runs Conceded After Bowling Change"
            )
            heatmap.update_layout(
                paper_bgcolor="#0b132b",
                plot_bgcolor="#0b132b",
                font_color="white"
            )
            st.plotly_chart(heatmap, use_container_width=True)

            # 📈 NEW VISUAL 2: Cumulative Efficiency Trend
            st.subheader("📈 Cumulative Efficiency Over Overs")
            changes_df["Cumulative Success %"] = (changes_df["Successful Change"].cumsum() / (changes_df.index + 1)) * 100
            line_chart = px.line(
                changes_df,
                x="Over Number",
                y="Cumulative Success %",
                title="Cumulative Bowling Change Efficiency Trend",
                markers=True
            )
            line_chart.update_layout(
                paper_bgcolor="#0b132b",
                plot_bgcolor="#0b132b",
                font_color="white"
            )
            st.plotly_chart(line_chart, use_container_width=True)

            # 🕸 NEW VISUAL 3: Bowler Impact Network Graph
            st.subheader("🕸 Bowler Success Network")
            import networkx as nx
            G = nx.Graph()
            for _, row in changes_df.iterrows():
                if row["Successful Change"]:
                    G.add_edge("Bowling Change", row["Bowler Name"])
            pos = nx.spring_layout(G)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            import plotly.graph_objects as go
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo='none', mode='lines')
            node_x, node_y = [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                textposition="bottom center",
                marker=dict(size=12, color="lightblue"), hoverinfo='text'
            )
            fig_net = go.Figure(data=[edge_trace, node_trace])
            fig_net.update_layout(
                paper_bgcolor="#0b132b",
                plot_bgcolor="#0b132b",
                title="Bowler Success Connection Network",
                font_color="white"
            )
            st.plotly_chart(fig_net, use_container_width=True)

        else:
            missing = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing))
    else:
        st.info("📁 Please upload a CSV file with bowling change data to continue.")

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

# ------------------------- Pitch Adaptive XI Selector ----------------------
elif selected_feature == "Pitch Adaptive XI Selector":
    st.subheader("🏟️ Pitch Adaptive XI Selector")

    uploaded_adaptive_file = st.file_uploader("📁 Upload Final XI CSV", type="csv", key="pitch_adaptive_upload")

    if uploaded_adaptive_file:
        df = pd.read_csv(uploaded_adaptive_file)

        required_columns = [
            "Player Name", "Role", "Format", "Batting Avg", "Batting SR",
            "Bowling Econ", "Bowling Avg", "Batting Rating", "Bowling Rating",
            "All-Round Rating"
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        if all(col in df.columns for col in ["Player Name", "Role", "Format"]):
            # --- Venue and Pitch Setup ---
            all_venues = {
                "Wankhede Stadium": "Red Soil", "Brabourne Stadium": "Red Soil", "MA Chidambaram Stadium": "Red Soil",
                "M. Chinnaswamy Stadium": "Red Soil", "Sawai Mansingh Stadium": "Red Soil", "Eden Gardens": "Red Soil",
                "Narendra Modi Stadium": "Black Soil", "Ekana Stadium": "Black Soil", "Arun Jaitley Stadium": "Black Soil",
                "Rajiv Gandhi Intl Stadium": "Black Soil", "Barsapara Stadium": "Black Soil", "Holkar Stadium": "Black Soil"
            }

            venue_thresholds = {
                "Wankhede Stadium": {"bat_avg": 32, "bat_sr": 140, "bat_rating": 6.5, "bowl_avg": 28, "bowl_econ": 8.0},
                "Brabourne Stadium": {"bat_avg": 30, "bat_sr": 135, "bat_rating": 6.3, "bowl_avg": 26, "bowl_econ": 7.6},
                "MA Chidambaram Stadium": {"bat_avg": 30, "bat_sr": 125, "bat_rating": 6.0, "bowl_avg": 24, "bowl_econ": 6.8},
                "M. Chinnaswamy Stadium": {"bat_avg": 35, "bat_sr": 145, "bat_rating": 7.0, "bowl_avg": 32, "bowl_econ": 9.0},
                "Sawai Mansingh Stadium": {"bat_avg": 31, "bat_sr": 132, "bat_rating": 6.2, "bowl_avg": 27, "bowl_econ": 7.4},
                "Narendra Modi Stadium": {"bat_avg": 28, "bat_sr": 130, "bat_rating": 6.3, "bowl_avg": 25, "bowl_econ": 7.5},
                "Ekana Stadium": {"bat_avg": 26, "bat_sr": 122, "bat_rating": 5.8, "bowl_avg": 22, "bowl_econ": 6.5},
                "Arun Jaitley Stadium": {"bat_avg": 29, "bat_sr": 128, "bat_rating": 6.1, "bowl_avg": 23, "bowl_econ": 6.9},
                "Rajiv Gandhi Intl Stadium": {"bat_avg": 30, "bat_sr": 130, "bat_rating": 6.2, "bowl_avg": 24, "bowl_econ": 7.0},
                "Barsapara Stadium": {"bat_avg": 31, "bat_sr": 134, "bat_rating": 6.4, "bowl_avg": 25, "bowl_econ": 7.2},
                "Holkar Stadium": {"bat_avg": 33, "bat_sr": 138, "bat_rating": 6.7, "bowl_avg": 26, "bowl_econ": 7.8},
                "Eden Gardens": {"bat_avg": 32, "bat_sr": 135, "bat_rating": 6.5, "bowl_avg": 27, "bowl_econ": 7.6},
            }

            selected_venue = st.selectbox("📍 Select Match Venue", list(all_venues.keys()))
            pitch_type = all_venues[selected_venue]
            match_time = st.selectbox("🕐 Match Time", ["Day", "Night"])

            st.markdown(f"🧱 **Pitch Type:** `{pitch_type}`")
            st.markdown(f"🕐 **Match Time:** `{match_time}`")

            def recommend_toss_decision(pitch_type, match_time):
                if pitch_type == "Red Soil" and match_time == "Day":
                    return "Bat", "Red soil crumbles and deteriorates more in day games..."
                elif pitch_type == "Red Soil" and match_time == "Night":
                    return "Field", "At night, batting becomes easier early on..."
                elif pitch_type == "Black Soil" and match_time == "Day":
                    return "Field", "Black soil holds moisture longer but slows down..."
                elif pitch_type == "Black Soil" and match_time == "Night":
                    return "Field", "Dew at night reduces spin and helps the ball skid..."

            suggested_toss, toss_reason = recommend_toss_decision(pitch_type, match_time)
            st.markdown(f"🧽 **Toss Recommendation:** The captain should **opt to `{suggested_toss}` first**.")
            st.info(f"📌 **Reason:** {toss_reason}")

            def classify_player(row, ptype=None, mtime=None):
                role = row["Role"].lower()
                ptype = ptype or pitch_type
                mtime = mtime or match_time

                if ptype == "Red Soil" and mtime == "Day":
                    if any(x in role for x in ["spinner", "spin all-rounder", "anchor"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["opener", "floater"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"
                elif ptype == "Red Soil" and mtime == "Night":
                    if any(x in role for x in ["fast", "death", "finisher", "seamer", "floater"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["opener", "spinner", "spin all-rounder"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"
                elif ptype == "Black Soil" and mtime == "Day":
                    if any(x in role for x in ["fast", "seamer", "anchor", "pace-all-rounder"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["spinner", "opener"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"
                elif ptype == "Black Soil" and mtime == "Night":
                    if any(x in role for x in ["fast", "death", "finisher", "opener"]):
                        return "✅ Strong"
                    elif any(x in role for x in ["spinner", "floater"]):
                        return "⚠️ Moderate"
                    else:
                        return "❌ Not Ideal"

            df["Pitch Adaptiveness"] = df.apply(lambda row: classify_player(row), axis=1)

            role_priority = {
                "opener": 0, "anchor": 0, "floater": 0, "finisher": 0,
                "spinner": 1, "fast": 1, "seamer": 1, "death": 1,
                "spin all-rounder": 2, "pace-all-rounder": 2, "all-rounder": 2,
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
            
            beehive = px.strip(
                df,
                x="Role",
                y="Pitch Adaptiveness",
                color="Format",
                hover_data=["Player Name"],
                stripmode="overlay",
                title="Beehive Plot of Player Roles and Adaptiveness"
            )
            beehive.update_layout(
                paper_bgcolor='#0b132b',  # Paper background
                plot_bgcolor='#0b132b',   # Plot area background
                font=dict(color="white"),  # Font color for contrast
                title_font=dict(size=18, color="white"),
                xaxis=dict(title="Role", color="white", tickfont=dict(color="white")),
                yaxis=dict(title="Adaptiveness", color="white", tickfont=dict(color="white"))
            )


            st.plotly_chart(beehive, use_container_width=True)

            # ------------------- NEW: MANUAL MULTI-ENTRY -------------------
            st.subheader("➕ Add Multiple Players Manually")
            num_manual = st.number_input("🔢 How many players do you want to add?", min_value=0, max_value=15, step=1)

            manual_players = []
            for i in range(num_manual):
                with st.expander(f"🧑 Player {i+1} Details"):
                    player = {}
                    player["Player Name"] = st.text_input(f"Name {i+1}", key=f"name_{i}")
                    player["Role"] = st.selectbox(f"Role {i+1}", list(role_priority.keys()), key=f"role_{i}")
                    player["Format"] = st.selectbox(f"Format {i+1}", ["ODI", "T20"], key=f"format_{i}")
                    player["Venue"] = st.selectbox(f"Venue for Player {i+1}", list(all_venues.keys()), key=f"venue_{i}")
                    pitch = all_venues[player["Venue"]]
                    threshold = venue_thresholds[player["Venue"]]

                    # Role specific inputs
                    if player["Role"] in ["opener", "anchor", "floater", "finisher"]:
                        player["Batting Avg"] = st.number_input(f"Batting Avg {i+1}", key=f"ba_{i}")
                        player["Batting SR"] = st.number_input(f"Strike Rate {i+1}", key=f"sr_{i}")
                        player["Batting Rating"] = st.number_input(f"Batting Rating {i+1}", key=f"br_{i}")
                    elif player["Role"] in ["spinner", "seamer", "fast", "death"]:
                        player["Bowling Avg"] = st.number_input(f"Bowling Avg {i+1}", key=f"bavg_{i}")
                        player["Bowling Econ"] = st.number_input(f"Econ Rate {i+1}", key=f"econ_{i}")
                        player["Bowling Rating"] = st.number_input(f"Bowling Rating {i+1}", key=f"bowlrat_{i}")
                    else:
                        player["Batting Avg"] = st.number_input(f"Bat Avg {i+1}", key=f"arb_{i}")
                        player["Bowling Avg"] = st.number_input(f"Bowl Avg {i+1}", key=f"arbow_{i}")
                        player["All-Round Rating"] = st.number_input(f"AR Rating {i+1}", key=f"arr_{i}")
                    
                    manual_players.append((player, pitch, threshold))

            if st.button("⚙️ Calculate All & Add Players"):
                for player_data, ptype, threshold in manual_players:
                    role = player_data["Role"]
                    name = player_data["Player Name"]
                    valid = False

                    if role in ["opener", "anchor", "floater", "finisher"]:
                        valid = (player_data["Batting Avg"] >= threshold["bat_avg"] and
                                 player_data["Batting SR"] >= threshold["bat_sr"] and
                                 player_data["Batting Rating"] >= threshold["bat_rating"])
                    elif role in ["spinner", "seamer", "fast", "death"]:
                        valid = (player_data["Bowling Avg"] <= threshold["bowl_avg"] and
                                 player_data["Bowling Econ"] <= threshold["bowl_econ"] and
                                 player_data["Bowling Rating"] >= 6.0)
                    else:
                        valid = player_data["All-Round Rating"] >= 6.0

                    player_data["Pitch Adaptiveness"] = classify_player(pd.Series({"Role": role}), ptype, match_time)

                    if valid:
                        df = pd.concat([df, pd.DataFrame([player_data])], ignore_index=True)
                        st.success(f"✅ `{name}` added successfully.")
                    else:
                        st.error(f"❌ `{name}` did not meet venue-specific metrics.")

            st.markdown("✅ Final Pitch Adaptive XI")
            st.dataframe(df)

            final_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Final Pitch Adaptive XI CSV", data=final_csv, file_name="final_pitch_adaptive_xi.csv", mime="text/csv")
        else:
            st.error("❌ Required columns missing: ['Player Name', 'Role', 'Format']")
    else:
        st.info("📂 Please upload the Final XI CSV to proceed.")

    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)
    st.markdown("""<hr style="margin-top: 50px;"/><div style='text-align: center; color: gray; font-size: 14px;'>© 2025 <b>TrueXI</b>. All rights reserved.</div>""", unsafe_allow_html=True)
