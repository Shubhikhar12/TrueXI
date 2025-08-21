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
st.sidebar.title("📊 Unbiased Tools")
selected_feature = st.sidebar.radio("Select Feature", [
    "Main App Flow",
    "Opponent-Specific Impact Scores",
    "Dismissal Mode Vulnerability"            
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
                "Wickets", "Bowling Economy", "Google Trends Score", "Instagram Followers", "Twitter Followers"
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

                
            # ------------------ SUBSTITUTES ------------------
            st.subheader("🛠️ Substitution Players (Bench)")
            remaining_candidates = unbiased_df[~unbiased_df["Player Name"].isin(final_xi["Player Name"])]
            substitutes = calculate_leadership_score(remaining_candidates).sort_values(
                by=["Leadership_Score", "Fame_score"], ascending=False
            ).head(4)

            if substitutes.empty:
                st.warning("⚠ No eligible substitutes available.")
            else:
                st.dataframe(substitutes[[
                    "Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score"
                ]])

                sub_csv = substitutes[[
                    "Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score"
                ]].to_csv(index=False).encode("utf-8")

                st.download_button("⬇ Download Substitutes CSV", sub_csv, "substitutes.csv", "text/csv")


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

# ------------------ OPPONENT-SPECIFIC IMPACT SCORE ------------------
elif selected_feature == "Opponent-Specific Impact Scores":
    st.subheader("🎯 Opponent-Specific Impact Scores (OSIS) - Matchup Analysis")

    osis_file = st.file_uploader("📂 Upload CSV with Player Match Stats", type="csv", key="osis_upload")

    # ---------- Helpers ----------
    def _dark_layout(fig, title=None, xlab=None, ylab=None):
        fig.update_layout(
            plot_bgcolor="#0b132b",
            paper_bgcolor="#0b132b",
            font=dict(color="white"),
            xaxis_title=xlab,
            yaxis_title=ylab,
            title=title,
            legend_title="Legend"
        )
        return fig

    if osis_file:
        df = pd.read_csv(osis_file)

        required_columns = [
            "Player", "Primary Role", "Opponent", "Format",
            "Runs", "Balls_Faced", "Wickets", "Overs_Bowled", "Runs_Conceded",
            "Catches", "Run_Outs", "Stumpings"
        ]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns (including Format).")

            # -------------------- Impact Calculation Functions --------------------
            def batting_impact(runs, balls):
                if pd.isna(balls) or balls == 0:
                    return 0.0
                strike_rate = (runs / balls) * 100.0
                return (runs * 0.6) + (strike_rate * 0.4)

            def bowling_impact(wickets, overs, runs_conceded):
                if pd.isna(overs) or overs == 0:
                    return 0.0
                economy = runs_conceded / overs
                return (wickets * 20 * 0.7) + ((6 - economy) * 10 * 0.3)

            def fielding_impact(catches, run_outs, stumpings):
                return (catches * 10) + (run_outs * 12) + (stumpings * 15)

            def calculate_role_impact(row):
                role = str(row["Primary Role"]).strip().lower()
                runs = row["Runs"]; balls = row["Balls_Faced"]
                wkts = row["Wickets"]; overs = row["Overs_Bowled"]; rc = row["Runs_Conceded"]
                catches = row["Catches"]; ro = row["Run_Outs"]; stp = row["Stumpings"]

                if role == "batter":
                    return batting_impact(runs, balls)
                elif role == "bowler":
                    return bowling_impact(wkts, overs, rc)
                elif role == "all-rounder":
                    bat = batting_impact(runs, balls)
                    bowl = bowling_impact(wkts, overs, rc)
                    return (bat + bowl) / 2.0
                elif role in ["wk-batter", "wk batter", "wicketkeeper", "wicket-keeper", "wk"]:
                    bat = batting_impact(runs, balls)
                    wk_field = fielding_impact(catches, ro, stp)
                    return bat + (wk_field * 0.5)
                else:
                    return batting_impact(runs, balls)

            # Safety fill for numeric columns
            num_cols = ["Runs", "Balls_Faced", "Wickets", "Overs_Bowled", "Runs_Conceded", "Catches", "Run_Outs", "Stumpings"]
            df[num_cols] = df[num_cols].fillna(0)

            # Per-match impact
            df["Impact"] = df.apply(calculate_role_impact, axis=1)

            # -------------------- OSIS Calculation --------------------
            overall = (
                df.groupby(["Player", "Primary Role", "Format"], as_index=False)["Impact"]
                .mean()
                .rename(columns={"Impact": "Overall_Avg_Impact"})
            )

            vs_opp = (
                df.groupby(["Player", "Primary Role", "Opponent", "Format"], as_index=False)["Impact"]
                .mean()
                .rename(columns={"Impact": "Opponent_Avg_Impact"})
            )

            osis_all = vs_opp.merge(overall, on=["Player", "Primary Role", "Format"], how="left")
            osis_all["OSIS"] = osis_all.apply(
                lambda r: (r["Opponent_Avg_Impact"] / r["Overall_Avg_Impact"] * 100.0) if r["Overall_Avg_Impact"] not in [0, None] else 0.0,
                axis=1
            )

            # -------------------- Remarks per Opponent --------------------
            def add_remarks_per_opponent(osis_df):
                out = []
                for (opp, fmt), sub in osis_df.groupby(["Opponent", "Format"]):
                    sub = sub.copy()
                    max_osis = sub["OSIS"].max() if len(sub) else 0
                    def remark(x):
                        if x == max_osis:
                            return "🏆 Top Matchup"
                        elif x >= 0.8 * max_osis:
                            return "🔥 Strong"
                        elif x >= 0.6 * max_osis:
                            return "✅ Solid"
                        elif x >= 0.4 * max_osis:
                            return "⚠️ Average"
                        else:
                            return "🔻 Weak"
                    sub["Remarks"] = sub["OSIS"].apply(remark)
                    out.append(sub)
                return pd.concat(out, ignore_index=True) if out else osis_df

            osis_all = add_remarks_per_opponent(osis_all)

            # -------------------- Format Filter --------------------
            format_list = sorted(osis_all["Format"].unique().tolist())
            chosen_format = st.selectbox("📌 Select Format", format_list)

            osis_all_fmt = osis_all[osis_all["Format"] == chosen_format]

            # -------------------- View Switcher --------------------
            opponent_list = sorted(osis_all_fmt["Opponent"].unique().tolist())
            view_choice = st.selectbox(
                "🔎 View Mode",
                ["All Opponents (Summary)"] + opponent_list,
                help="Choose 'All Opponents' for an overview or pick an opponent for a deep-dive."
            )

            # -------------------- ALL OPPONENTS (SUMMARY) --------------------
            if view_choice == "All Opponents (Summary)":
                st.subheader(f"🌐 Summary Across All Opponents — {chosen_format}")

                # Heatmap
                pivot_osis = osis_all_fmt.pivot_table(index="Player", columns="Opponent", values="OSIS", aggfunc="mean").fillna(0).round(2)
                st.markdown("**OSIS Heatmap (Players × Opponents)**")
                import plotly.express as px
                fig_heat = px.imshow(
                    pivot_osis,
                    labels=dict(x="Opponent", y="Player", color="OSIS"),
                    aspect="auto",
                    title=f"OSIS Heatmap Across Opponents ({chosen_format})"
                )
                _dark_layout(fig_heat)
                st.plotly_chart(fig_heat, use_container_width=True)

                # Average OSIS per Opponent
                avg_vs_opp = osis_all_fmt.groupby("Opponent", as_index=False)["OSIS"].mean().sort_values("OSIS", ascending=False)
                fig_avg = px.bar(
                    avg_vs_opp, x="Opponent", y="OSIS", text_auto=True,
                    title=f"Average OSIS by Opponent (Team-wide) — {chosen_format}"
                )
                _dark_layout(fig_avg, xlab="Opponent", ylab="Average OSIS")
                st.plotly_chart(fig_avg, use_container_width=True)

                # Best/Worst Opponent per Player
                best = osis_all_fmt.loc[osis_all_fmt.groupby("Player")["OSIS"].idxmax()].rename(columns={"Opponent": "Best_Opponent", "OSIS": "Best_OSIS"})
                worst = osis_all_fmt.loc[osis_all_fmt.groupby("Player")["OSIS"].idxmin()].rename(columns={"Opponent": "Worst_Opponent", "OSIS": "Worst_OSIS"})
                bw = best[["Player", "Primary Role", "Best_Opponent", "Best_OSIS"]].merge(
                    worst[["Player", "Worst_Opponent", "Worst_OSIS"]],
                    on="Player", how="left"
                ).sort_values(["Primary Role", "Player"])
                st.markdown("**Best vs Worst Opponent per Player**")
                st.dataframe(bw.reset_index(drop=True))

                # Top 5 per Opponent
                st.markdown("**Top 6 Matchups per Opponent**")
                for opp in opponent_list:
                    sub = osis_all_fmt[osis_all_fmt["Opponent"] == opp].sort_values("OSIS", ascending=False).head(5)
                    st.markdown(f"**🏟️ {opp} — Top 6**")
                    st.dataframe(sub[["Player", "Primary Role", "Opponent_Avg_Impact", "Overall_Avg_Impact", "OSIS", "Remarks"]].reset_index(drop=True))

                # Downloads
                st.download_button(
                    f"⬇ Download Full OSIS ({chosen_format})",
                    data=osis_all_fmt.to_csv(index=False).encode("utf-8"),
                    file_name=f"osis_all_opponents_{chosen_format}.csv",
                    mime="text/csv"
                )

                # Insights
                st.subheader("🔍 Summary Insights")
                top_opp = avg_vs_opp.iloc[0]
                low_opp = avg_vs_opp.iloc[-1]
                st.info(
                    f"✅ **Best team-wide matchup in {chosen_format}:** **{top_opp['Opponent']}** (avg OSIS `{top_opp['OSIS']:.2f}`)\n\n"
                    f"⚠️ **Most challenging opponent in {chosen_format}:** **{low_opp['Opponent']}** (avg OSIS `{low_opp['OSIS']:.2f}`)."
                )

            # -------------------- SINGLE OPPONENT (DEEP-DIVE) --------------------
            else:
                selected_opponent = view_choice
                st.subheader(f"🧭 Deep-Dive: OSIS vs **{selected_opponent}** — {chosen_format}")

                osis_df = (
                    osis_all_fmt[osis_all_fmt["Opponent"] == selected_opponent]
                    .copy()
                    .sort_values("OSIS", ascending=False)
                    .reset_index(drop=True)
                )

                # Recompute remarks
                max_osis = osis_df["OSIS"].max() if len(osis_df) else 0
                def get_remark(osis_score, max_):
                    if osis_score == max_:
                        return "🏆 Top Matchup"
                    elif osis_score >= 0.8 * max_:
                        return "🔥 Strong"
                    elif osis_score >= 0.6 * max_:
                        return "✅ Solid"
                    elif osis_score >= 0.4 * max_:
                        return "⚠️ Average"
                    else:
                        return "🔻 Weak"
                osis_df["Remarks"] = osis_df["OSIS"].apply(lambda x: get_remark(x, max_osis))

                # 📋 OSIS Report
                st.markdown(f"### 📋 OSIS Report vs {selected_opponent} — {chosen_format}")
                st.dataframe(osis_df[["Player", "Primary Role", "Overall_Avg_Impact", "Opponent_Avg_Impact", "OSIS", "Remarks"]])

                # ⬇ CSV Download
                csv_data = osis_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇ Download OSIS Report CSV ({chosen_format})",
                    data=csv_data,
                    file_name=f"osis_report_vs_{selected_opponent}_{chosen_format}.csv",
                    mime="text/csv"
                )

                # 📊 Bar Chart
                bar_fig = px.bar(
                    osis_df, x="Player", y="OSIS", color="Remarks", text_auto=True,
                    title=f"OSIS vs {selected_opponent} — {chosen_format}"
                )
                _dark_layout(bar_fig, xlab="Player", ylab="OSIS")
                st.plotly_chart(bar_fig, use_container_width=True)

                # 🐝 Beehive Plot
                st.subheader("🐝 Beehive Plot (Impact vs Opponent)")
                bee_plot = px.strip(
                    osis_df,
                    x="Primary Role",
                    y="OSIS",
                    hover_data=["Player", "Remarks"],
                    color="Remarks",
                    title=f"Beehive Plot of OSIS vs {selected_opponent} — {chosen_format}"
                )
                bee_plot.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                _dark_layout(bee_plot, xlab="Role", ylab="OSIS")
                st.plotly_chart(bee_plot, use_container_width=True)

                # 🔍 Conclusion
                st.subheader("🔍 Conclusion & Insights")
                if len(osis_df) > 0:
                    top_player = osis_df.iloc[0]
                    least_player = osis_df.iloc[-1]
                    st.success(
                        f"🏅 **Top Matchup Performer:** {top_player['Player']} ({top_player['Primary Role']}) "
                        f"with an OSIS of `{top_player['OSIS']:.2f}` against **{selected_opponent}** in {chosen_format}."
                    )
                    st.error(
                        f"📉 **Weakest Matchup Performer:** {least_player['Player']} ({least_player['Primary Role']}) "
                        f"with an OSIS of `{least_player['OSIS']:.2f}` in {chosen_format}."
                    )

                    st.markdown("### 📝 Remarks Summary")
                    summary_counts = osis_df["Remarks"].value_counts().to_dict()
                    for remark, count in summary_counts.items():
                        st.markdown(f"- **{remark}**: {count} player(s)")
                else:
                    st.info("No records found for this opponent.")

            # --- Footer ---
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

        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing_cols))

    else:
        st.info("📁 Please upload a CSV file to proceed with OSIS calculation.")

# ------------------ DISMISSAL MODE VULNERABILITY (DMV) ------------------
elif selected_feature == "Dismissal Mode Vulnerability":
    st.subheader("⚡ Dismissal Mode Vulnerability (DMV) - Weakness Analysis")

    dmv_file = st.file_uploader("📂 Upload CSV with Player Dismissal Stats", type="csv", key="dmv_upload")

    # ---------- Helpers ----------
    def _dark_layout(fig, title=None, xlab=None, ylab=None):
        fig.update_layout(
            plot_bgcolor="#0b132b",
            paper_bgcolor="#0b132b",
            font=dict(color="white"),
            xaxis_title=xlab,
            yaxis_title=ylab,
            title=title,
            legend_title="Legend"
        )
        return fig

    if dmv_file:
        df = pd.read_csv(dmv_file)

        required_columns = ["Player", "Primary Role", "Format", "Opponent", "Dismissal_Mode"]
        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns.")

            # -------------------- DMV Calculation --------------------
            dismissals_count = (
                df.groupby(["Player", "Primary Role", "Format", "Dismissal_Mode"])
                .size()
                .reset_index(name="Dismissal_Count")
            )

            total_dismissals = (
                df.groupby(["Player", "Primary Role", "Format"])
                .size()
                .reset_index(name="Total_Dismissals")
            )

            dmv_all = dismissals_count.merge(
                total_dismissals, on=["Player", "Primary Role", "Format"], how="left"
            )

            dmv_all["DMV"] = (dmv_all["Dismissal_Count"] / dmv_all["Total_Dismissals"]) * 100

            # -------------------- Remarks --------------------
            def add_dmv_remarks(sub):
                max_dmv = sub["DMV"].max() if len(sub) else 0
                def remark(x):
                    if x == max_dmv:
                        return "🚨 Major Weakness"
                    elif x >= 0.8 * max_dmv:
                        return "⚠️ Vulnerable"
                    elif x >= 0.5 * max_dmv:
                        return "😐 Moderate"
                    else:
                        return "✅ Low Risk"
                sub["Remarks"] = sub["DMV"].apply(remark)
                return sub

            dmv_all = dmv_all.groupby(["Player", "Format"], group_keys=False).apply(add_dmv_remarks)

            # -------------------- Format Filter --------------------
            format_list = sorted(dmv_all["Format"].unique().tolist())
            chosen_format = st.selectbox("📌 Select Format", format_list)

            dmv_all_fmt = dmv_all[dmv_all["Format"] == chosen_format]

            # -------------------- View Switcher --------------------
            player_list = sorted(dmv_all_fmt["Player"].unique().tolist())
            view_choice = st.selectbox(
                "🔎 View Mode",
                ["All Players (Summary)"] + player_list,
                help="Choose 'All Players' for an overview or pick a player for a deep-dive."
            )

            # -------------------- ALL PLAYERS (SUMMARY) --------------------
            if view_choice == "All Players (Summary)":
                st.subheader(f"🌐 DMV Summary Across All Players — {chosen_format}")

                # Heatmap
                pivot_dmv = dmv_all_fmt.pivot_table(
                    index="Player", columns="Dismissal_Mode", values="DMV", aggfunc="mean"
                ).fillna(0).round(2)

                import plotly.express as px
                fig_heat = px.imshow(
                    pivot_dmv,
                    labels=dict(x="Dismissal Mode", y="Player", color="DMV %"),
                    aspect="auto",
                    title=f"Dismissal Mode Vulnerability Heatmap ({chosen_format})"
                )
                _dark_layout(fig_heat)
                st.plotly_chart(fig_heat, use_container_width=True)

                # Average DMV by Mode
                avg_by_mode = (
                    dmv_all_fmt.groupby("Dismissal_Mode", as_index=False)["DMV"].mean()
                    .sort_values("DMV", ascending=False)
                )
                fig_avg = px.bar(
                    avg_by_mode, x="Dismissal_Mode", y="DMV", text_auto=True,
                    title=f"Average DMV by Dismissal Mode — {chosen_format}"
                )
                _dark_layout(fig_avg, xlab="Dismissal Mode", ylab="Avg DMV %")
                st.plotly_chart(fig_avg, use_container_width=True)

                # Most Vulnerable Mode per Player
                most_vuln = dmv_all_fmt.loc[dmv_all_fmt.groupby("Player")["DMV"].idxmax()].rename(
                    columns={"Dismissal_Mode": "Most_Vulnerable_Mode", "DMV": "Max_DMV"}
                )
                st.markdown("**Most Vulnerable Mode per Player**")
                st.dataframe(most_vuln[["Player", "Primary Role", "Most_Vulnerable_Mode", "Max_DMV", "Remarks"]])

                # Download
                st.download_button(
                    f"⬇ Download Full DMV ({chosen_format})",
                    data=dmv_all_fmt.to_csv(index=False).encode("utf-8"),
                    file_name=f"dmv_all_players_{chosen_format}.csv",
                    mime="text/csv"
                )

                # Insights
                st.subheader("🔍 Summary Insights")
                top_mode = avg_by_mode.iloc[0]
                low_mode = avg_by_mode.iloc[-1]
                st.info(
                    f"🚨 **Most common dismissal mode in {chosen_format}:** **{top_mode['Dismissal_Mode']}** "
                    f"(avg DMV `{top_mode['DMV']:.2f}%`)\n\n"
                    f"✅ **Least common dismissal mode:** **{low_mode['Dismissal_Mode']}** "
                    f"(avg DMV `{low_mode['DMV']:.2f}%`)."
                )

            # -------------------- SINGLE PLAYER (DEEP-DIVE) --------------------
            else:
                selected_player = view_choice
                st.subheader(f"🧭 Deep-Dive: DMV for **{selected_player}** — {chosen_format}")

                dmv_df = (
                    dmv_all_fmt[dmv_all_fmt["Player"] == selected_player]
                    .copy()
                    .sort_values("DMV", ascending=False)
                    .reset_index(drop=True)
                )

                # 📋 DMV Report
                st.markdown(f"### 📋 DMV Report for {selected_player} — {chosen_format}")
                st.dataframe(dmv_df[["Player", "Primary Role", "Dismissal_Mode", "Dismissal_Count", "DMV", "Remarks"]])

                # ⬇ CSV Download
                csv_data = dmv_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇ Download DMV Report CSV ({chosen_format})",
                    data=csv_data,
                    file_name=f"dmv_report_{selected_player}_{chosen_format}.csv",
                    mime="text/csv"
                )

                # 📊 Bar Chart
                bar_fig = px.bar(
                    dmv_df, x="Dismissal_Mode", y="DMV", color="Remarks", text_auto=True,
                    title=f"Dismissal Mode Vulnerability for {selected_player} — {chosen_format}"
                )
                _dark_layout(bar_fig, xlab="Dismissal Mode", ylab="DMV %")
                st.plotly_chart(bar_fig, use_container_width=True)

                # 🔍 Conclusion
                st.subheader("🔍 Conclusion & Insights")
                if len(dmv_df) > 0:
                    top_vuln = dmv_df.iloc[0]
                    least_vuln = dmv_df.iloc[-1]
                    st.error(
                        f"🚨 **Most Vulnerable:** {selected_player} ({top_vuln['Primary Role']}) "
                        f"is most often dismissed by **{top_vuln['Dismissal_Mode']}** "
                        f"({top_vuln['DMV']:.2f}% of dismissals)."
                    )
                    st.success(
                        f"✅ **Safest Mode:** Rarely dismissed by **{least_vuln['Dismissal_Mode']}** "
                        f"({least_vuln['DMV']:.2f}%)."
                    )

                    st.markdown("### 📝 Remarks Summary")
                    summary_counts = dmv_df["Remarks"].value_counts().to_dict()
                    for remark, count in summary_counts.items():
                        st.markdown(f"- **{remark}**: {count} mode(s)")
                else:
                    st.info("No dismissal data available for this player.")

            # --- Footer ---
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

        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing_cols))

    else:
        st.info("📁 Please upload a CSV file to proceed with DMV calculation.")
