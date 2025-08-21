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
    "Format Specialization Dilemma",
    "Unstable Performance Issue",
    "Transition Mismanagement"
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
                st.markdown("**Top 10 Matchups per Opponent**")
                for opp in opponent_list:
                    sub = osis_all_fmt[osis_all_fmt["Opponent"] == opp].sort_values("OSIS", ascending=False).head(5)
                    st.markdown(f"**🏟️ {opp} — Top 10**")
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

# ------------------ FORMAT SPECIALIZATION DILEMMA ------------------
elif selected_feature == "Format Specialization Dilemma":
    st.subheader("⚖️ Format Specialization Dilemma (FSD) - Cross-Format Consistency")

    fsd_file = st.file_uploader("📂 Upload CSV with Player Match Stats (by Format)", type="csv", key="fsd_upload")

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

    if fsd_file:
        df = pd.read_csv(fsd_file)

        required_columns = [
            "Player", "Primary Role", "Format", "Runs", "Balls_Faced",
            "Wickets", "Overs_Bowled", "Runs_Conceded",
            "Catches", "Run_Outs", "Stumpings"
        ]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns.")

            # -------------------- Impact Calculation Functions --------------------
            def batting_impact(runs, balls, fmt):
                if pd.isna(balls) or balls == 0:
                    return 0.0
                strike_rate = (runs / balls) * 100.0
                if fmt == "Test":
                    return (runs * 0.7) + (strike_rate * 0.3)
                elif fmt == "ODI":
                    return (runs * 0.6) + (strike_rate * 0.4)
                else:  # T20
                    return (runs * 0.4) + (strike_rate * 0.6)

            def bowling_impact(wickets, overs, runs_conceded, fmt):
                if pd.isna(overs) or overs == 0:
                    return 0.0
                economy = runs_conceded / overs
                if fmt == "Test":
                    return (wickets * 20 * 0.6) + ((6 - economy) * 10 * 0.4)
                elif fmt == "ODI":
                    return (wickets * 20 * 0.65) + ((6 - economy) * 10 * 0.35)
                else:  # T20
                    return (wickets * 20 * 0.5) + ((6 - economy) * 10 * 0.5)

            def fielding_impact(catches, run_outs, stumpings, fmt):
                base = (catches * 10) + (run_outs * 12) + (stumpings * 15)
                if fmt == "Test":
                    return base * 0.5
                elif fmt == "ODI":
                    return base * 0.8
                else:
                    return base * 1.0

            def calculate_role_impact(row):
                role = str(row["Primary Role"]).strip().lower()
                fmt = row["Format"]
                runs = row["Runs"]; balls = row["Balls_Faced"]
                wkts = row["Wickets"]; overs = row["Overs_Bowled"]; rc = row["Runs_Conceded"]
                catches = row["Catches"]; ro = row["Run_Outs"]; stp = row["Stumpings"]

                if role == "batter":
                    return batting_impact(runs, balls, fmt)
                elif role == "bowler":
                    return bowling_impact(wkts, overs, rc, fmt)
                elif role == "all-rounder":
                    bat = batting_impact(runs, balls, fmt)
                    bowl = bowling_impact(wkts, overs, rc, fmt)
                    # workload share
                    bat_share = balls / (balls + overs * 6 + 1e-6)
                    bowl_share = 1 - bat_share
                    return (bat * bat_share) + (bowl * bowl_share)
                elif role in ["wk-batter", "wk batter", "wicketkeeper", "wicket-keeper", "wk"]:
                    bat = batting_impact(runs, balls, fmt)
                    wk_field = fielding_impact(catches, ro, stp, fmt)
                    return bat + (wk_field * 0.5)
                else:
                    return batting_impact(runs, balls, fmt)

            # Safety fill for numeric columns
            num_cols = ["Runs", "Balls_Faced", "Wickets", "Overs_Bowled", "Runs_Conceded", "Catches", "Run_Outs", "Stumpings"]
            df[num_cols] = df[num_cols].fillna(0)

            # Per-match impact
            df["Impact"] = df.apply(calculate_role_impact, axis=1)

            # -------------------- FSD Calculation --------------------
            overall = (
                df.groupby(["Player", "Primary Role"], as_index=False)["Impact"]
                .mean()
                .rename(columns={"Impact": "Overall_Avg_Impact"})
            )

            vs_format = (
                df.groupby(["Player", "Primary Role", "Format"], as_index=False)["Impact"]
                .mean()
                .rename(columns={"Impact": "Format_Avg_Impact"})
            )

            fsd_all = vs_format.merge(overall, on=["Player", "Primary Role"], how="left")

            def compute_fsd(row):
                if row["Overall_Avg_Impact"] <= 20:  # threshold
                    return 0.0
                if row["Overall_Avg_Impact"] in [0, None]:
                    return 0.0
                return (row["Format_Avg_Impact"] / row["Overall_Avg_Impact"] * 100.0)

            fsd_all["FSD"] = fsd_all.apply(compute_fsd, axis=1)

            # -------------------- Remarks per Format --------------------
            def add_remarks_per_format(fsd_df):
                out = []
                for fmt, sub in fsd_df.groupby("Format"):
                    sub = sub.copy()
                    max_fsd = sub["FSD"].max() if len(sub) else 0

                    def remark(x):
                        if x >= 120 and x == max_fsd:
                            return "🏆 Format Specialist"
                        elif x >= 100:
                            return "🔥 Strong"
                        elif x >= 80:
                            return "✅ Balanced"
                        elif x >= 60:
                            return "⚠️ Average"
                        else:
                            return "🔻 Weak"

                    sub["Remarks"] = sub["FSD"].apply(remark)
                    out.append(sub)
                return pd.concat(out, ignore_index=True) if out else fsd_df

            fsd_all = add_remarks_per_format(fsd_all)

            # -------------------- View Switcher --------------------
            format_list = sorted(fsd_all["Format"].unique().tolist())
            view_choice = st.selectbox(
                "🔎 View Mode",
                ["All Formats (Summary)"] + format_list,
                help="Choose 'All Formats' for an overview or pick a format for a deep-dive."
            )

            # -------------------- ALL FORMATS (SUMMARY) --------------------
            if view_choice == "All Formats (Summary)":
                st.subheader("🌐 Summary Across All Formats")

                # Heatmap
                pivot_fsd = fsd_all.pivot_table(index="Player", columns="Format", values="FSD", aggfunc="mean").fillna(0).round(2)
                st.markdown("**FSD Heatmap (Players × Formats)**")
                import plotly.express as px
                fig_heat = px.imshow(
                    pivot_fsd,
                    labels=dict(x="Format", y="Player", color="FSD"),
                    aspect="auto",
                    title="FSD Heatmap Across Formats"
                )
                _dark_layout(fig_heat)
                st.plotly_chart(fig_heat, use_container_width=True)

                # Average FSD per Format
                avg_vs_fmt = fsd_all.groupby("Format", as_index=False)["FSD"].mean().sort_values("FSD", ascending=False)
                fig_avg = px.bar(
                    avg_vs_fmt, x="Format", y="FSD", text_auto=True,
                    title="Average FSD by Format (Team-wide)"
                )
                _dark_layout(fig_avg, xlab="Format", ylab="Average FSD")
                st.plotly_chart(fig_avg, use_container_width=True)

                # Best/Worst Format per Player
                best = fsd_all.loc[fsd_all.groupby("Player")["FSD"].idxmax()].rename(columns={"Format": "Best_Format", "FSD": "Best_FSD"})
                worst = fsd_all.loc[fsd_all.groupby("Player")["FSD"].idxmin()].rename(columns={"Format": "Worst_Format", "FSD": "Worst_FSD"})
                bw = best[["Player", "Primary Role", "Best_Format", "Best_FSD"]].merge(
                    worst[["Player", "Worst_Format", "Worst_FSD"]],
                    on="Player", how="left"
                ).sort_values(["Primary Role", "Player"])
                st.markdown("**Best vs Worst Format per Player**")
                st.dataframe(bw.reset_index(drop=True))

                # Top 5 per Format
                st.markdown("**Top 5 Performers per Format**")
                for fmt in format_list:
                    sub = fsd_all[fsd_all["Format"] == fmt].sort_values("FSD", ascending=False).head(5)
                    st.markdown(f"**🏏 {fmt} — Top 5**")
                    st.dataframe(sub[["Player", "Primary Role", "Format_Avg_Impact", "Overall_Avg_Impact", "FSD", "Remarks"]].reset_index(drop=True))

                # Downloads
                st.download_button(
                    "⬇ Download Full FSD (All Formats)",
                    data=fsd_all.to_csv(index=False).encode("utf-8"),
                    file_name="fsd_all_formats.csv",
                    mime="text/csv"
                )
                st.download_button(
                    "⬇ Download FSD Heatmap (table form)",
                    data=pivot_fsd.to_csv().encode("utf-8"),
                    file_name="fsd_heatmap_table.csv",
                    mime="text/csv"
                )

                # Insights
                st.subheader("🔍 Summary Insights")
                top_fmt = avg_vs_fmt.iloc[0]
                low_fmt = avg_vs_fmt.iloc[-1]
                st.info(
                    f"✅ **Best format for team-wide impact:** **{top_fmt['Format']}** (avg FSD `{top_fmt['FSD']:.2f}`)\n\n"
                    f"⚠️ **Most challenging format:** **{low_fmt['Format']}** (avg FSD `{low_fmt['FSD']:.2f}`)."
                )

            # -------------------- SINGLE FORMAT (DEEP-DIVE) --------------------
            else:
                selected_format = view_choice
                st.subheader(f"🧭 Deep-Dive: FSD in **{selected_format}**")

                fsd_df = (
                    fsd_all[fsd_all["Format"] == selected_format]
                    .copy()
                    .sort_values("FSD", ascending=False)
                    .reset_index(drop=True)
                )

                # 📋 FSD Report
                st.markdown(f"### 📋 FSD Report in {selected_format}")
                st.dataframe(fsd_df[["Player", "Primary Role", "Overall_Avg_Impact", "Format_Avg_Impact", "FSD", "Remarks"]])

                # ⬇ CSV Download
                csv_data = fsd_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download FSD Report CSV",
                    data=csv_data,
                    file_name=f"fsd_report_in_{selected_format}.csv",
                    mime="text/csv"
                )

                # 📊 Bar Chart
                bar_fig = px.bar(
                    fsd_df, x="Player", y="FSD", color="Remarks", text_auto=True,
                    title=f"FSD in {selected_format}"
                )
                _dark_layout(bar_fig, xlab="Player", ylab="FSD")
                st.plotly_chart(bar_fig, use_container_width=True)

                # 🐝 Beehive Plot
                st.subheader("🐝 Beehive Plot (Impact in Format)")
                bee_plot = px.strip(
                    fsd_df,
                    x="Primary Role",
                    y="FSD",
                    hover_data=["Player", "Remarks"],
                    color="Remarks",
                    stripmode="overlay",
                    title=f"Beehive Plot of FSD in {selected_format}"
                )
                bee_plot.update_traces(jitter=0.35, marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                _dark_layout(bee_plot, xlab="Role", ylab="FSD")
                st.plotly_chart(bee_plot, use_container_width=True)

                # 🔍 Conclusion
                st.subheader("🔍 Conclusion & Insights")
                if len(fsd_df) > 0:
                    top_player = fsd_df.iloc[0]
                    least_player = fsd_df.iloc[-1]
                    st.success(
                        f"🏅 **Top Format Specialist:** {top_player['Player']} ({top_player['Primary Role']}) "
                        f"with an FSD of `{top_player['FSD']:.2f}` in **{selected_format}**."
                    )
                    st.error(
                        f"📉 **Weakest Performer in {selected_format}:** {least_player['Player']} ({least_player['Primary Role']}) "
                        f"with an FSD of `{least_player['FSD']:.2f}`."
                    )

                    st.markdown("### 📝 Remarks Summary")
                    summary_counts = fsd_df["Remarks"].value_counts().to_dict()
                    for remark, count in summary_counts.items():
                        st.markdown(f"- **{remark}**: {count} player(s)")
                else:
                    st.info("No records found for this format.")

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
        st.info("📁 Please upload a CSV file to proceed with FSD calculation.")

# ------------------ UNSTABLE PERFORMANCE ISSUE ------------------
elif selected_feature == "Unstable Performance Issue":
    st.subheader("🌪️ Unstable Performance Issue (UPI) - Variability & Consistency Analysis Across Formats")

    upi_file = st.file_uploader("📂 Upload CSV with Player Match-by-Match Stats (Multi-Format Supported)", type="csv", key="upi_upload")

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

    if upi_file:
        df = pd.read_csv(upi_file)

        required_columns = [
            "Player", "Primary Role", "Format", "Runs", "Balls_Faced",
            "Wickets", "Overs_Bowled", "Runs_Conceded",
            "Catches", "Run_Outs", "Stumpings"
        ]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns.")

            # -------------------- Match Metric Functions --------------------
            def batting_metric(runs, balls):
                if pd.isna(balls) or balls == 0:
                    return 0.0
                strike_rate = (runs / balls) * 100
                return runs * 0.6 + strike_rate * 0.4

            def bowling_metric(wkts, overs, runs_conceded):
                if pd.isna(overs) or overs == 0:
                    return 0.0
                econ = runs_conceded / overs
                return (wkts * 20) - (econ * 2)

            def fielding_metric(catches, run_outs, stumpings):
                return (catches * 8) + (run_outs * 10) + (stumpings * 12)

            def calculate_metric(row):
                role = str(row["Primary Role"]).strip().lower()
                bat = batting_metric(row["Runs"], row["Balls_Faced"])
                bowl = bowling_metric(row["Wickets"], row["Overs_Bowled"], row["Runs_Conceded"])
                fld = fielding_metric(row["Catches"], row["Run_Outs"], row["Stumpings"])

                if role == "batter":
                    return bat
                elif role == "bowler":
                    return bowl
                elif role == "all-rounder":
                    return (bat + bowl) / 2.0
                elif role in ["wk-batter", "wk batter", "wicketkeeper", "wicket-keeper", "wk"]:
                    return bat + (fld * 0.5)
                else:
                    return bat  # default fallback

            # Fill missing numeric values
            num_cols = ["Runs", "Balls_Faced", "Wickets", "Overs_Bowled", "Runs_Conceded", "Catches", "Run_Outs", "Stumpings"]
            df[num_cols] = df[num_cols].fillna(0)

            df["Match_Metric"] = df.apply(calculate_metric, axis=1)

            # -------------------- UPI Calculation --------------------
            import numpy as np

            def instability_measures(sub):
                values = sub["Match_Metric"].tolist()
                if len(values) < 2:
                    return pd.Series({
                        "Mean": np.mean(values),
                        "SD": 0,
                        "CV": 0,
                        "FailureRate": 0,
                        "AvgMovingRange": 0,
                        "UPI": 0
                    })

                mean_val = np.mean(values)
                sd_val = np.std(values, ddof=1)
                cv_val = sd_val / mean_val if mean_val != 0 else 0
                failures = sum(1 for v in values if v < 10) / len(values) * 100
                mr = np.mean([abs(values[i] - values[i-1]) for i in range(1, len(values))])
                upi_score = (cv_val * 50) + (failures * 0.3) + (mr * 0.2)

                return pd.Series({
                    "Mean": mean_val,
                    "SD": sd_val,
                    "CV": cv_val,
                    "FailureRate": failures,
                    "AvgMovingRange": mr,
                    "UPI": upi_score
                })

            # 🔑 Now group by Format also
            upi_all = df.groupby(["Player", "Primary Role", "Format"]).apply(instability_measures).reset_index()

            # -------------------- Remarks --------------------
            def add_remarks(upi_df):
                max_upi = upi_df["UPI"].max() if len(upi_df) else 0
                def remark(x):
                    if x == 0:
                        return "✅ Stable (No Variability)"
                    elif x <= 0.3 * max_upi:
                        return "✅ Stable"
                    elif x <= 0.6 * max_upi:
                        return "⚠️ Moderate Instability"
                    elif x <= 0.8 * max_upi:
                        return "🔥 High Instability"
                    else:
                        return "🌪️ Very Unstable"
                upi_df["Remarks"] = upi_df["UPI"].apply(remark)
                return upi_df

            upi_all = add_remarks(upi_all)

            # -------------------- View Mode --------------------
            view_choice = st.selectbox(
                "🔎 View Mode",
                ["Summary View", "Format-wise Deep-Dive", "Player-wise Deep-Dive"],
                help="Choose 'Summary' for overview, 'Format-wise' for breakdown, or 'Player-wise' for individual analysis."
            )

            import plotly.express as px
            import plotly.graph_objects as go

            if view_choice == "Summary View":
                st.subheader("🌐 Summary of Instability (UPI)")

                # Average UPI per Role × Format
                avg_upi = upi_all.groupby(["Primary Role", "Format"], as_index=False)["UPI"].mean().sort_values("UPI", ascending=False)
                fig_avg = px.bar(
                    avg_upi, x="Primary Role", y="UPI", color="Format", barmode="group",
                    title="Average UPI by Role and Format", text_auto=True
                )
                _dark_layout(fig_avg, xlab="Role", ylab="Average UPI")
                st.plotly_chart(fig_avg, use_container_width=True)

                # 🐝 Beehive Plot Replacement (simulate jitter with scatter)
                import numpy as np
                jittered = upi_all.copy()
                jittered["jitter_x"] = jittered["Primary Role"].apply(lambda r: list(upi_all["Primary Role"].unique()).index(r))
                jittered["jitter_x"] = jittered["jitter_x"] + np.random.uniform(-0.2, 0.2, size=len(jittered))

                fig_bee = px.scatter(
                    jittered, x="jitter_x", y="UPI", color="Format",
                    hover_data=["Player", "Primary Role", "Format"],
                    title="Beehive Plot of UPI Distribution Across Roles & Formats"
                )
                fig_bee.update_xaxes(
                    tickvals=list(range(len(upi_all["Primary Role"].unique()))),
                    ticktext=upi_all["Primary Role"].unique()
                )
                _dark_layout(fig_bee, xlab="Role", ylab="UPI")
                st.plotly_chart(fig_bee, use_container_width=True)

                # Downloads
                st.download_button(
                    "⬇ Download Full UPI Report",
                    data=upi_all.to_csv(index=False).encode("utf-8"),
                    file_name="upi_all_players.csv",
                    mime="text/csv"
                )

                # Insights
                st.subheader("🔍 Summary Insights")
                top_unstable = upi_all.loc[upi_all["UPI"].idxmax()]
                low_unstable = upi_all.loc[upi_all["UPI"].idxmin()]
                st.info(
                    f"🌪️ **Most Unstable Player:** {top_unstable['Player']} ({top_unstable['Primary Role']} - {top_unstable['Format']}) "
                    f"with UPI `{top_unstable['UPI']:.2f}`\n\n"
                    f"✅ **Most Stable Player:** {low_unstable['Player']} ({low_unstable['Primary Role']} - {low_unstable['Format']}) "
                    f"with UPI `{low_unstable['UPI']:.2f}`."
                )

            elif view_choice == "Format-wise Deep-Dive":
                st.subheader("📊 Format-wise Instability Deep-Dive")
                selected_format = st.selectbox("Select Format", sorted(upi_all["Format"].unique().tolist()))
                format_df = upi_all[upi_all["Format"] == selected_format]

                fig_box = px.box(format_df, x="Primary Role", y="UPI", color="Primary Role",
                                 title=f"UPI Distribution by Role in {selected_format}")
                _dark_layout(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)

                # Beehive replacement
                jittered_fmt = format_df.copy()
                jittered_fmt["jitter_x"] = jittered_fmt["Primary Role"].apply(lambda r: list(format_df["Primary Role"].unique()).index(r))
                jittered_fmt["jitter_x"] = jittered_fmt["jitter_x"] + np.random.uniform(-0.2, 0.2, size=len(jittered_fmt))

                fig_bee_fmt = px.scatter(
                    jittered_fmt, x="jitter_x", y="UPI", color="Primary Role",
                    hover_data=["Player", "Format"],
                    title=f"Beehive Plot of UPI in {selected_format}"
                )
                fig_bee_fmt.update_xaxes(
                    tickvals=list(range(len(format_df["Primary Role"].unique()))),
                    ticktext=format_df["Primary Role"].unique()
                )
                _dark_layout(fig_bee_fmt, xlab="Role", ylab="UPI")
                st.plotly_chart(fig_bee_fmt, use_container_width=True)

                st.dataframe(format_df)

            else:
                st.subheader("🧭 Player-wise Deep-Dive on Instability")
                selected_player = st.selectbox("Select Player", sorted(upi_all["Player"].unique().tolist()))
                sub_df = df[df["Player"] == selected_player].reset_index(drop=True)

                if len(sub_df) > 0:
                    st.markdown(f"### 📋 Match-by-Match Metrics for {selected_player}")
                    st.dataframe(sub_df[["Player", "Primary Role", "Format", "Runs", "Balls_Faced", "Wickets", "Overs_Bowled", "Runs_Conceded", "Catches", "Run_Outs", "Stumpings", "Match_Metric"]])

                    # Trend chart
                    fig_line = go.Figure()
                    for fmt in sub_df["Format"].unique():
                        temp = sub_df[sub_df["Format"] == fmt]
                        fig_line.add_trace(go.Scatter(
                            y=temp["Match_Metric"],
                            x=list(range(1, len(temp) + 1)),
                            mode="lines+markers",
                            name=f"{fmt}"
                        ))
                    _dark_layout(fig_line, xlab="Match #", ylab="Metric", title=f"Match-to-Match Instability for {selected_player}")
                    st.plotly_chart(fig_line, use_container_width=True)

                    # Beehive replacement
                    jittered_player = sub_df.copy()
                    jittered_player["jitter_x"] = jittered_player["Format"].apply(lambda f: list(sub_df["Format"].unique()).index(f))
                    jittered_player["jitter_x"] = jittered_player["jitter_x"] + np.random.uniform(-0.2, 0.2, size=len(jittered_player))

                    fig_bee_player = px.scatter(
                        jittered_player, x="jitter_x", y="Match_Metric", color="Format",
                        hover_data=["Player", "Format"],
                        title=f"Beehive Plot of Match Metrics for {selected_player}"
                    )
                    fig_bee_player.update_xaxes(
                        tickvals=list(range(len(sub_df["Format"].unique()))),
                        ticktext=sub_df["Format"].unique()
                    )
                    _dark_layout(fig_bee_player, xlab="Format", ylab="Match Metric")
                    st.plotly_chart(fig_bee_player, use_container_width=True)

                    # Player summary
                    player_summary = upi_all[upi_all["Player"] == selected_player]
                    st.markdown("### 📝 Player Instability Summary")
                    st.dataframe(player_summary)
                else:
                    st.info("No records found for this player.")

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
        st.info("📁 Please upload a CSV file to proceed with UPI calculation.")

# ------------------ TRANSITION MISMANAGEMENT INDEX ------------------
elif selected_feature == "Transition Mismanagement":
    st.subheader("🔄 Transition Mismanagement Index (TMI) - Collapse & Phase Handling")

    tmi_file = st.file_uploader("📂 Upload CSV with Ball-by-Ball or Partnership Stats", type="csv", key="tmi_upload")

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

    if tmi_file:
        df = pd.read_csv(tmi_file)

        required_columns = [
            "Match_ID", "Player", "NonStriker", "Primary Role", "Format",
            "Over", "Runs_Scored", "Wickets_Fallen", "Partnership_Runs"
        ]

        if all(col in df.columns for col in required_columns):
            st.success("✅ File loaded with all required columns, including NonStriker.")

            # -------------------- Format-Aware Parameters --------------------
            format_params = {
                "T20": {"collapse_window": 3, "recovery_thresh": 20, "phases": [(1,6),(7,15),(16,20)]},
                "ODI": {"collapse_window": 5, "recovery_thresh": 30, "phases": [(1,10),(11,40),(41,50)]},
            }

            # -------------------- TMI Metric Functions --------------------
            import numpy as np

            def collapse_sensitivity(sub, fmt):
                """% of times ≥2 wickets fell in collapse_window overs after dismissal"""
                window = format_params.get(fmt, format_params["ODI"])["collapse_window"]
                dismissals = sub[sub["Wickets_Fallen"] > 0]
                count, trigger = 0, 0
                for i in dismissals.index:
                    trigger += 1
                    post_window = sub.loc[i+1:i+window, "Wickets_Fallen"].sum()
                    if post_window >= 2:
                        count += 1
                return (count / trigger * 100) if trigger > 0 else 0

            def momentum_drop(sub, fmt):
                """Max drop between sequential phases"""
                phases = format_params.get(fmt, format_params["ODI"])["phases"]
                phase_rates = []
                for (start, end) in phases:
                    phase_runs = sub[(sub["Over"] >= start) & (sub["Over"] <= end)]["Runs_Scored"].sum()
                    phase_overs = len(sub[(sub["Over"] >= start) & (sub["Over"] <= end)])
                    rr = phase_runs / max(1, phase_overs)
                    phase_rates.append(rr)
                # check sequential drops
                drops = [max(0, phase_rates[i] - phase_rates[i+1]) for i in range(len(phase_rates)-1)]
                return max(drops) if drops else 0

            def recovery_failure_rate(sub, fmt):
                """% of times next partnership < threshold"""
                thresh = format_params.get(fmt, format_params["ODI"])["recovery_thresh"]
                total, fails = 0, 0
                for val in sub["Partnership_Runs"]:
                    total += 1
                    if val < thresh:
                        fails += 1
                return (fails / total * 100) if total > 0 else 0

            def tmi_measures(sub):
                fmt = sub["Format"].iloc[0] if len(sub) > 0 else "ODI"
                cs = collapse_sensitivity(sub, fmt)
                md = momentum_drop(sub, fmt)
                rfr = recovery_failure_rate(sub, fmt)

                # Normalize each to 0-100
                cs_n, md_n, rfr_n = min(cs,100), min(md*10,100), min(rfr,100)

                tmi_score = (cs_n * 0.4) + (md_n * 0.3) + (rfr_n * 0.3)
                return pd.Series({
                    "CollapseSensitivity": cs,
                    "MomentumDrop": md,
                    "RecoveryFailureRate": rfr,
                    "TMI": tmi_score
                })

            # 🔑 Group by Partnership × Format
            tmi_all = df.groupby(["Player", "NonStriker", "Primary Role", "Format"]).apply(tmi_measures).reset_index()

            # -------------------- Remarks --------------------
            def add_remarks(tmi_df):
                max_tmi = tmi_df["TMI"].max() if len(tmi_df) else 0
                def remark(x):
                    if x == 0:
                        return "✅ Excellent Transition Handling"
                    elif x <= 0.3 * max_tmi:
                        return "✅ Stable"
                    elif x <= 0.6 * max_tmi:
                        return "⚠️ Moderate Mismanagement"
                    elif x <= 0.8 * max_tmi:
                        return "🔥 High Mismanagement"
                    else:
                        return "🌪️ Very Poor Transition Handling"
                tmi_df["Remarks"] = tmi_df["TMI"].apply(remark)
                return tmi_df

            tmi_all = add_remarks(tmi_all)

            # -------------------- View Modes --------------------
            view_choice = st.selectbox(
                "🔎 View Mode",
                ["Summary View", "Format-wise Deep-Dive", "Partnership-wise Deep-Dive"],
                help="Choose 'Summary' for overview, 'Format-wise' for breakdown, or 'Partnership-wise' for individual batter pair analysis."
            )

            import plotly.express as px

            if view_choice == "Summary View":
                st.subheader("🌐 Summary of Transition Mismanagement (TMI)")

                avg_tmi = tmi_all.groupby(["Primary Role", "Format"], as_index=False)["TMI"].mean().sort_values("TMI", ascending=False)
                fig_avg = px.bar(
                    avg_tmi, x="Primary Role", y="TMI", color="Format", barmode="group",
                    title="Average TMI by Role and Format", text_auto=True
                )
                _dark_layout(fig_avg, xlab="Role", ylab="Average TMI")
                st.plotly_chart(fig_avg, use_container_width=True)

                # Beehive plot
                jittered = tmi_all.copy()
                jittered["jitter_x"] = jittered["Primary Role"].apply(lambda r: list(tmi_all["Primary Role"].unique()).index(r))
                jittered["jitter_x"] = jittered["jitter_x"] + np.random.uniform(-0.2, 0.2, size=len(jittered))

                fig_bee = px.scatter(
                    jittered, x="jitter_x", y="TMI", color="Format",
                    hover_data=["Player", "NonStriker", "Primary Role", "Format"],
                    title="Beehive Plot of TMI Distribution"
                )
                fig_bee.update_xaxes(
                    tickvals=list(range(len(tmi_all["Primary Role"].unique()))),
                    ticktext=tmi_all["Primary Role"].unique()
                )
                _dark_layout(fig_bee, xlab="Role", ylab="TMI")
                st.plotly_chart(fig_bee, use_container_width=True)

                st.download_button(
                    "⬇ Download Full TMI Report",
                    data=tmi_all.to_csv(index=False).encode("utf-8"),
                    file_name="tmi_all_partnerships.csv",
                    mime="text/csv"
                )

            elif view_choice == "Format-wise Deep-Dive":
                st.subheader("📊 Format-wise TMI Deep-Dive")
                selected_format = st.selectbox("Select Format", sorted(tmi_all["Format"].unique().tolist()))
                format_df = tmi_all[tmi_all["Format"] == selected_format]

                fig_box = px.box(format_df, x="Primary Role", y="TMI", color="Primary Role",
                                 title=f"TMI Distribution by Role in {selected_format}")
                _dark_layout(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)

                st.dataframe(format_df)

            else:
                st.subheader("🤝 Partnership-wise Deep-Dive on TMI")
                selected_pair = st.selectbox(
                    "Select Partnership",
                    sorted(tmi_all[["Player", "NonStriker"]].apply(lambda x: f"{x['Player']} & {x['NonStriker']}", axis=1).unique().tolist())
                )

                p1, p2 = selected_pair.split(" & ")
                sub_df = df[((df["Player"] == p1) & (df["NonStriker"] == p2)) | ((df["Player"] == p2) & (df["NonStriker"] == p1))].reset_index(drop=True)

                if len(sub_df) > 0:
                    st.markdown(f"### 📋 Match Data for Partnership: {selected_pair}")
                    st.dataframe(sub_df[["Match_ID", "Over", "Player", "NonStriker", "Runs_Scored", "Wickets_Fallen", "Partnership_Runs"]])

                    pair_summary = tmi_all[((tmi_all["Player"] == p1) & (tmi_all["NonStriker"] == p2)) | ((tmi_all["Player"] == p2) & (tmi_all["NonStriker"] == p1))]
                    st.markdown("### 📝 Partnership TMI Summary")
                    st.dataframe(pair_summary)
                else:
                    st.info("No records found for this partnership.")

            # --- Footer ---
            st.markdown("---")
            st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)

        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing_cols))

    else:
        st.info("📁 Please upload a CSV file to proceed with TMI calculation.")
