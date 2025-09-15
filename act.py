# --------------------------------------------------
# TrueXI App
# Author: Nihira Khare
# Date: July 2025 (updated Sept 2025)
# --------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
import random

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
        r = requests.get(url, timeout=6)
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
    "Opponent-Specific Impact Scores"
    ])

# ------------------ HEADER ------------------
st.image("app logo.png", width=150)
st.markdown("<h1>🏏 True XI</h1>", unsafe_allow_html=True)
st.markdown("<h4>Make Data-Driven Cricket Selections Without Bias</h4>", unsafe_allow_html=True)
if lottie_cricket:
    try:
        st_lottie(lottie_cricket, height=150, key="cricket_header")
    except Exception:
        pass

# ------------------ UTILITY FUNCTIONS ------------------
def safe_scale(column):
    """Return numpy array scaled to [0,1]. If constant, return 0.5 array."""
    col = pd.Series(column).fillna(0).astype(float)
    if len(np.unique(col)) > 1:
        return MinMaxScaler().fit_transform(col.values.reshape(-1, 1)).flatten()
    return np.full(len(col), 0.5)

def convert_social_media_value(val):
    """Convert strings like '2.5M', '300K' to numbers; keep numeric as-is."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        v = val.strip().upper().replace(',', '')
        # strip currency / plus signs
        v = re.sub(r'[^\d\.KBM]', '', v)
        try:
            if v.endswith("M"):
                return float(v[:-1]) * 1_000_000
            elif v.endswith("K"):
                return float(v[:-1]) * 1_000
            elif v.endswith("B"):
                return float(v[:-1]) * 1_000_000_000
            else:
                return float(v)
        except:
            return 0.0
    return 0.0

def parse_numeric_currency(val):
    """Parse salary/price-like strings to numeric rupees/dollars (best-effort)."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # remove currency symbols
    s = re.sub(r'[^\dKMkmb\.]', '', s)
    s = s.upper()
    try:
        if s.endswith('M'):
            return float(s[:-1]) * 1_000_000
        if s.endswith('K'):
            return float(s[:-1]) * 1_000
        return float(s)
    except:
        try:
            return float(re.sub(r'[^\d\.]', '', s))
        except:
            return np.nan

# ------------------ CORE METRIC FUNCTIONS ------------------
def compute_performance(row):
    """Role-aware performance calculation (same as your original but safer)."""
    role = str(row.get("Primary Role", "")).strip().lower()
    # use pre-scaled columns expected by calling code
    if role in ["batter", "wk-batter"]:
        return row.get("Batting Avg (scaled)", 0) * 0.6 + row.get("Batting SR (scaled)", 0) * 0.4
    elif role == "bowler":
        return row.get("Wickets (scaled)", 0) * 0.6 + row.get("Bowling Econ (scaled)", 0) * 0.4
    elif role == "all-rounder":
        batting = row.get("Batting Avg (scaled)", 0) * 0.3 + row.get("Batting SR (scaled)", 0) * 0.2
        bowling = row.get("Wickets (scaled)", 0) * 0.3 + row.get("Bowling Econ (scaled)", 0) * 0.2
        return batting + bowling
    return 0.0

def calculate_leadership_score(df):
    df = df.copy()
    df["Leadership_Score"] = 0.6 * df["Performance_score"] + 0.4 * df["Fame_score"]
    return df

# Validate final XI against official squad (keeps your implementation)
def validate_and_fix_squad(final_xi, official_squad, role_dict):
    validated_xi = []
    replacements = []
    for _, row in final_xi.iterrows():
        player_name = row["Player Name"]
        player_role = row["Primary Role"]
        if player_name not in official_squad:
            possible_replacements = [p for p in official_squad if role_dict.get(p, "") == player_role]
            if possible_replacements:
                replacement = possible_replacements[0]
                replacements.append((player_name, replacement, player_role))
                row["Player Name"] = replacement
                validated_xi.append(row)
            else:
                fallback_choices = [p for p in official_squad if p not in [v["Player Name"] if isinstance(v, dict) else v for v in validated_xi]]
                if fallback_choices:
                    replacement = fallback_choices[0]
                    replacements.append((player_name, replacement, "fallback"))
                    row["Player Name"] = replacement
                    row["Primary Role"] = role_dict.get(replacement, "Unknown")
                    validated_xi.append(row)
                else:
                    # no replacement available; keep original row
                    validated_xi.append(row)
        else:
            validated_xi.append(row)
    return pd.DataFrame(validated_xi), replacements

# ------------------ PAYMENT PERFORMANCE (Value-for-Money) ------------------
def compute_payment_performance(df):
    """
    Compute Payment Performance (value-for-money).
    Priority:
      - If dataset contains 'Salary' or 'Auction Price' or similar, use it (normalized).
      - Else fallback to Performance_score / (1 + Fame_score) as value-for-money proxy.
    Output column: 'Payment_Performance' (higher = better value)
    """
    df = df.copy()
    # detect salary-like columns
    salary_cols = [c for c in df.columns if re.search(r'salary|price|auction|fee|cost', c, re.IGNORECASE)]
    salary_col = salary_cols[0] if salary_cols else None

    if salary_col:
        # parse numeric salary
        df['Salary_num'] = df[salary_col].apply(parse_numeric_currency).fillna(np.nan)
        # if all NaN, fallback
        if df['Salary_num'].notna().sum() > 0:
            # smaller salary => better value if performance same; so we invert salary after normalization
            df['Salary_norm'] = safe_scale(df['Salary_num'])
            # Payment_Performance = Performance_score / (1 + Salary_norm)
            df['Payment_Performance'] = df['Performance_score'] / (1.0 + df['Salary_norm'])
            return df
    # fallback: Performance_score divided by (1 + Fame_score) to penalize fame-heavy picks
    df['Payment_Performance'] = df['Performance_score'] / (1.0 + df['Fame_score'])
    return df

# ------------------ MAIN APP FLOW ------------------
if selected_feature == "Main App Flow":
    st.subheader("🏏 Main App Flow")

    if st.session_state.step == 0:
        uploaded_file = st.file_uploader("📁 Upload Final XI CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.dropna(how='all', inplace=True)
            # normalize role
            if "Primary Role" in df.columns:
                df["Primary Role"] = df["Primary Role"].astype(str).str.strip().str.lower()
            else:
                df["Primary Role"] = df.get("Role", "batter").astype(str).str.strip().str.lower()

            # Socials
            if "Twitter Followers" in df.columns and "Instagram Followers" in df.columns:
                df["Twitter Followers"] = df["Twitter Followers"].apply(convert_social_media_value)
                df["Instagram Followers"] = df["Instagram Followers"].apply(convert_social_media_value)
                df["Social Media Reach"] = df["Twitter Followers"] + df["Instagram Followers"]

            if "Social Media Reach" in df.columns and df["Social Media Reach"].dtype == object:
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
            st.subheader("📋 Uploaded Player Data")
            format_filter = st.selectbox("Filter by Format", options=df["Format"].unique())
            role_filter = st.multiselect("Filter by Primary Role", options=df["Primary Role"].unique(), default=df["Primary Role"].unique())
            filtered_df = df[(df["Format"] == format_filter) & (df["Primary Role"].isin(role_filter))]
            st.dataframe(filtered_df)

        if st.button("Detect Biased Players"):
            st.subheader("🕵️‍♂️ Biased Players Detected")

            format_filter = st.selectbox("🎯 Choose Format for Bias Detection", df["Format"].unique())
            df = df[df["Format"] == format_filter].copy()

            # scaled metrics
            df["Batting Avg (scaled)"] = safe_scale(df["Batting Avg"])
            df["Batting SR (scaled)"] = safe_scale(df["Batting SR"])
            df["Wickets (scaled)"] = safe_scale(df["Wickets"])
            df["Bowling Econ (scaled)"] = 1 - safe_scale(df["Bowling Economy"])

            # performance & fame
            df["Performance_score_raw"] = df.apply(compute_performance, axis=1)
            df["Performance_score"] = safe_scale(df["Performance_score_raw"])
            df["Google Trends (scaled)"] = safe_scale(df["Google Trends Score"])
            if "Social Media Reach" not in df.columns:
                # fallback compute from socials if present
                if "Instagram Followers" in df.columns and "Twitter Followers" in df.columns:
                    df["Instagram Followers"] = df["Instagram Followers"].apply(convert_social_media_value)
                    df["Twitter Followers"] = df["Twitter Followers"].apply(convert_social_media_value)
                    df["Social Media Reach"] = df["Instagram Followers"] + df["Twitter Followers"]
                else:
                    df["Social Media Reach"] = 0
            df["Social Media Reach (scaled)"] = safe_scale(df["Social Media Reach"])

            df["Fame_score"] = (
                df["Google Trends (scaled)"] * 0.5 +
                df["Social Media Reach (scaled)"] * 0.5
            )

            df["bias_score"] = df["Fame_score"] - df["Performance_score"]

            df["Is_Biased"] = (df["Fame_score"] > 0.7) & (df["Performance_score"] < 0.4)

            # compute payment performance (value for money)
            df = compute_payment_performance(df)

            st.session_state.df = df

            st.dataframe(df[df["Is_Biased"]][[
                "Player Name", "Primary Role", "Fame_score", "Performance_score", "bias_score", "Is_Biased"
            ]])

            # scatter
            fig = px.scatter(
                df, x="Fame_score", y="Performance_score", color="Is_Biased",
                hover_data=["Player Name", "Primary Role"],
                title="Fame vs Performance Bias Map"
            )
            fig.update_layout(
                xaxis_title="Fame Score",
                yaxis_title="Performance Score",
                legend_title="Bias Status",
                plot_bgcolor="#0b132b",
                paper_bgcolor="#0b132b",
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)

            # Beehive / Beeswarm
            st.subheader("🐝 Beehive View: Performance by Primary Role")
            df["Primary Role_num"] = df["Primary Role"].astype('category').cat.codes
            df["Jittered_x"] = df["Primary Role_num"] + np.random.normal(0, 0.15, size=len(df))

            beehive_fig = px.scatter(
                df, x="Jittered_x", y="Performance_score",
                color="Primary Role", hover_data=["Player Name", "Fame_score", "Is_Biased"],
                title="Simulated Beehive Plot: Performance Score by Primary Role"
            )
            role_mapping = {role: idx for idx, role in enumerate(df["Primary Role"].unique())}
            beehive_fig.update_layout(
                xaxis=dict(
                    tickvals=list(role_mapping.values()),
                    ticktext=list(role_mapping.keys()),
                    title="Primary Role"
                ),
                yaxis_title="Performance Score",
                plot_bgcolor="#0b132b",
                paper_bgcolor="#0b132b",
                font=dict(color="white"),
                showlegend=True
            )
            st.plotly_chart(beehive_fig, use_container_width=True)

            # 3D scatter: Performance vs Fame vs Payment Performance (if exists) or OSIS placeholder
            st.subheader("📦 3D Scatter: Performance vs Fame vs Payment Performance")
            z_col = "Payment_Performance" if "Payment_Performance" in df.columns else "Performance_score"
            scatter3d = go.Figure(data=[go.Scatter3d(
                x=df["Performance_score"],
                y=df["Fame_score"],
                z=df[z_col],
                mode='markers',
                marker=dict(size=6, color=df["Performance_score"], colorscale='Viridis', showscale=True),
                text=df["Player Name"]
            )])
            scatter3d.update_layout(scene=dict(
                xaxis_title='Performance',
                yaxis_title='Fame',
                zaxis_title=(z_col.replace("_"," ") if z_col else 'Metric')
            ), paper_bgcolor="#0b132b", font=dict(color="white"))
            st.plotly_chart(scatter3d, use_container_width=True)

        if st.button("Generate Final Unbiased XI"):
            st.subheader("🏆 Final Unbiased XI")
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
                    wk_batter = pd.DataFrame()

            # protect when wk_batter empty
            remaining_pool = unbiased_df.copy()
            if not wk_batter.empty:
                remaining_pool = unbiased_df[~unbiased_df["Player Name"].isin(wk_batter["Player Name"])]
            batters = remaining_pool[remaining_pool["Primary Role"] == "batter"].nlargest(4, "Performance_score")
            bowlers = remaining_pool[remaining_pool["Primary Role"] == "bowler"].nlargest(4, "Performance_score")
            allrounders = remaining_pool[remaining_pool["Primary Role"] == "all-rounder"].nlargest(2, "Performance_score")

            final_xi = pd.concat([wk_batter, batters, bowlers, allrounders]).drop_duplicates("Player Name").head(11)
            final_xi = calculate_leadership_score(final_xi)

            final_xi = final_xi.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).reset_index(drop=True)
            # add captain/vice
            final_xi["Captain"] = False
            final_xi["Vice_Captain"] = False
            if len(final_xi) > 0:
                final_xi.at[0, "Captain"] = True
            if len(final_xi) > 1:
                final_xi.at[1, "Vice_Captain"] = True

            # Validate with official squad if present
            if "Official_Squad" in df.columns:
                official_squad = df["Official_Squad"].dropna().unique().tolist()
                role_dict = dict(zip(df["Player Name"], df["Primary Role"]))
                validated_xi, replacements = validate_and_fix_squad(final_xi, official_squad, role_dict)
                final_xi = validated_xi
                if replacements:
                    st.subheader("🔄 Replacements Made")
                    for old, new, role in replacements:
                        st.warning(f"Replaced {old} ➝ {new} (Role: {role})")
                else:
                    st.success("All Final XI players are part of the official squad ✅")

            st.session_state.final_xi = final_xi
            if not final_xi.empty:
                st.dataframe(final_xi[[
                    "Player Name", "Primary Role", "Performance_score", "Fame_score", "Is_Biased", "Captain", "Vice_Captain"
                ]])
            else:
                st.warning("No Final XI could be generated with current filters/data.")

            csv = final_xi[[
                "Player Name", "Primary Role", "Performance_score", "Fame_score", "Captain", "Vice_Captain"
            ]].to_csv(index=False).encode("utf-8")

            st.download_button("⬇ Download Final XI CSV", csv, "final_xi.csv", "text/csv")

            if not final_xi.empty:
                captain = final_xi[final_xi["Captain"]].iloc[0]
                vice_captain = final_xi[final_xi["Vice_Captain"]].iloc[0]

                st.success(f"🏏 Recommended Captain: {captain['Player Name']} | Leadership Score: {captain['Leadership_Score']:.2f}")
                st.info(f"🥢 Vice-Captain: {vice_captain['Player Name']} | Leadership Score: {vice_captain['Leadership_Score']:.2f}")

                if "rohit sharma" in final_xi["Player Name"].str.lower().values and captain["Player Name"].lower() != "rohit sharma":
                    rohit_score = final_xi[final_xi["Player Name"].str.lower() == "rohit sharma"]["Leadership_Score"].values[0]
                    st.warning(f"⚠ Rohit Sharma is the current captain, but *{captain['Player Name']}* has a higher Leadership Score ({captain['Leadership_Score']:.2f}) vs Rohit's ({rohit_score:.2f}).")

            # ------------------ SUBSTITUTES ------------------
            st.subheader("🛠 Substitution Players (Bench)")
            remaining_candidates = unbiased_df[~unbiased_df["Player Name"].isin(final_xi["Player Name"])] if not final_xi.empty else unbiased_df
            substitutes = calculate_leadership_score(remaining_candidates).sort_values(
                by=["Leadership_Score", "Fame_score"], ascending=False
            ).head(4)

            if substitutes.empty:
                st.warning("⚠ No eligible substitutes available.")
            else:
                st.dataframe(substitutes[[
                    "Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score"
                ]])

                st.session_state.substitutes = substitutes

                sub_csv = substitutes[[
                    "Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score"
                ]].to_csv(index=False).encode("utf-8")

                st.download_button("⬇ Download Substitutes CSV", sub_csv, "substitutes.csv", "text/csv")

            # Manual leadership selection UI
            if "final_xi" in st.session_state:
                st.markdown("---")
                st.subheader("✍ Select Future Leadership Manually")

                # Combine Final XI + Substitutes if available
                combined_players = st.session_state.final_xi.copy()
                if "substitutes" in st.session_state and not st.session_state.substitutes.empty:
                    combined_players = pd.concat([combined_players, st.session_state.substitutes]).drop_duplicates("Player Name")

                with st.form("manual_leadership_form"):
                    manual_candidates = st.multiselect(
                        "Select at least 2 players from the Unbiased XI and Substitutes for custom captain & vice-captain evaluation:",
                        options=combined_players["Player Name"].tolist()
                    )

                    submitted = st.form_submit_button("🧠 Calculate Leadership")

                    if submitted:
                        if len(manual_candidates) >= 2:
                            manual_df = combined_players[combined_players["Player Name"].isin(manual_candidates)].copy()
                            # Recalculate scaled performance & fame for selected players
                            manual_df["Batting Avg (scaled)"] = safe_scale(manual_df.get("Batting Avg", pd.Series(np.zeros(len(manual_df)))))
                            manual_df["Batting SR (scaled)"] = safe_scale(manual_df.get("Batting SR", pd.Series(np.zeros(len(manual_df)))))
                            manual_df["Wickets (scaled)"] = safe_scale(manual_df.get("Wickets", pd.Series(np.zeros(len(manual_df)))))
                            manual_df["Bowling Econ (scaled)"] = 1 - safe_scale(manual_df.get("Bowling Economy", pd.Series(np.zeros(len(manual_df)))))
                            manual_df["Performance_score_raw"] = manual_df.apply(compute_performance, axis=1)
                            manual_df["Performance_score"] = safe_scale(manual_df["Performance_score_raw"])
                            manual_df["Google Trends (scaled)"] = safe_scale(manual_df.get("Google Trends Score", pd.Series(np.zeros(len(manual_df)))))
                            if "Social Media Reach" not in manual_df.columns:
                                if "Instagram Followers" in manual_df.columns and "Twitter Followers" in manual_df.columns:
                                    manual_df["Instagram Followers"] = manual_df["Instagram Followers"].apply(convert_social_media_value)
                                    manual_df["Twitter Followers"] = manual_df["Twitter Followers"].apply(convert_social_media_value)
                                    manual_df["Social Media Reach"] = manual_df["Instagram Followers"] + manual_df["Twitter Followers"]
                                else:
                                    manual_df["Social Media Reach"] = 0
                            manual_df["Social Media Reach (scaled)"] = safe_scale(manual_df["Social Media Reach"])
                            manual_df["Fame_score"] = (
                                manual_df["Google Trends (scaled)"] * 0.5 +
                                manual_df["Social Media Reach (scaled)"] * 0.5
                            )

                            # Leadership Score (now correct)
                            manual_df = calculate_leadership_score(manual_df)
                            manual_df = manual_df.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False)

                            # Assign C/VC
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
            xaxis_title=xlab if xlab else fig.layout.xaxis.title.text if 'xaxis' in fig.layout else None,
            yaxis_title=ylab if ylab else fig.layout.yaxis.title.text if 'yaxis' in fig.layout else None,
            title=title if title else fig.layout.title.text if fig.layout.title else None,
            legend_title="Legend"
        )
        return fig

    if osis_file:
        df = pd.read_csv(osis_file)

        # Accept common column aliases: unify names
        df = df.rename(columns=lambda c: c.strip())

        required_columns = [
            "Player", "Primary Role", "Opponent", "Format",
            "Runs", "Balls_Faced", "Wickets", "Overs_Bowled", "Runs_Conceded",
            "Catches", "Run_Outs", "Stumpings"
        ]

        # Try to map similar columns if exact names not present
        col_map = {}
        if "Player" not in df.columns:
            for c in df.columns:
                if c.lower() in ["player name", "name", "player_name"]:
                    col_map[c] = "Player"
        if "Balls_Faced" not in df.columns:
            for c in df.columns:
                if c.lower() in ["balls", "balls_faced", "bf"]:
                    col_map[c] = "Balls_Faced"
        if "Overs_Bowled" not in df.columns:
            for c in df.columns:
                if c.lower() in ["overs", "overs_bowled", "ov"]:
                    col_map[c] = "Overs_Bowled"
        if "Runs_Conceded" not in df.columns:
            for c in df.columns:
                if c.lower() in ["runs_conceded", "conceded", "runs_against"]:
                    col_map[c] = "Runs_Conceded"
        if "Run_Outs" not in df.columns:
            for c in df.columns:
                if c.lower() in ["run_outs", "runouts"]:
                    col_map[c] = "Run_Outs"
        if "Primary Role" not in df.columns:
            for c in df.columns:
                if c.lower() in ["role", "primary_role"]:
                    col_map[c] = "Primary Role"

        if col_map:
            df = df.rename(columns=col_map)

        # Verify required columns now present
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error("❌ Missing required columns:\n\n- " + "\n- ".join(missing_cols))
        else:
            st.success("✅ File loaded with all required columns (including Format).")
            # -------------------- Impact Calculation Functions --------------------
            def batting_impact(runs, balls):
                try:
                    runs = float(runs)
                    balls = float(balls)
                except:
                    return 0.0
                if balls <= 0:
                    return runs * 0.6
                strike_rate = (runs / balls) * 100.0
                return (runs * 0.6) + (strike_rate * 0.4)

            def bowling_impact(wickets, overs, runs_conceded):
                try:
                    wickets = float(wickets)
                    overs = float(overs)
                    runs_conceded = float(runs_conceded)
                except:
                    return 0.0
                if overs <= 0:
                    return wickets * 20.0
                economy = runs_conceded / overs if overs > 0 else 999.0
                # higher wickets and lower economy increases impact
                econ_component = max(0, (6.0 - economy))  # reward sub-6 economy
                return (wickets * 20.0 * 0.7) + (econ_component * 10.0 * 0.3)

            def fielding_impact(catches, run_outs, stumpings):
                try:
                    return (float(catches) * 10.0) + (float(run_outs) * 12.0) + (float(stumpings) * 15.0)
                except:
                    return 0.0

            def calculate_role_impact(row):
                role = str(row["Primary Role"]).strip().lower()
                runs = row.get("Runs", 0); balls = row.get("Balls_Faced", 0)
                wkts = row.get("Wickets", 0); overs = row.get("Overs_Bowled", 0); rc = row.get("Runs_Conceded", 0)
                catches = row.get("Catches", 0); ro = row.get("Run_Outs", 0); stp = row.get("Stumpings", 0)

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
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            # Per-row impact
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

            # Avoid division by zero: if Overall_Avg_Impact is zero, set OSIS to 0
            def _compute_osis(row):
                overall_val = row.get("Overall_Avg_Impact", 0.0)
                opp_val = row.get("Opponent_Avg_Impact", 0.0)
                if pd.isna(overall_val) or overall_val == 0:
                    return 0.0
                return (opp_val / overall_val) * 100.0  # percentage: >100 means better vs that opponent

            osis_all["OSIS"] = osis_all.apply(_compute_osis, axis=1)

            # -------------------- Remarks per Opponent --------------------
            def add_remarks_per_opponent(osis_df):
                out = []
                for (opp, fmt), sub in osis_df.groupby(["Opponent", "Format"]):
                    sub = sub.copy()
                    max_osis = sub["OSIS"].max() if len(sub) else 0
                    def remark(x):
                        if max_osis == 0:
                            return "🔸 No Data"
                        if x == max_osis:
                            return "🏆 Top Matchup"
                        elif x >= 0.8 * max_osis:
                            return "🔥 Strong"
                        elif x >= 0.6 * max_osis:
                            return "✅ Solid"
                        elif x >= 0.4 * max_osis:
                            return "⚠ Average"
                        else:
                            return "🔻 Weak"
                    sub["Remarks"] = sub["OSIS"].apply(remark)
                    out.append(sub)
                return pd.concat(out, ignore_index=True) if out else osis_df

            osis_all = add_remarks_per_opponent(osis_all)

            # -------------------- Format Filter --------------------
            format_list = sorted(osis_all["Format"].unique().tolist())
            chosen_format = st.selectbox("📌 Select Format", format_list)

            osis_all_fmt = osis_all[osis_all["Format"] == chosen_format].copy()

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
                st.markdown("OSIS Heatmap (Players × Opponents)")
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
                st.markdown("Best vs Worst Opponent per Player")
                st.dataframe(bw.reset_index(drop=True))

                # Top 6 per Opponent (Team-wide)
                st.markdown("Top 6 Matchups per Opponent")
                for opp in opponent_list:
                    sub = osis_all_fmt[osis_all_fmt["Opponent"] == opp].sort_values("OSIS", ascending=False).head(6)
                    st.markdown(f"🏟 {opp} — Top 6")
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
                if not avg_vs_opp.empty:
                    top_opp = avg_vs_opp.iloc[0]
                    low_opp = avg_vs_opp.iloc[-1]
                    st.info(
                        f"✅ Best team-wide matchup in {chosen_format}: {top_opp['Opponent']} (avg OSIS {top_opp['OSIS']:.2f})\n\n"
                        f"⚠ Most challenging opponent in {chosen_format}: {low_opp['Opponent']} (avg OSIS {low_opp['OSIS']:.2f})."
                    )
                else:
                    st.info("No OSIS data to compute team-wide insights.")

            # -------------------- SINGLE OPPONENT (DEEP-DIVE) --------------------
            else:
                selected_opponent = view_choice
                st.subheader(f"🧭 Deep-Dive: OSIS vs {selected_opponent} — {chosen_format}")

                osis_df = (
                    osis_all_fmt[osis_all_fmt["Opponent"] == selected_opponent]
                    .copy()
                    .sort_values("OSIS", ascending=False)
                    .reset_index(drop=True)
                )

                # Recompute remarks (safe)
                max_osis = osis_df["OSIS"].max() if len(osis_df) else 0
                def get_remark(osis_score, max_):
                    if max_ == 0:
                        return "🔸 No Data"
                    if osis_score == max_:
                        return "🏆 Top Matchup"
                    elif osis_score >= 0.8 * max_:
                        return "🔥 Strong"
                    elif osis_score >= 0.6 * max_:
                        return "✅ Solid"
                    elif osis_score >= 0.4 * max_:
                        return "⚠ Average"
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

                # 🕸 Radar / Spider web comparing top players across multiple opponents:
                st.subheader("🕸 Radar: Compare Top Players' OSIS Across Opponents")
                top_players_for_radar = st.multiselect("Select top players (max 6)", options=osis_df["Player"].tolist(), default=osis_df["Player"].tolist()[:3], max_selections=6)
                if top_players_for_radar:
                    # Build a small radar across many opponents for selected players
                    opponents_all = sorted(osis_all_fmt["Opponent"].unique().tolist())
                    radar_rows = []
                    for p in top_players_for_radar:
                        row_vals = []
                        for opp in opponents_all:
                            sub = osis_all_fmt[(osis_all_fmt["Player"]==p) & (osis_all_fmt["Opponent"]==opp)]
                            val = sub["OSIS"].values[0] if len(sub) else 0.0
                            row_vals.append(val)
                        radar_rows.append((p, row_vals))
                    radar_fig = go.Figure()
                    for name, vals in radar_rows:
                        radar_fig.add_trace(go.Scatterpolar(
                            r=list(vals) + [vals[0]],
                            theta=opponents_all + [opponents_all[0]],
                            fill='toself',
                            name=name
                        ))
                    radar_fig.update_layout(paper_bgcolor="#0b132b", font_color="white")
                    st.plotly_chart(radar_fig, use_container_width=True)

                # 🔬 3D scatter: Performance vs Fame vs OSIS (for selected opponent)
                st.subheader("📦 3D Scatter: Perf vs Fame vs OSIS (selected opponent)")
                # we need performance & fame per player; try to source from overall df if available via session state
                base_metrics = {}
                if "df" in locals():
                    # we might not have session-state df here; try to compute approximate performance & fame
                    # use Overall_Avg_Impact as a proxy for OSIS baseline
                    pass
                # build 3d points
                merged_for_3d = osis_df.copy()
                # if session data present with Performance_score & Fame_score, merge
                try:
                    session_df = st.session_state.get("df", None)
                    if session_df is not None:
                        metrics = session_df[["Player Name","Performance_score","Fame_score"]].rename(columns={"Player Name":"Player"})
                        merged_for_3d = merged_for_3d.merge(metrics, on="Player", how="left")
                except Exception:
                    pass
                # fallbacks
                if "Performance_score" not in merged_for_3d.columns:
                    merged_for_3d["Performance_score"] = safe_scale(merged_for_3d["Overall_Avg_Impact"])
                if "Fame_score" not in merged_for_3d.columns:
                    merged_for_3d["Fame_score"] = safe_scale(merged_for_3d["Overall_Avg_Impact"] * 0.5)

                scatter3d_osis = go.Figure(data=[go.Scatter3d(
                    x=merged_for_3d["Performance_score"],
                    y=merged_for_3d["Fame_score"],
                    z=merged_for_3d["OSIS"],
                    mode='markers',
                    marker=dict(size=6, color=merged_for_3d["OSIS"], colorscale='Viridis', showscale=True),
                    text=merged_for_3d["Player"]
                )])
                scatter3d_osis.update_layout(scene=dict(
                    xaxis_title='Performance (proxy)',
                    yaxis_title='Fame (proxy)',
                    zaxis_title='OSIS'
                ), paper_bgcolor="#0b132b", font=dict(color="white"))
                st.plotly_chart(scatter3d_osis, use_container_width=True)

                # 🔍 Conclusion
                st.subheader("🔍 Conclusion & Insights")
                if len(osis_df) > 0:
                    top_player = osis_df.iloc[0]
                    least_player = osis_df.iloc[-1]
                    st.success(
                        f"🏅 Top Matchup Performer: {top_player['Player']} ({top_player['Primary Role']}) "
                        f"with an OSIS of {top_player['OSIS']:.2f} against {selected_opponent} in {chosen_format}."
                    )
                    st.error(
                        f"📉 Weakest Matchup Performer: {least_player['Player']} ({least_player['Primary Role']}) "
                        f"with an OSIS of {least_player['OSIS']:.2f} in {chosen_format}."
                    )

                    st.markdown("### 📝 Remarks Summary")
                    summary_counts = osis_df["Remarks"].value_counts().to_dict()
                    for remark, count in summary_counts.items():
                        st.markdown(f"- {remark}: {count} player(s)")
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
        st.info("📁 Please upload a CSV file to proceed with OSIS calculation.")
