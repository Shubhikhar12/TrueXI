# --------------------------------------------------
# TrueXI App - Updated
# Author: Nihira Khare (updated by ChatGPT)
# Date: July 2025 (Updated Sept 2025)
# --------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from streamlit_lottie import st_lottie
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="TrueXI Selector", layout="wide")

# ------------------ SESSION INIT ------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "df" not in st.session_state:
    st.session_state.df = None
if "final_xi" not in st.session_state:
    st.session_state.final_xi = None
if "substitutes" not in st.session_state:
    st.session_state.substitutes = None

# ------------------ LOTTIE LOADER ------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

lottie_cricket = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_vu1huepg.json")

# ------------------ STYLING ------------------
st.markdown("""
    <style>
        .stApp { background-color: #0b132b; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3, h4 { color: #ffd700; text-align: center; }
        section[data-testid="stSidebar"] { background-color: #1c2541; color: white; border-right: 2px solid #2a9d8f; }
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
selected_feature = st.sidebar.radio("Select Feature", ["Main App Flow"])

# ------------------ HEADER ------------------
# If you don't have 'app logo.png' remove this line or provide the file in the working dir
try:
    st.image("app logo.png", width=150)
except Exception:
    pass

st.markdown("<h1>🏏 True XI</h1>", unsafe_allow_html=True)
st.markdown("<h4>Make Data-Driven Cricket Selections Without Bias</h4>", unsafe_allow_html=True)
if lottie_cricket:
    try:
        st_lottie(lottie_cricket, height=140, key="cricket_header")
    except Exception:
        # lottie might fail if dependency missing; that's OK
        pass

# ------------------ UTILITY FUNCTIONS ------------------

def safe_scale(column):
    """Scales a pandas Series/array to 0-1 using MinMax. Returns flattened 1D numpy array."""
    col = np.array(column, dtype=float)
    if np.nanstd(col) == 0:
        return np.full(len(col), 0.5)
    scaler = MinMaxScaler()
    return scaler.fit_transform(col.reshape(-1, 1)).reshape(-1)

def convert_social_media_value(val):
    """Converts strings like '1.2M', '350K', '10,000' to numeric"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, str):
        v = val.strip().upper()
        try:
            if v.endswith("M"):
                return float(v[:-1]) * 1_000_000
            if v.endswith("K"):
                return float(v[:-1]) * 1_000
            if v.endswith("B"):
                return float(v[:-1]) * 1_000_000_000
            return float(v.replace(",", ""))
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def compute_performance_row(row):
    """Calculate a raw performance score depending on format and role."""
    fmt = str(row.get("Format", "")).strip().lower()
    role = str(row.get("Primary Role", "")).strip().lower()

    # safe retrieval with defaults
    try:
        bat_avg = float(row.get("Batting Avg", 0) if not pd.isna(row.get("Batting Avg", np.nan)) else 0)
    except Exception:
        bat_avg = 0.0
    try:
        bat_sr = float(row.get("Batting SR", 0) if not pd.isna(row.get("Batting SR", np.nan)) else 0)
    except Exception:
        bat_sr = 0.0
    try:
        wickets = float(row.get("Wickets", 0) if not pd.isna(row.get("Wickets", np.nan)) else 0)
    except Exception:
        wickets = 0.0
    try:
        bowl_econ = float(row.get("Bowling Economy", 10) if not pd.isna(row.get("Bowling Economy", np.nan)) else 10)
    except Exception:
        bowl_econ = 10.0
    try:
        dismissals = float(row.get("Dismissals", 0) if not pd.isna(row.get("Dismissals", np.nan)) else 0)
    except Exception:
        dismissals = 0.0

    # role/format-aware formulas
    batter_score = bowler_score = keeper_score = 0.0
    if fmt in ["test", "first-class", "first class", "firstclass"]:
        batter_score = 0.7 * bat_avg + 0.3 * (bat_sr / 100)
        bowler_score = 0.7 * (1 / (bowl_econ + 1)) * 100 + 0.3 * wickets
    elif fmt in ["odi", "list a", "one day"]:
        batter_score = 0.5 * bat_avg + 0.5 * (bat_sr / 100)
        bowler_score = 0.5 * (1 / (bowl_econ + 1)) * 100 + 0.5 * wickets
    elif fmt in ["t20", "t20i", "franchise t20", "domestic t20", "ipl", "bbl", "cpl"]:
        batter_score = 0.3 * bat_avg + 0.7 * (bat_sr / 100)
        bowler_score = 0.8 * (1 / (bowl_econ + 1)) * 100 + 0.2 * wickets
    else:
        # generic fallback
        batter_score = 0.5 * bat_avg + 0.5 * (bat_sr / 100)
        bowler_score = 0.5 * (1 / (bowl_econ + 1)) * 100 + 0.5 * wickets

    if role in ["batter", "wk-batter", "wk batter", "top-order", "batting allrounder"]:
        return batter_score
    if role == "bowler":
        return bowler_score
    if role in ["all-rounder", "all rounder", "allrounder"]:
        return 0.5 * batter_score + 0.5 * bowler_score
    if role in ["wicketkeeper", "keeper"]:
        keeper_score = 0.6 * (0.5 * bat_avg + 0.5 * (bat_sr / 100)) + 0.4 * dismissals
        return keeper_score

    # default conservative mix
    return 0.5 * batter_score + 0.5 * bowler_score

def calculate_leadership_score(df):
    """Leadership score = weighted mix of performance and fame (both in 0-1)."""
    df = df.copy()
    # ensure columns exist
    if "Performance_score" not in df.columns:
        df["Performance_score"] = 0.0
    if "Fame_score" not in df.columns:
        df["Fame_score"] = 0.0
    df["Leadership_Score"] = 0.6 * df["Performance_score"] + 0.4 * df["Fame_score"]
    return df

def validate_and_fix_squad(final_xi, official_squad_flat, role_dict):
    """Ensure players in final_xi exist in official_squad. Replace where necessary by same-role players from official_squad."""
    validated = final_xi.copy().reset_index(drop=True)
    replacements = []
    official_set = set([p.strip() for p in official_squad_flat if isinstance(p, str) and p.strip() != ""])
    # if no official squad passed, return as-is
    if not official_set:
        return validated, replacements

    for i, row in validated.iterrows():
        name = row.get("Player Name", "")
        role_needed = row.get("Primary Role", "")
        if pd.isna(name) or name not in official_set:
            # find replacement in official_set with same role
            candidates = [p for p, r in role_dict.items() if r == role_needed and p in official_set]
            candidate = None
            for c in candidates:
                # pick first candidate not already in validated
                if c not in validated["Player Name"].values:
                    candidate = c
                    break
            # fallback: any official player not used yet
            if candidate is None:
                for c in official_set:
                    if c not in validated["Player Name"].values:
                        candidate = c
                        break
            if candidate is None:
                # last resort leave as-is
                continue
            replacements.append((name, candidate, role_needed))
            validated.at[i, "Player Name"] = candidate
            validated.at[i, "Primary Role"] = role_dict.get(candidate, role_needed)
    return validated, replacements

# ------------------ MAIN APP FLOW ------------------
if selected_feature == "Main App Flow":
    st.subheader("🏏 Main App Flow")

    # --- Upload step ---
    if st.session_state.step == 0:
        uploaded_file = st.file_uploader("📁 Upload Final XI CSV", type="csv")
        st.info("Required columns: Player Name, Primary Role, Format, Batting Avg, Batting SR, Wickets, Bowling Economy, Google Trends Score, Instagram Followers, Twitter Followers. Optional: Dismissals, Official_Squad (comma-separated).")

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df = None

            if df is not None:
                # drop rows that are entirely empty
                df.dropna(how="all", inplace=True)

                # Normalize column names (but keep original)
                # Ensure required columns exist
                df_columns_lower = [c.lower() for c in df.columns]

                # Standardize Primary Role and Format columns
                if "Primary Role" in df.columns:
                    df["Primary Role"] = df["Primary Role"].astype(str).str.strip().str.lower()
                else:
                    st.warning("Primary Role column not found. Please include it.")
                    df["Primary Role"] = df.get("Primary Role", "").astype(str).str.strip().str.lower()

                if "Format" in df.columns:
                    df["Format"] = df["Format"].astype(str).str.strip()
                else:
                    df["Format"] = df.get("Format", "").astype(str).str.strip()

                # Convert social followers if present
                if "Twitter Followers" in df.columns:
                    df["Twitter Followers"] = df["Twitter Followers"].apply(convert_social_media_value)
                else:
                    df["Twitter Followers"] = np.nan

                if "Instagram Followers" in df.columns:
                    df["Instagram Followers"] = df["Instagram Followers"].apply(convert_social_media_value)
                else:
                    df["Instagram Followers"] = np.nan

                # Aggregate social reach
                df["Social Media Reach"] = df["Instagram Followers"].fillna(0) + df["Twitter Followers"].fillna(0)

                # Google Trends default
                if "Google Trends Score" not in df.columns:
                    df["Google Trends Score"] = 0

                # Official_Squad parse (if provided)
                if "Official_Squad" in df.columns:
                    df["Official_Squad"] = df["Official_Squad"].apply(lambda x: [p.strip() for p in str(x).split(",")] if pd.notna(x) else [])

                # Check required columns presence (case-sensitive names expected)
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
                    st.success("✅ File uploaded successfully. Now run 'Detect Biased Players'.")

    # --- After upload: main interactions ---
    if st.session_state.step >= 1 and st.session_state.df is not None:
        df = st.session_state.df.copy()

        # Show data area
        if st.button("Show Player Data"):
            st.subheader("📋 Uploaded Player Data")
            try:
                format_options = sorted(df["Format"].fillna("Unknown").unique().tolist())
            except Exception:
                format_options = []
            format_filter = st.selectbox("Filter by Format", options= ["All"] + format_options)
            role_options = sorted(df["Primary Role"].fillna("").unique().tolist())
            role_filter = st.multiselect("Filter by Primary Role", options=role_options, default=role_options)
            if format_filter == "All":
                filtered_df = df[df["Primary Role"].isin(role_filter)]
            else:
                filtered_df = df[(df["Format"] == format_filter) & (df["Primary Role"].isin(role_filter))]
            st.dataframe(filtered_df)

        # Detect biased players
        if st.button("Detect Biased Players"):
            st.header("Detect Biased Players")
            format_filter = st.selectbox(
                "Select format to analyze",
                ["All", "Test", "ODI", "T20I", "Franchise T20", "First-Class", "List A", "Domestic T20"]
            )

            # normalize Format for matching
            df["Format_clean"] = df["Format"].astype(str).str.strip().str.lower()

            fmt_map = {
                "Test": ["test"],
                "ODI": ["odi"],
                "T20I": ["t20i", "t20"],
                "Franchise T20": ["ipl", "bbl", "cpl", "franchise t20"],
                "First-Class": ["first-class", "first class", "firstclass"],
                "List A": ["list a", "list-a", "la"],
                "Domestic T20": ["domestic t20", "syed mushtaq ali", "blast"]
            }

            # build subdf
            if format_filter == "All":
                subdf = df.copy()
            else:
                selected_keys = [k.lower() for k in fmt_map.get(format_filter, [format_filter.lower()])]
                # exact match OR substring match fallback
                mask_exact = df["Format_clean"].isin(selected_keys)
                mask_contains = df["Format_clean"].apply(lambda x: any(k in x for k in selected_keys))
                mask = mask_exact | mask_contains
                subdf = df[mask].copy()

            if subdf.empty:
                st.warning(f"No players found for {format_filter}. Check your Format column text.")
            else:
                # Compute performance raw and scaled (once)
                subdf["Performance_raw"] = subdf.apply(compute_performance_row, axis=1)
                subdf["Performance_score"] = safe_scale(subdf["Performance_raw"])

                # Fame components: convert and scale
                subdf["Instagram Followers"] = subdf["Instagram Followers"].apply(lambda x: convert_social_media_value(x) if pd.notna(x) else 0)
                subdf["Twitter Followers"] = subdf["Twitter Followers"].apply(lambda x: convert_social_media_value(x) if pd.notna(x) else 0)
                subdf["Social Media Reach"] = subdf["Instagram Followers"].fillna(0) + subdf["Twitter Followers"].fillna(0)

                # scale fame components (guard if constant)
                subdf["Insta_scaled"] = safe_scale(subdf["Instagram Followers"].fillna(0))
                subdf["Twitter_scaled"] = safe_scale(subdf["Twitter Followers"].fillna(0))
                subdf["Google_scaled"] = safe_scale(subdf["Google Trends Score"].fillna(0))

                # Fame score (weights)
                subdf["Fame_score"] = (0.4 * subdf["Insta_scaled"] + 0.4 * subdf["Twitter_scaled"] + 0.2 * subdf["Google_scaled"])

                # Bias = fame - performance
                subdf["bias_score"] = subdf["Fame_score"] - subdf["Performance_score"]

                # Flag biased players: tweak threshold if you want
                subdf["Is_Biased"] = (subdf["Fame_score"] > 0.7) & (subdf["Performance_score"] < 0.4)

                # Alert levels helper
                def alert_level(row):
                    f = row["Fame_score"]
                    p = row["Performance_score"]
                    if f > 0.7 and p < 0.4:
                        return "RED"
                    if (0.4 <= f <= 0.7) and (0.4 <= p <= 0.7):
                        return "YELLOW"
                    if f < 0.4 and p > 0.7:
                        return "GREEN"
                    # other combinations
                    return "YELLOW" if f >= p else "YELLOW"

                subdf["Alert"] = subdf.apply(alert_level, axis=1)

                # Merge computed metrics back into session df (left join on Player Name)
                merge_cols = ["Player Name", "Performance_score", "Fame_score", "bias_score", "Is_Biased", "Alert"]
                # if Player Name duplicate issues exist ensure unique keys - assume unique player names
                st.session_state.df = df.merge(subdf[merge_cols], on="Player Name", how="left")

                # Show results
                st.subheader("Players flagged as Biased (High Fame, Low Performance)")
                biased_display = subdf[subdf["Is_Biased"]][["Player Name", "Primary Role", "Performance_score", "Fame_score", "bias_score"]]
                if biased_display.empty:
                    st.info("No high-fame / low-performance players detected in this format.")
                else:
                    st.dataframe(biased_display)

                # Average performance score (safe because we created Performance_score)
                avg_score = subdf["Performance_score"].mean()
                st.write(f"Average Performance Score for {format_filter}: {avg_score:.2f}")

                # Scatter plot: Fame vs Performance
                color_map = {"RED": "red", "YELLOW": "yellow", "GREEN": "green"}
                fig = px.scatter(
                    subdf, x="Fame_score", y="Performance_score", color="Alert",
                    hover_data=["Player Name", "Primary Role", "Fame_score", "Performance_score"],
                    title=f"Fame vs Performance — {format_filter}"
                )
                fig.update_traces(marker=dict(size=12))
                fig.update_layout(
                    xaxis_title="Fame Score (0-1)",
                    yaxis_title="Performance Score (0-1)",
                    plot_bgcolor="#0b132b",
                    paper_bgcolor="#0b132b",
                    font=dict(color="white")
                )
                # map legend categories to colors
                for t in fig.data:
                    if t.name in color_map:
                        t.marker.color = color_map[t.name]

                st.plotly_chart(fig, use_container_width=True)

                # Beehive-style role view
                st.subheader("🐝 Performance by Primary Role (Beehive-style)")
                subdf = subdf.reset_index(drop=True)
                role_order = list(subdf["Primary Role"].unique())
                role_mapping = {r: i for i, r in enumerate(role_order)}
                subdf["Primary Role_num"] = subdf["Primary Role"].map(role_mapping)
                subdf["Jittered_x"] = subdf["Primary Role_num"] + np.random.normal(0, 0.15, size=len(subdf))

                beehive_fig = px.scatter(
                    subdf, x="Jittered_x", y="Performance_score",
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
                    plot_bgcolor="#0b132b",
                    paper_bgcolor="#0b132b",
                    font=dict(color="white")
                )
                st.plotly_chart(beehive_fig, use_container_width=True)

        # Generate Final XI
        if st.button("Generate Final Unbiased XI"):
            st.subheader("🏆 Final Unbiased XI")
            if st.session_state.df is None:
                st.error("No data available. Please upload a CSV first.")
            else:
                df_full = st.session_state.df.copy()

                # Ensure Is_Biased exists (prompt user if they didn't run detection)
                if "Is_Biased" not in df_full.columns:
                    st.warning("Please run 'Detect Biased Players' first to compute scores.")
                else:
                    unbiased_df = df_full[(df_full["Is_Biased"] == False) | (df_full["Is_Biased"].isna())].copy()

                    # If any of the performance/fame columns missing in df_full, compute them
                    if "Performance_score" not in df_full.columns or df_full["Performance_score"].isna().all():
                        df_full["Performance_raw"] = df_full.apply(compute_performance_row, axis=1)
                        df_full["Performance_score"] = safe_scale(df_full["Performance_raw"])

                    if "Fame_score" not in df_full.columns or df_full["Fame_score"].isna().all():
                        df_full["Instagram Followers"] = df_full["Instagram Followers"].apply(lambda x: convert_social_media_value(x) if pd.notna(x) else 0)
                        df_full["Twitter Followers"] = df_full["Twitter Followers"].apply(lambda x: convert_social_media_value(x) if pd.notna(x) else 0)
                        df_full["Insta_scaled"] = safe_scale(df_full["Instagram Followers"].fillna(0))
                        df_full["Twitter_scaled"] = safe_scale(df_full["Twitter Followers"].fillna(0))
                        df_full["Google_scaled"] = safe_scale(df_full["Google Trends Score"].fillna(0))
                        df_full["Fame_score"] = 0.4 * df_full["Insta_scaled"] + 0.4 * df_full["Twitter_scaled"] + 0.2 * df_full["Google_scaled"]

                    # Work on unbiased pool
                    unbiased_df = df_full[(df_full["Is_Biased"] == False) | (df_full["Is_Biased"].isna())].copy()

                    # Select wicket-keeper (prefer unbiased)
                    wk_variants = ["wk-batter", "wicketkeeper", "keeper", "wk batter", "wk"]
                    wk_batter = None
                    wk_unbiased = unbiased_df[unbiased_df["Primary Role"].isin(wk_variants)].copy()
                    if not wk_unbiased.empty:
                        wk_unbiased = calculate_leadership_score(wk_unbiased)
                        wk_batter = wk_unbiased.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).head(1)
                        st.info(f"✅ WK-Batter selected from unbiased list: {wk_batter.iloc[0]['Player Name']}")
                    else:
                        # fallback - pick best available from full dataset
                        wk_all = df_full[df_full["Primary Role"].isin(wk_variants)].copy()
                        if not wk_all.empty:
                            wk_all = calculate_leadership_score(wk_all)
                            wk_batter = wk_all.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).head(1)
                            st.warning(f"⚠ No unbiased WK-Batter found. Selected best available: {wk_batter.iloc[0]['Player Name']}")
                        else:
                            st.error("❌ No wicket-keeper found in dataset. Cannot build Final XI.")
                            wk_batter = pd.DataFrame()

                    # Build remaining pool excluding selected WK
                    selected_wk_names = wk_batter["Player Name"].tolist() if (wk_batter is not None and not wk_batter.empty) else []
                    remaining_pool = unbiased_df[~unbiased_df["Player Name"].isin(selected_wk_names)].copy()

                    # Choose top players per role (using Performance_score)
                    batters = remaining_pool[remaining_pool["Primary Role"] == "batter"].nlargest(4, "Performance_score") if not remaining_pool[remaining_pool["Primary Role"] == "batter"].empty else remaining_pool[remaining_pool["Primary Role"] == "batter"]
                    bowlers = remaining_pool[remaining_pool["Primary Role"] == "bowler"].nlargest(4, "Performance_score") if not remaining_pool[remaining_pool["Primary Role"] == "bowler"].empty else remaining_pool[remaining_pool["Primary Role"] == "bowler"]
                    allrounders = remaining_pool[remaining_pool["Primary Role"].isin(["all-rounder", "all rounder", "allrounder"])].nlargest(2, "Performance_score") if not remaining_pool[remaining_pool["Primary Role"].isin(["all-rounder", "all rounder", "allrounder"])].empty else remaining_pool[remaining_pool["Primary Role"].isin(["all-rounder", "all rounder", "allrounder"])]

                    parts = []
                    if wk_batter is not None and not wk_batter.empty:
                        parts.append(wk_batter)
                    if not batters.empty:
                        parts.append(batters)
                    if not bowlers.empty:
                        parts.append(bowlers)
                    if not allrounders.empty:
                        parts.append(allrounders)

                    if parts:
                        final_xi = pd.concat(parts).drop_duplicates("Player Name").head(11).reset_index(drop=True)
                    else:
                        final_xi = pd.DataFrame()

                    if final_xi.empty or len(final_xi) < 1:
                        st.error("Could not assemble Final XI. Check roles & availability in the uploaded dataset.")
                    else:
                        # Ensure leadership metrics present
                        if "Performance_score" not in final_xi.columns:
                            final_xi["Performance_score"] = final_xi.apply(compute_performance_row, axis=1)
                            final_xi["Performance_score"] = safe_scale(final_xi["Performance_score"])
                        if "Fame_score" not in final_xi.columns:
                            final_xi["Insta_scaled"] = safe_scale(final_xi["Instagram Followers"].fillna(0))
                            final_xi["Twitter_scaled"] = safe_scale(final_xi["Twitter Followers"].fillna(0))
                            final_xi["Google_scaled"] = safe_scale(final_xi["Google Trends Score"].fillna(0))
                            final_xi["Fame_score"] = 0.4 * final_xi["Insta_scaled"] + 0.4 * final_xi["Twitter_scaled"] + 0.2 * final_xi["Google_scaled"]

                        final_xi = calculate_leadership_score(final_xi)
                        final_xi = final_xi.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).reset_index(drop=True)

                        # Assign Captain & Vice-Captain
                        final_xi["Captain"] = False
                        final_xi["Vice_Captain"] = False
                        final_xi.at[0, "Captain"] = True
                        if len(final_xi) > 1:
                            final_xi.at[1, "Vice_Captain"] = True

                        # Validate with Official Squad if available
                        if "Official_Squad" in df_full.columns:
                            official_entries = df_full["Official_Squad"].dropna().tolist()
                            flat_official = [item for sublist in official_entries for item in sublist] if official_entries else []
                            role_dict = dict(zip(df_full["Player Name"], df_full["Primary Role"]))
                            validated_xi, replacements = validate_and_fix_squad(final_xi, flat_official, role_dict)
                            final_xi = validated_xi
                            if replacements:
                                st.subheader("🔄 Replacements Made")
                                for old, new, role in replacements:
                                    st.warning(f"Replaced {old} ➝ {new} (Role: {role})")
                            else:
                                st.success("All Final XI players are part of the official squad ✅")

                        # Save to session and show
                        st.session_state.final_xi = final_xi
                        display_cols = ["Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score", "Captain", "Vice_Captain"]
                        st.dataframe(final_xi[display_cols])

                        # Download CSV
                        csv = final_xi[display_cols].to_csv(index=False).encode("utf-8")
                        st.download_button("⬇ Download Final XI CSV", csv, "final_xi.csv", "text/csv")

                        captain = final_xi[final_xi["Captain"]].iloc[0]
                        vice_captain = final_xi[final_xi["Vice_Captain"]].iloc[0] if len(final_xi[final_xi["Vice_Captain"]]) > 0 else None
                        st.success(f"🏏 Recommended Captain: {captain['Player Name']} | Leadership Score: {captain['Leadership_Score']:.2f}")
                        if vice_captain is not None:
                            st.info(f"🥢 Vice-Captain: {vice_captain['Player Name']} | Leadership Score: {vice_captain['Leadership_Score']:.2f}")

                        # Substitutes bench
                        remaining_candidates = unbiased_df[~unbiased_df["Player Name"].isin(final_xi["Player Name"])].copy()
                        substitutes = calculate_leadership_score(remaining_candidates).sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).head(4)
                        if substitutes.empty:
                            st.warning("⚠ No eligible substitutes available.")
                        else:
                            st.subheader("🛠 Substitution Players (Bench)")
                            st.dataframe(substitutes[["Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score"]])
                            st.session_state.substitutes = substitutes
                            sub_csv = substitutes[["Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score"]].to_csv(index=False).encode("utf-8")
                            st.download_button("⬇ Download Substitutes CSV", sub_csv, "substitutes.csv", "text/csv")

                        # Manual leadership selection form
                        st.markdown("---")
                        st.subheader("✍ Select Future Leadership Manually")
                        combined_players = final_xi.copy()
                        if st.session_state.substitutes is not None and not st.session_state.substitutes.empty:
                            combined_players = pd.concat([combined_players, st.session_state.substitutes], ignore_index=True).drop_duplicates("Player Name")

                        with st.form("manual_leadership_form"):
                            manual_candidates = st.multiselect(
                                "Select at least 2 players from the Unbiased XI and Substitutes for custom captain & vice-captain evaluation:",
                                options=combined_players["Player Name"].tolist()
                            )
                            submitted = st.form_submit_button("🧠 Calculate Leadership")
                            if submitted:
                                if len(manual_candidates) < 2:
                                    st.warning("Please select at least 2 players.")
                                else:
                                    manual_df = combined_players[combined_players["Player Name"].isin(manual_candidates)].copy()
                                    manual_df["Performance_score_raw"] = manual_df.apply(compute_performance_row, axis=1)
                                    manual_df["Performance_score"] = safe_scale(manual_df["Performance_score_raw"])
                                    manual_df["Insta_scaled"] = safe_scale(manual_df["Instagram Followers"].fillna(0))
                                    manual_df["Twitter_scaled"] = safe_scale(manual_df["Twitter Followers"].fillna(0))
                                    manual_df["Google_scaled"] = safe_scale(manual_df["Google Trends Score"].fillna(0))
                                    manual_df["Fame_score"] = (0.4 * manual_df["Insta_scaled"] + 0.4 * manual_df["Twitter_scaled"] + 0.2 * manual_df["Google_scaled"])
                                    manual_df = calculate_leadership_score(manual_df)
                                    manual_df = manual_df.sort_values(by=["Leadership_Score", "Fame_score"], ascending=False).reset_index(drop=True)
                                    manual_df["Captain"] = False
                                    manual_df["Vice_Captain"] = False
                                    manual_df.at[0, "Captain"] = True
                                    if len(manual_df) > 1:
                                        manual_df.at[1, "Vice_Captain"] = True

                                    st.success(f"🥢 Manually Selected Captain: {manual_df.iloc[0]['Player Name']} | Leadership Score: {manual_df.iloc[0]['Leadership_Score']:.2f}")
                                    if len(manual_df) > 1:
                                        st.info(f"🎖 Manually Selected Vice-Captain: {manual_df.iloc[1]['Player Name']} | Leadership Score: {manual_df.iloc[1]['Leadership_Score']:.2f}")

                                    st.dataframe(manual_df[["Player Name", "Primary Role", "Performance_score", "Fame_score", "Leadership_Score", "Captain", "Vice_Captain"]])

                                    manual_csv = manual_df[["Player Name", "Primary Role", "Performance_score", "Fame_score", "Captain", "Vice_Captain"]].to_csv(index=False).encode("utf-8")
                                    st.download_button("⬇ Download Manual Captain-Vice CSV", manual_csv, "manual_captain_vice.csv", "text/csv")

    # footer
    st.markdown("---")
    st.markdown("<p style='text-align: right; font-size: 20px; font-weight: bold; color: white;'>~Made By Nihira Khare</p>", unsafe_allow_html=True)
    st.markdown("""
        <hr style="margin-top: 50px;"/>
        <div style='text-align: center; color: gray; font-size: 14px;'>
            © 2025 <b>TrueXI</b>. All rights reserved.
        </div>
        """, unsafe_allow_html=True)
