"""
RIDEQUEST - Streamlit Dashboard
================================================================
Run:  streamlit run streamlit_app.py

Loads the CSV outputs already produced by phase1/phase2/phase3
scripts (fast — no re-running the full pipeline on every refresh)
and renders the EDA, clustering, and model benchmark results as an
interactive multi-tab app. Phase 4's write-up is rendered as markdown.

Expected files in the same folder (from your existing pipeline run):
    phase1_cleaned_features.csv[.gz]
    phase2_zone_timeseries.csv
    phase3_model_comparison_summary.csv
    phase3_model_comparison_by_fold.csv
    phase4_business_writeup.md
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="RideQuest — Demand Forecast", layout="wide")

# ------------------------------------------------------------------
# CONFIG — file paths (edit if yours live elsewhere)
# ------------------------------------------------------------------
PHASE1_PATH = "phase1_cleaned_features.csv"   # pandas reads .gz directly; swap to .csv if unzipped
PHASE2_PATH = "phase2_zone_timeseries.csv"
PHASE3_SUMMARY_PATH = "phase3_model_comparison_summary.csv"
PHASE3_BYFOLD_PATH = "phase3_model_comparison_by_fold.csv"
PHASE4_PATH = "phase4_business_writeup.md"


# ------------------------------------------------------------------
# CACHED DATA LOADERS
# st.cache_data means these only re-run when the file path/content changes,
# not on every widget interaction — critical for the multi-million-row files.
# ------------------------------------------------------------------
@st.cache_data
def count_rows(path):
    """Fast line count so we can take a random sample without loading
    the whole file into memory first."""
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return sum(1 for _ in f) - 1  # minus header


@st.cache_data
def load_phase1(path, max_rows=300_000):
    """
    Loads Phase 1's cleaned dataset. This file can be 8M+ rows / ~800MB
    once decompressed, which is too heavy to load in full for an
    interactive dashboard (can OOM on a typical machine). Instead, we take
    a random sample of `max_rows` rows via `skiprows`, which keeps memory
    bounded to the sample size rather than the full file size.
    Increase max_rows if your machine has plenty of RAM and you want the
    full dataset.
    """
    total_rows = count_rows(path)
    if total_rows <= max_rows:
        return pd.read_csv(path, parse_dates=["ts"], dtype={"number": str})

    rng = np.random.default_rng(42)
    n_to_skip = total_rows - max_rows
    skip_idx = set(rng.choice(np.arange(1, total_rows + 1), size=n_to_skip, replace=False))
    df = pd.read_csv(path, parse_dates=["ts"], dtype={"number": str},
                      skiprows=lambda i: i in skip_idx)
    return df


@st.cache_data
def load_phase2(path):
    df = pd.read_csv(path, parse_dates=["time_window"])
    return df


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("RideQuest")
st.sidebar.caption("Ola Bike Ride Demand Forecast — interactive dashboard")
page = st.sidebar.radio(
    "Go to phase",
    ["Phase 1: Cleaning & EDA", "Phase 2: Clustering & Features",
     "Phase 3: Model Benchmark", "Phase 4: Business Case"],
)

eda_sample_size = st.sidebar.slider(
    "EDA plot sample size (rows)", 10_000, 200_000, 50_000, step=10_000,
    help="Larger samples = more accurate plots but slower rendering."
)
phase1_max_rows = st.sidebar.number_input(
    "Phase 1 data load cap (rows)", min_value=50_000, max_value=2_000_000,
    value=300_000, step=50_000,
    help="Phase 1's file has 8M+ rows. A random sample of this size is loaded "
         "instead of the full file to keep memory usage manageable."
)

# ------------------------------------------------------------------
# PHASE 1 — CLEANING & EDA
# ------------------------------------------------------------------
if page == "Phase 1: Cleaning & EDA":
    st.title("Phase 1 — Data Cleaning, Analysis & Visualization")

    try:
        df = load_phase1(PHASE1_PATH, max_rows=phase1_max_rows)
    except FileNotFoundError:
        st.error(f"Couldn't find `{PHASE1_PATH}`. Run `phase1_cleaning_eda.py` first, "
                 f"or update PHASE1_PATH at the top of this script.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows loaded (sampled)", f"{len(df):,}")
    col2.metric("Date range start", str(df["ts"].min().date()))
    col3.metric("Date range end", str(df["ts"].max().date()))
    st.caption("Note: this is a random sample of the full cleaned dataset "
               "(adjust the cap in the sidebar), not all 8.2M+ rows — kept "
               "this way so the dashboard stays responsive.")

    st.subheader("Ride Demand Heatmap: Hour of Day vs Day of Week")
    pivot = df.groupby(["day_of_week", "hour_of_day"]).size().unstack(fill_value=0)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="YlOrRd", yticklabels=day_labels, ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    st.pyplot(fig)

    st.subheader("Hourly Ride Request Volume Over Time")
    hourly = df.set_index("ts").resample("1h").size().rename("request_count").reset_index()
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(hourly["ts"], hourly["request_count"], linewidth=0.7)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Requests per hour")
    st.pyplot(fig2)

    st.subheader("Pickup Locations: Peak vs Off-Peak Hours")
    sample = df.sample(min(eda_sample_size, len(df)), random_state=42)
    is_peak = sample["hour_of_day"].isin([8, 9, 18, 19, 20]).map({True: "Peak", False: "Off-peak"})
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x=sample["pick_lng"], y=sample["pick_lat"], hue=is_peak,
                     s=4, alpha=0.4, palette={"Peak": "red", "Off-peak": "steelblue"}, ax=ax3)
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    st.pyplot(fig3)

    with st.expander("View raw cleaned data sample"):
        st.dataframe(df.sample(min(1000, len(df))))


# ------------------------------------------------------------------
# PHASE 2 — CLUSTERING & FEATURES
# ------------------------------------------------------------------
elif page == "Phase 2: Clustering & Features":
    st.title("Phase 2 — Data Clustering & Feature Engineering")

    try:
        zone_ts = load_phase2(PHASE2_PATH)
    except FileNotFoundError:
        st.error(f"Couldn't find `{PHASE2_PATH}`. Run `phase2_clustering_features.py` first.")
        st.stop()

    n_zones = zone_ts["zone_id"].nunique()
    st.metric("Number of zones (K)", n_zones)

    st.subheader("Total demand by zone")
    zone_totals = zone_ts.groupby("zone_id")["demand_count"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    zone_totals.plot(kind="bar", ax=ax, color="teal")
    ax.set_xlabel("Zone ID")
    ax.set_ylabel("Total demand (all time)")
    st.pyplot(fig)
    st.caption("Zones with very low totals relative to the rest are likely outlier "
               "pockets outside the main metro area — worth filtering before production use.")

    st.subheader("Demand curve for a selected zone")
    selected_zone = st.selectbox("Zone", sorted(zone_ts["zone_id"].unique()))
    zone_slice = zone_ts[zone_ts["zone_id"] == selected_zone].sort_values("time_window")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(zone_slice["time_window"], zone_slice["demand_count"], linewidth=0.6)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Demand count (per 15-min window)")
    st.pyplot(fig2)

    with st.expander("View zone x time feature table sample"):
        st.dataframe(zone_ts.sample(min(1000, len(zone_ts))))


# ------------------------------------------------------------------
# PHASE 3 — MODEL BENCHMARK
# ------------------------------------------------------------------
elif page == "Phase 3: Model Benchmark":
    st.title("Phase 3 — Evaluating Various Model Approaches")

    try:
        summary = load_csv(PHASE3_SUMMARY_PATH)
        by_fold = load_csv(PHASE3_BYFOLD_PATH)
    except FileNotFoundError:
        st.error(f"Couldn't find `{PHASE3_SUMMARY_PATH}` / `{PHASE3_BYFOLD_PATH}`. "
                 f"Run `phase3_model_benchmark.py` first.")
        st.stop()

    st.subheader("Model comparison (mean across CV folds)")
    st.dataframe(summary.style.highlight_min(subset=["rmse_mean", "mae_mean"], color="lightgreen"))

    best_model = summary.loc[summary["rmse_mean"].idxmin(), "model"]
    st.success(f"Best model by RMSE: **{best_model}**")

    st.subheader("RMSE by fold (variance check)")
    fig, ax = plt.subplots(figsize=(9, 4))
    for model in by_fold["model"].unique():
        subset = by_fold[by_fold["model"] == model]
        ax.plot(subset["fold"], subset["rmse"], marker="o", label=model)
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("RMSE")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Metric definitions")
    st.markdown("""
    - **RMSE / MAE / MAPE / R²** — standard regression metrics.
    - **Underprediction penalty** — a weighted MAE that penalizes
      *under*-forecasting demand more heavily than over-forecasting, since
      missed demand loses revenue while idle driver time is comparatively cheap.
    """)


# ------------------------------------------------------------------
# PHASE 4 — BUSINESS CASE
# ------------------------------------------------------------------
elif page == "Phase 4: Business Case":
    st.title("Phase 4 — Presentation & Business Case")
    try:
        text = load_text(PHASE4_PATH)
        st.markdown(text)
    except FileNotFoundError:
        st.error(f"Couldn't find `{PHASE4_PATH}` in this folder.")
