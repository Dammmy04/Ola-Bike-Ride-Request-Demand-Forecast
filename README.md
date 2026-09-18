# RideQuest — Ola Bike Ride Demand Forecast

End-to-end demand forecasting pipeline: raw ride logs → cleaned/featured data →
zone clustering + time-series features → model benchmark → business write-up.

## Folder layout (how to arrange this in VS Code)

Create one project folder, e.g. `ridequest/`, and lay it out like this:

```
ridequest/
├── requirements.txt
├── data/
│   └── raw_data.csv          <- unzip raw_data.csv.gz here (or point RAW_PATH at the .gz directly)
├── phase1_cleaning_eda.py
├── phase2_clustering_features.py
├── phase3_model_benchmark.py
├── phase4_business_writeup.md
└── outputs/                  <- scripts write their CSVs/PNGs here (created automatically)
```

Each script writes its outputs to the current working directory, so either:
- run each script from inside `ridequest/` directly (simplest), or
- edit the `INPUT_PATH` / `OUTPUT_*` constants near the top of each file to point
  at whatever folder layout you prefer.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run order

Run these **in order** — each phase's output feeds the next phase's input.

```bash
# 1. Clean raw data, extract time features, generate EDA plots
python phase1_cleaning_eda.py
# -> produces: phase1_cleaned_features.csv, phase1_heatmap_hour_vs_day.png,
#              phase1_hourly_trend.png, phase1_spatial_peak_scatter.png

# 2. K-Means zone clustering + zone x time lag/rolling features
python phase2_clustering_features.py
# -> produces: phase2_zone_timeseries.csv, phase2_elbow_silhouette.png,
#              phase2_zone_map.png

# 3. Model benchmark (Naive baseline, Ridge, XGBoost, LightGBM) with
#    TimeSeriesSplit CV and the custom under-prediction penalty metric
python phase3_model_benchmark.py
# -> produces: phase3_model_comparison_summary.csv,
#              phase3_model_comparison_by_fold.csv,
#              phase3_feature_importance.png, phase3_predictions_vs_actual.png
```

Phase 4 (`phase4_business_writeup.md`) is a written document, not a script —
open it directly, no execution needed.

## Streamlit dashboard (optional, all-in-one view)

`streamlit_app.py` wraps Phases 1–4 into one interactive app instead of
running each script separately. It reads the CSV/PNG outputs the scripts
already produced (fast) rather than recomputing the pipeline live.

```bash
pip install -r requirements.txt   # now includes streamlit
streamlit run streamlit_app.py
```

It opens in your browser at `http://localhost:8501`. Use the sidebar to
switch between phases. Two things to know:
- **Phase 1's file is large** (8.2M+ rows once decompressed). The app loads
  a random sample (default 300,000 rows, adjustable in the sidebar) instead
  of the full file — this avoids the app running out of memory. Bump the
  cap up if your machine has plenty of RAM and you want the full dataset.
- All four CSV/markdown files (`phase1_cleaned_features.csv.gz`,
  `phase2_zone_timeseries.csv`, `phase3_model_comparison_summary.csv`,
  `phase3_model_comparison_by_fold.csv`, `phase4_business_writeup.md`) need
  to sit in the same folder as `streamlit_app.py`, or you can edit the
  path constants near the top of the file.

## What's already been run for you

All three scripts were already executed once against the real ~8.4M-row raw
dataset (not a toy sample) so you have working reference output to compare
against once you re-run them yourself:

| File | Rows/content |
|---|---|
| `phase1_cleaned_features.csv.gz` | 8.23M cleaned rows with time features (gzip'd, ~165MB — pandas reads `.csv.gz` directly, no need to unzip) |
| `phase2_zone_timeseries.csv` | 210,786 rows: 6 zones × 35,131 fifteen-minute windows, zero-filled, with lag/rolling features |
| `phase3_model_comparison_summary.csv` | Mean metrics per model across 5 CV folds |
| `phase3_model_comparison_by_fold.csv` | Metrics broken out per fold (for sanity-checking variance) |

## Real results from this run (for your reference/presentation)

- **Cleaning:** 8,381,556 raw rows → 8,227,844 after removing duplicates,
  out-of-bounds coordinates, and near-zero-distance trips.
- **Clustering:** K=6 selected by silhouette score. **Caveat:** 2 of those 6
  zones are small outlier pockets far outside the main metro cluster (visible
  in `phase2_zone_map.png`) — worth excluding or re-clustering on the primary
  metro area only before using this in a real pipeline.
- **Best model by RMSE:** Ridge Regression narrowly beat LightGBM and XGBoost
  on the `demand_next_15min` target. This is a legitimate result: with
  `lag_15min` as a dominant, near-linear predictor, the extra non-linear
  capacity of tree ensembles didn't pay off at this short horizon. Worth
  re-testing on `demand_next_1hour` (already computed as a target column in
  Phase 2's output, just swap the `TARGET` constant in Phase 3) where
  non-linear interactions are more likely to matter.
- **Feature importance:** `lag_15min` dominates, followed by `rolling_mean_1h`
  and `lag_30min` — recent momentum matters far more than day-of-week/hour
  features for a 15-minute-ahead target.

## Notes on scope / honest deviations from the original prompt

- The raw dataset has no `request_id`, `driver_id`, or `status` field, so
  "true demand" can't be split from "fulfilled demand" — every row is
  treated as a demand signal. See the docstring in `phase1_cleaning_eda.py`.
- Prophet/Auto-ARIMA weren't included in the automated Phase 3 benchmark
  since they fit one model per zone rather than one panel model for all
  zones — a stub function (`fit_prophet_per_zone`) is included at the bottom
  of `phase3_model_benchmark.py` if you want to extend it yourself.
- SHAP is listed in requirements.txt but the script currently uses XGBoost's
  built-in `feature_importances_` for speed; swapping in a proper SHAP
  summary plot is a straightforward next step if your presentation needs it.# Ola-Bike-Ride-Request-Demand-Forecast
