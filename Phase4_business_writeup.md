# RideQuest: Demand Forecasting — Business & Architecture Write-Up

## 1. Business & Revenue Impact

**Why forecasting 15–30 minutes ahead matters**
Right now, dispatch is reactive: a driver is found only *after* a rider requests one. Forecasting near-term demand lets the platform reposition idle drivers toward zones about to spike, *before* the requests land. Two effects follow directly from our own results:

- Our benchmark shows `lag_15min` (the most recent observed demand) is by far the strongest predictor of the next 15 minutes — meaning short-horizon demand is highly persistent and momentum-driven. This is good news operationally: a dispatch system doesn't need perfect long-range foresight to be useful, it needs to react fast to the current trend per zone, which is exactly what a 15-minute-ahead model supports.
- The clustering step surfaced 3 dominant high-volume zones (out of 6 total) that account for the overwhelming majority of demand. Concentrating repositioning effort on these zones during their known peak windows (mornings ~9–10am, evenings ~6–8pm, per the EDA heatmap) is a low-risk, high-leverage first rollout.

**Reduced ETA / wait times & cancellations:** Every extra minute of wait time is a rider more likely to cancel or open a competing app. Directionally repositioning supply into a zone 15 minutes before its predicted spike shortens the average driver-to-rider distance at the moment of request, which is the single biggest lever on ETA.

**Surge pricing & repositioning heatmaps:** the zone × time-window demand table (Phase 2 output) is exactly the data structure a live "heatmap" UI or a surge-multiplier engine needs — both are just different consumers of the same predicted-demand-per-zone-per-window number.

**Rough ROI framing (illustrative, needs real unit economics to finalize):**
- Fuel/idle-time savings: fewer wasted repositioning trips because drivers move toward *predicted* demand instead of driving blind or waiting.
- Driver utilization: less idle time per driver-hour → same driver pool serves more riders.
- Fleet efficiency: fewer "no driver found" events in known high-demand zones, directly protecting revenue that would otherwise be lost to the competing app.

---

## 2. Production Deployment & MLOps Architecture

```
                ┌──────────────────┐
Raw ride events │  Kafka / Kinesis  │  (ride requests, streamed in real time)
                └────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Stream processor     │  assigns zone_id (via H3 index lookup —
              │ (Flink / Spark       │  cheap, deterministic — see Phase 2 notes
              │  Streaming)          │  on H3 vs K-Means for why H3 wins here)
              └─────────┬────────────┘
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
   ┌──────────────────┐    ┌───────────────────────┐
   │  Feature Store     │    │  Batch feature builder │
   │  (Feast / Redis)   │◄───┤  (recomputes lags/     │
   │  online rolling     │    │   rolling stats hourly)│
   │  lags + zone state  │    └───────────────────────┘
   └─────────┬───────────┘
             │  low-latency read
             ▼
   ┌──────────────────────┐
   │ Model Serving          │   FastAPI microservice wrapping the trained
   │ (FastAPI + XGBoost/    │   model; called by the dispatch system for
   │  Ridge model artifact) │   "predicted demand next 15min" per zone
   └─────────┬──────────────┘
             │
             ▼
   ┌──────────────────────┐
   │ Driver Dispatch /      │  repositioning suggestions, surge multiplier
   │ Surge Pricing Engine   │  input, driver-facing heatmap
   └────────────────────────┘

   ┌──────────────────────────────────────────┐
   │ Monitoring: prediction drift (PSI on      │
   │ predicted vs actual demand distribution), │
   │ feature store read latency, model         │
   │ staleness (retrain trigger)               │
   └──────────────────────────────────────────┘
```

**Real-time vs batch split:**
- **Batch** (hourly/daily): recompute lag/rolling features over the full history, retrain or fine-tune the model, refresh zone boundaries if using K-Means (not needed if using H3).
- **Real-time**: on each incoming ride event, update the zone's rolling counters in the online feature store (Redis/Feast), and serve a fresh prediction on request from the dispatch system — this is the loop that actually needs sub-second latency.

---

## 3. Edge Cases & Anomaly Handling

- **Sudden demand shocks (heavy rain, sports events, concerts):** a pure time-series/lag model reacts *after* the spike starts, not before it. Mitigation: an "event calendar" feature (known concerts, matches) and a weather-API feed as additional model inputs, plus a manual "surge override" switch product/ops teams can trigger for known scheduled events the model hasn't seen a pattern for yet.
- **App downtime / data gaps:** if the ingestion pipeline goes down, the online feature store's rolling stats go stale. The serving layer should detect "feature staleness" (e.g., last-updated timestamp on a zone's counters) and fall back to the naive lag/historical-average baseline rather than serving predictions built on outdated state — this is exactly why the naive baseline was kept in the Phase 3 benchmark, not just as a benchmark floor but as a legitimate fallback path.
- **New zones / cold start:** a newly active zone (e.g., service just launched there) has no lag history. Fall back to the average demand of geographically similar/neighboring zones until enough history accumulates.

---

## 4. Honest Caveats From This Build

- The clustering step found that 2 of the 6 K-Means zones are small outlier pockets far outside the main metro area (visible in `phase2_zone_map.png`) rather than genuine operational zones — worth excluding or re-clustering on the primary metro area only before this goes further.
- Ridge Regression outperformed XGBoost/LightGBM on RMSE in this benchmark. That's a legitimate result, not a bug: with lag/rolling features this strong, the extra non-linear capacity of tree ensembles didn't add value on this particular target/horizon. It's still worth keeping XGBoost/LightGBM in the comparison for longer-horizon targets (e.g., `demand_next_1hour` or further out), where non-linear interactions between time-of-day, zone, and lag history are more likely to matter.
- Prophet/Auto-ARIMA per-zone benchmarking was scoped out of the automated run (see `phase3_model_benchmark.py` docstring) since it requires a per-zone fitting loop rather than a single panel model — flagged as a follow-up rather than skipped silently.