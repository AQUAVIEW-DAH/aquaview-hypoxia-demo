#!/usr/bin/env python3
"""
TimesFM fallback prediction for stations without XGBoost models.

Reads the existing latest.json, identifies stations that either:
  1. Have no model at all but have realtime DO data
  2. Have an XGBoost model but poor performance (AUC-ROC < 0.5)
  3. Have a model but are historical-only (could still forecast from recent data)

For these stations, it runs TimesFM zero-shot forecasting on the recent DO
time series and adds predictions to the JSON with model_type="timesfm".

Stations with working XGBoost models keep model_type="xgboost".
"""

import json
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

HYPOXIA_THRESHOLD = 2.0
LEAD_DAYS = [1, 3, 5, 7]


def fetch_recent_do(station_id, days_back=180):
    """Fetch recent DO data from NDBC OPeNDAP for a station."""
    current_year = datetime.now().year
    years_to_try = list(range(current_year, current_year - 3, -1))

    frames = []
    for year in years_to_try:
        url = f"https://dods.ndbc.noaa.gov/thredds/dodsC/data/ocean/{station_id.lower()}/{station_id.lower()}o{year}.nc"
        try:
            ds = xr.open_dataset(url, decode_times=True)
            time_vals = pd.to_datetime(ds["time"].values)
            df = pd.DataFrame({"time": time_vals})

            if "dissolved_oxygen" in ds:
                vals = ds["dissolved_oxygen"].values.flatten()[:len(df)].astype(float)
                vals[(vals > 900) | (vals < -10) | (vals > 25)] = np.nan
                df["o2"] = vals

            ds.close()
            if "o2" in df.columns and df["o2"].notna().sum() > 0:
                frames.append(df)
        except:
            continue

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    if "o2" not in combined.columns or combined["o2"].notna().sum() < 30:
        return None

    combined["date"] = combined["time"].dt.date
    daily = combined.groupby("date")["o2"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Keep last N days
    cutoff = daily.index.max() - timedelta(days=days_back)
    daily = daily[daily.index >= cutoff]

    return daily


def timesfm_predict(tfm, do_series, context_len=512):
    """
    Run TimesFM forecast on recent DO data.
    Returns predicted DO values and hypoxia probabilities for 1,3,5,7 days.
    """
    do_clean = do_series.interpolate(method='linear', limit=3).dropna()
    if len(do_clean) < 30:
        return None

    # Use the most recent context_len days
    context = do_clean.values[-min(context_len, len(do_clean)):].tolist()

    try:
        point_forecast, quantile_forecast = tfm.forecast(
            [context],
            freq=[0],  # daily
        )
    except Exception as e:
        print(f"    Forecast error: {e}")
        return None

    results = {}
    for h in LEAD_DAYS:
        if h > point_forecast.shape[1]:
            continue

        pred_do = float(point_forecast[0, h - 1])

        # Estimate P(hypoxia) from quantiles
        p_hyp = 0.0
        if quantile_forecast is not None:
            try:
                q_vals = quantile_forecast[0, h - 1, :]
                q_levels = np.linspace(0.1, 0.9, len(q_vals))

                below = q_vals < HYPOXIA_THRESHOLD
                if below.all():
                    p_hyp = 0.95
                elif not below.any():
                    p_hyp = 0.05
                else:
                    last_below = np.where(below)[0][-1]
                    if last_below == len(q_vals) - 1:
                        p_hyp = float(q_levels[-1])
                    else:
                        frac = (HYPOXIA_THRESHOLD - q_vals[last_below]) / \
                               (q_vals[last_below + 1] - q_vals[last_below] + 1e-8)
                        p_hyp = float(q_levels[last_below] + frac * (q_levels[last_below + 1] - q_levels[last_below]))
                        p_hyp = max(0.0, min(1.0, p_hyp))
            except:
                p_hyp = float(pred_do < HYPOXIA_THRESHOLD)

        risk = "low" if p_hyp < 0.3 else ("moderate" if p_hyp < 0.6 else "high")

        results[f"{h}d"] = {
            "prob": round(p_hyp, 4),
            "risk": risk,
            "predicted_do": round(pred_do, 2),
        }

    return results


def enhance_latest_json(input_path, output_path, tfm):
    """
    Enhance latest.json with TimesFM predictions for stations lacking XGBoost models.
    """
    with open(input_path) as f:
        data = json.load(f)

    # Track stats
    xgb_stations = 0
    tfm_stations = 0
    tfm_failed = 0

    for station in data["stations"]:
        sid = station["id"]

        # Stations with working XGBoost predictions keep them
        if station.get("predictions") and station.get("has_model"):
            # Check if XGBoost metrics are decent (AUC-ROC > 0.5 on any lead time)
            metrics = station.get("metrics", {})
            has_decent_xgb = any(
                m.get("auc_roc", 0) > 0.5
                for m in metrics.values()
                if isinstance(m, dict)
            )

            if has_decent_xgb or station["predictions"]:
                station["model_type"] = "xgboost"
                xgb_stations += 1
                continue

        # Try TimesFM for stations without models or with poor XGBoost
        print(f"  [{sid}] Attempting TimesFM forecast...")

        do_series = fetch_recent_do(sid)
        if do_series is None or len(do_series) < 30:
            print(f"    Insufficient DO data for TimesFM")
            tfm_failed += 1
            station["model_type"] = None
            continue

        predictions = timesfm_predict(tfm, do_series)
        if predictions is None:
            print(f"    TimesFM forecast failed")
            tfm_failed += 1
            station["model_type"] = None
            continue

        # Add TimesFM predictions
        station["predictions"] = predictions
        station["model_type"] = "timesfm"
        station["has_model"] = True

        # Add DO forecast values (unique to TimesFM)
        station["do_forecast"] = {
            k: v["predicted_do"] for k, v in predictions.items()
        }

        tfm_stations += 1

        # Determine if currently hypoxic based on recent DO
        last_do = do_series.dropna().iloc[-1] if len(do_series.dropna()) > 0 else None
        if last_do is not None:
            station["currently_hypoxic"] = bool(last_do < HYPOXIA_THRESHOLD)
            if station.get("current") is None:
                station["current"] = {}
            # Don't override existing realtime DO, but fill if missing
            if station["current"].get("do") is None:
                station["current"]["do"] = round(float(last_do), 2)

        print(f"    TimesFM predictions: " +
              ", ".join(f"{k}={v['prob']:.1%}" for k, v in predictions.items()))

    # Update summary
    data["summary"]["stations_with_models"] = xgb_stations + tfm_stations
    data["summary"]["stations_with_predictions"] = sum(
        1 for s in data["stations"] if s.get("predictions")
    )
    data["summary"]["stations_with_warnings"] = sum(
        1 for s in data["stations"]
        if s.get("predictions") and any(
            p.get("prob", 0) >= 0.6 for p in s["predictions"].values()
        )
    )
    data["summary"]["timesfm_stations"] = tfm_stations
    data["summary"]["xgboost_stations"] = xgb_stations

    # Add metadata about the enhancement
    data["model_info"] = {
        "xgboost": {
            "description": "Station-specific XGBoost classifier with engineered features",
            "features": "DO lags, rolling stats, satellite SST/chlorophyll, cross-station deltas",
            "training": "Per-station with 70/30 temporal split",
        },
        "timesfm": {
            "description": "Google TimesFM 1.0 (200M params) zero-shot forecast",
            "approach": "Direct DO time-series forecasting with quantile uncertainty",
            "note": "Used for stations with insufficient training events for XGBoost",
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  Summary: {xgb_stations} XGBoost + {tfm_stations} TimesFM stations "
          f"({tfm_failed} failed)")

    return data


if __name__ == "__main__":
    import timesfm
    from pathlib import Path

    ROOT = Path(__file__).parent.parent
    latest_path = ROOT / "docs" / "data" / "latest.json"

    if not latest_path.exists():
        print(f"No latest.json found at {latest_path}")
        print("Run update_predictions.py first.")
        exit(1)

    print("=" * 60)
    print("TIMESFM ENHANCEMENT")
    print("=" * 60)
    print("\nLoading TimesFM (google/timesfm-1.0-200m-pytorch)...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    print("Model loaded.\n")

    enhance_latest_json(
        input_path=str(latest_path),
        output_path=str(latest_path),  # overwrite in-place
        tfm=tfm,
    )
    print("\nTimesFM enhancement complete.")
