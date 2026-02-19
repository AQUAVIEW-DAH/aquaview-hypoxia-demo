#!/usr/bin/env python3
"""
Multi-Source Hypoxia Onset Forecast Pipeline
============================================
AQUAVIEW → NDBC + WOD + CoastWatch → XGBoost Multi-Horizon Onset Prediction

Predicts the ONSET of coastal hypoxia events (DO ≤ 2.0 mg/L) at multiple
lead times (1, 3, 5, 7 days) using data from AQUAVIEW-discovered sources:

  Source 1: NDBC       — fixed water quality stations (THREDDS OPeNDAP)
  Source 2: WOD        — Argo float oxygen profiles (NOAA S3)
  Source 3: CoastWatch — satellite SST, chlorophyll, SSH, SSS, EKE (ERDDAP)

Both station discovery and data access are routed through AQUAVIEW's STAC API.
The extraction script (multistation_extract.py) queries AQUAVIEW to find
stations, retrieves OPeNDAP URLs from AQUAVIEW item assets, and downloads
data from those URLs. This pipeline can also independently discover and
download station data via AQUAVIEW.

Usage:
  pip install -r requirements.txt
  python multistation_extract.py        # Step 1: extract data via AQUAVIEW
  python hypoxia_forecast_pipeline.py   # Step 2: train models

Data discovery + access: AQUAVIEW STAC API (aquaview.ai)
API docs: https://aquaview-sfeos-1025757962819.us-east1.run.app/api.html
"""
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
import requests
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ─── AQUAVIEW STAC API ──────────────────────────────────────────────────────
AQUAVIEW_API = "https://aquaview-sfeos-1025757962819.us-east1.run.app"
BBOX = {"west": -91.0, "south": 28.0, "east": -85.0, "north": 31.0}

# ─── Configuration ───────────────────────────────────────────────────────────
# Common variable mapping for all NDBC ocean stations
NDBC_VARS = {'wt': 'water_temperature', 'sal': 'salinity',
             'o2': 'dissolved_oxygen', 'os': 'o2_saturation',
             'dep': 'depth'}

# WOD features file (produced by multistation_extract.py)
# CoastWatch satellite file (produced by multistation_extract.py)
SAT_FEATURES_FILE = Path("station_data") / "satellite_merged.csv"
WOD_FEATURES_FILE = Path("station_data") / "wod_features.csv"

# Station registry file (produced by multistation_extract.py via AQUAVIEW)
STATION_REGISTRY_FILE = Path("station_data") / "station_registry.json"

HYPOXIA_THRESHOLD = 2.0  # mg/L
LEAD_TIMES = [1, 3, 5, 7]  # days ahead


# ─── AQUAVIEW API Client ────────────────────────────────────────────────────

def aquaview_search(collection=None, q=None, bbox=None, limit=100):
    """Search AQUAVIEW STAC catalog (GET /search)."""
    params = {"limit": limit}
    if q:
        params["q"] = q
    if bbox:
        params["bbox"] = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
    if collection:
        params["collections"] = collection
    url = f"{AQUAVIEW_API}/search"
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("numberMatched", 0), data.get("features", [])
    except Exception as e:
        print(f"  AQUAVIEW search failed: {e}")
        return 0, []


def aquaview_get_item(collection_id, item_id):
    """Get item details + asset URLs (GET /collections/{id}/items/{id})."""
    url = f"{AQUAVIEW_API}/collections/{collection_id}/items/{item_id}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  AQUAVIEW get_item failed for {item_id}: {e}")
        return None


def discover_stations_via_aquaview():
    """
    Discover NDBC stations with dissolved oxygen via AQUAVIEW STAC API.
    Returns a dict of station configs with OPeNDAP URLs from AQUAVIEW assets.
    """
    print("\n  [AQUAVIEW] Discovering NDBC stations...")
    print(f"  API: GET {AQUAVIEW_API}/search?collections=NDBC&bbox={BBOX}")

    n_total, features = aquaview_search(collection="NDBC", bbox=BBOX, limit=100)
    print(f"  Found {n_total} NDBC stations in study area")

    # Filter for dissolved_oxygen and get full item details
    stations = {}
    for f in features:
        props = f.get("properties", {})
        if "dissolved_oxygen" not in props.get("aquaview:variables", []):
            continue

        item_id = f["id"]
        # Get full item with assets
        item = aquaview_get_item("NDBC", item_id)
        if item is None:
            continue

        sid = item_id.replace("ndbc_", "")
        coords = item.get("geometry", {}).get("coordinates", [0, 0])
        assets = item.get("assets", {})

        # Extract OPeNDAP URLs from AQUAVIEW assets
        opendap_urls = []
        for ak, av in assets.items():
            if ak.startswith("od_ocean_"):
                href = av["href"]
                opendap_urls.append(href.replace("/thredds/fileServer/", "/thredds/dodsC/"))

        if not opendap_urls:
            continue

        lat = coords[1] if isinstance(coords[0], (int, float)) else coords[1][0]
        lon = coords[0] if isinstance(coords[0], (int, float)) else coords[0][0]

        stations[sid] = {
            'name': props.get("title", sid),
            'lat': lat, 'lon': lon,
            'opendap_urls': sorted(opendap_urls),
            'aquaview_item_id': item_id,
            'vars': NDBC_VARS,
        }
        print(f"    {sid.upper()}: {len(opendap_urls)} data files")

    return stations


def load_station_registry():
    """
    Load station registry from extraction script output, or discover via AQUAVIEW.
    The registry contains AQUAVIEW item IDs and OPeNDAP URLs for each station.
    """
    if STATION_REGISTRY_FILE.exists():
        print(f"\n  Loading AQUAVIEW station registry from {STATION_REGISTRY_FILE}")
        with open(STATION_REGISTRY_FILE) as f:
            registry = json.load(f)
        # Convert to pipeline format
        stations = {}
        for sid, info in registry.items():
            stations[sid] = {
                'name': info['name'],
                'lat': info['lat'], 'lon': info['lon'],
                'opendap_urls': info['opendap_urls'],
                'aquaview_item_id': info['aquaview_item_id'],
                'vars': NDBC_VARS,
            }
        print(f"  Loaded {len(stations)} stations (discovered via AQUAVIEW)")
        return stations
    else:
        print(f"\n  Station registry not found ({STATION_REGISTRY_FILE})")
        print("  Discovering stations directly via AQUAVIEW API...")
        return discover_stations_via_aquaview()


# ─── Step 1: Download Station Data ──────────────────────────────────────────

def download_station_data(station_id, config):
    """Download and aggregate daily data from AQUAVIEW-discovered THREDDS endpoint."""
    print(f"\n  Downloading {station_id.upper()} ({config['name']})...")
    print(f"    AQUAVIEW item: {config.get('aquaview_item_id', 'N/A')}")
    print(f"    Data files: {len(config['opendap_urls'])} (from AQUAVIEW assets)")

    all_daily = []
    for url in config['opendap_urls']:
        fname = url.split("/")[-1]
        try:
            ds = xr.open_dataset(url)
            df_year = pd.DataFrame({'time': ds['time'].values})

            for local_name, remote_name in config['vars'].items():
                if remote_name in ds:
                    vals = ds[remote_name].values.flatten()
                    vals = vals[:len(df_year)].astype(float)
                    # Replace fill values / out-of-range with NaN
                    vals[(vals > 900) | (vals < -10)] = np.nan
                    # Apply realistic range limits per variable
                    if remote_name == "dissolved_oxygen":
                        vals[vals > 25] = np.nan   # DO cannot exceed ~20 mg/L
                    elif remote_name == "o2_saturation":
                        vals[vals > 200] = np.nan
                    elif remote_name == "water_temperature":
                        vals[vals > 50] = np.nan
                    elif remote_name == "salinity":
                        vals[vals > 45] = np.nan
                    df_year[local_name] = vals
                else:
                    df_year[local_name] = np.nan

            ds.close()

            # Skip files where DO data is entirely missing
            if 'o2' not in df_year.columns or df_year['o2'].dropna().empty:
                print(f"    {fname}: SKIP (no DO data)")
                continue

            df_year['time'] = pd.to_datetime(df_year['time'])
            df_year['date'] = df_year['time'].dt.date

            # Daily aggregation
            agg_dict = {'time': [('n', 'count')]}
            if 'wt' in df_year.columns:
                agg_dict['wt'] = [('wm', 'mean'), ('wn', 'min'), ('wx', 'max')]
            if 'sal' in df_year.columns:
                agg_dict['sal'] = [('sal', 'mean')]
            if 'o2' in df_year.columns:
                agg_dict['o2'] = [('o2', 'mean'), ('o2n', 'min'),
                                  ('o2x', 'max'), ('o2sd', 'std')]
            if 'os' in df_year.columns:
                agg_dict['os'] = [('os', 'mean')]
            if 'dep' in df_year.columns:
                agg_dict['dep'] = [('dep', 'mean')]

            grouped = df_year.groupby('date')
            daily_parts = []
            for col, ops in agg_dict.items():
                for out_name, func in ops:
                    daily_parts.append(grouped[col].agg(func).rename(out_name))
            daily = pd.concat(daily_parts, axis=1).reset_index()

            daily['dt'] = pd.to_datetime(daily['date'])
            daily['hyp'] = (daily['o2'] <= HYPOXIA_THRESHOLD).astype(int)
            all_daily.append(daily)
            print(f"    {fname}: {len(daily)} days")
        except Exception as e:
            print(f"    {fname}: SKIP ({e})")

    if not all_daily:
        raise RuntimeError(f"No data downloaded for {station_id}. Check network connection.")

    result = pd.concat(all_daily).sort_values('dt').reset_index(drop=True)
    result = result.drop(columns=['date'], errors='ignore')

    # Quality check: reject stations with insufficient valid DO data
    if 'o2' in result.columns:
        valid_do = result['o2'].dropna()
        if len(valid_do) < 30:
            print(f"  *** REJECTED {station_id.upper()}: only {len(valid_do)} valid DO days ***")
            return None
        if valid_do.std() < 0.01:
            print(f"  *** REJECTED {station_id.upper()}: DO has no variance (fill values) ***")
            return None

    print(f"  Total: {len(result)} days, {result['hyp'].sum()} hypoxic")
    return result


# ─── Step 2: Feature Engineering ─────────────────────────────────────────────
def engineer_features(df, cross_station_dfs=None, wod_features_df=None, sat_df=None):
    """Engineer features including multi-station network + WOD offshore + satellite context."""
    # Temporal
    doy = df['dt'].dt.dayofyear
    month = df['dt'].dt.month
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['is_summer'] = month.isin([5, 6, 7, 8, 9]).astype(int)

    base_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in ['hyp', 'onset', 'n']]

    # Lags, rolling stats, rate of change
    for col in base_cols:
        for lag in [1, 3, 5, 7]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_roll7_mean'] = df[col].rolling(7, min_periods=3).mean()
        df[f'{col}_roll7_std'] = df[col].rolling(7, min_periods=3).std()
        df[f'{col}_roll14_mean'] = df[col].rolling(14, min_periods=5).mean()
        df[f'{col}_diff1'] = df[col].diff(1)
        df[f'{col}_diff3'] = df[col].diff(3)
        df[f'{col}_diff7'] = df[col].diff(7)

    # Stratification proxy
    if 'sal' in df.columns and 'wm' in df.columns:
        df['temp_sal_product'] = df['wm'] * df['sal']
        df['temp_sal_ratio'] = df['wm'] / df['sal'].replace(0, np.nan)

    # DO-specific features
    if 'o2' in df.columns:
        df['o2_anomaly'] = df['o2'] - df['o2'].rolling(7, min_periods=3).mean()

        def rolling_slope(s, w):
            return s.rolling(w).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == w else np.nan,
                raw=False)

        df['o2_trend3'] = rolling_slope(df['o2'], 3)
        df['o2_trend7'] = rolling_slope(df['o2'], 7)

        streak = 0
        streaks = []
        for val in df['o2']:
            streak = streak + 1 if val < 4.0 else 0
            streaks.append(streak)
        df['low_do_streak'] = streaks

    # Days since last hypoxia
    days_since = []
    last_hyp = -999
    for i in range(len(df)):
        if df.iloc[i]['hyp'] == 1:
            last_hyp = i
        days_since.append(i - last_hyp if last_hyp >= 0 else 999)
    df['days_since_hyp'] = days_since

    # Cross-station features (multi-station network)
    if cross_station_dfs is not None:
        for sid, csdf in cross_station_dfs.items():
            if csdf is None or len(csdf) == 0:
                continue
            prefix = sid
            rename_map = {}
            for col in ['wm', 'sal', 'o2', 'os', 'dep']:
                if col in csdf.columns:
                    rename_map[col] = f'{prefix}_{col}'
            csdf_renamed = csdf.rename(columns=rename_map)
            merge_cols = ['dt'] + [c for c in csdf_renamed.columns if c.startswith(prefix)]
            df = df.merge(csdf_renamed[merge_cols], on='dt', how='left')

            # Delta features (DPHA1 minus this station)
            for col in ['wm', 'sal', 'o2']:
                src = f'{prefix}_{col}'
                if col in df.columns and src in df.columns:
                    df[f'delta_{col}_{prefix}'] = df[col] - df[src]

            # Lags and rolling means for cross-station DO
            for col in [c for c in df.columns if c.startswith(f'{prefix}_o2')]:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_roll7'] = df[col].rolling(7, min_periods=3).mean()

    # WOD cross-source features (offshore Argo float context)
    if wod_features_df is not None and len(wod_features_df) > 0:
        wod_merge = wod_features_df.copy()
        if not isinstance(wod_merge.index, pd.DatetimeIndex):
            wod_merge.index = pd.to_datetime(wod_merge.index)
        wod_merge = wod_merge.reset_index().rename(columns={"index": "dt", "dt": "dt"})
        if "dt" in wod_merge.columns:
            wod_merge["dt"] = pd.to_datetime(wod_merge["dt"])
            df = df.merge(wod_merge, on="dt", how="left")

    # CoastWatch satellite features (SST, chlorophyll, SSH, SSS, EKE)
    if sat_df is not None and len(sat_df) > 0:
        sat_merge = sat_df.copy()
        sat_merge["dt"] = pd.to_datetime(sat_merge["dt"])
        df = df.merge(sat_merge, on="dt", how="left")

        # Satellite-specific lag and trend features
        sat_cols = [c for c in sat_merge.columns if c.startswith("sat_")]
        for col in sat_cols:
            if col in df.columns:
                for lag in [1, 3, 7]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
                df[f'{col}_roll7'] = df[col].rolling(7, min_periods=3).mean()
                df[f'{col}_diff3'] = df[col].diff(3)
                df[f'{col}_diff7'] = df[col].diff(7)

        # Satellite SST anomaly vs in-situ water temp
        if 'sat_sst' in df.columns and 'wm' in df.columns:
            df['sst_insitu_delta'] = df['sat_sst'] - df['wm']

    # Fill remaining NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().fillna(0)

    return df


# ─── Step 3: Train Onset Models ──────────────────────────────────────────────
def train_onset_models(df, events, cross_station_ids, split_date=None):
    """Train multi-horizon onset prediction models with 4 feature sets."""
    # Create onset targets
    for lead in LEAD_TIMES:
        target = pd.Series(0, index=df.index)
        for e in events:
            onset_idx = e['start']
            for d in range(1, lead + 1):
                if onset_idx - d >= 0:
                    target.iloc[onset_idx - d] = 1
        df[f'onset_{lead}d'] = target

    df_valid = df[df['hyp'] == 0].copy()

    exclude = {'dt', 'hyp', 'onset'} | {f'onset_{l}d' for l in LEAD_TIMES}
    all_features = [c for c in df_valid.columns if c not in exclude
                    and df_valid[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Feature sets
    cross_source = all_features  # everything: NDBC network + WOD + satellite
    ndbc_plus_sat = [c for c in all_features if not c.startswith('wod_')]
    ndbc_only = [c for c in ndbc_plus_sat
                 if not c.startswith('sat_') and 'sst_insitu' not in c]
    local_only = [c for c in ndbc_only
                  if not any(c.startswith(f'{sid}_') or c.startswith(f'delta_')
                             for sid in cross_station_ids)]
    no_do = [c for c in all_features
             if not any(x in c for x in ['o2', 'os', 'low_do', 'days_since'])]

    if split_date is None:
        split_date = pd.Timestamp('2018-01-01')
    train = df_valid[df_valid['dt'] < split_date].dropna(subset=all_features)
    test = df_valid[df_valid['dt'] >= split_date].dropna(subset=all_features)

    print(f"  Feature sets: cross_source={len(cross_source)}, "
          f"ndbc_plus_sat={len(ndbc_plus_sat)}, ndbc_only={len(ndbc_only)}, "
          f"local_only={len(local_only)}, no_do={len(no_do)}")
    print(f"  Train: {len(train)} days, Test: {len(test)} days")
    print(f"  Train events: {train[[f'onset_{l}d' for l in LEAD_TIMES]].max().max()}"
          f" | Test positive samples (7d): {test['onset_7d'].sum()}")

    results = {}
    for lead in LEAD_TIMES:
        for name, feats in [('cross_source', cross_source),
                            ('ndbc_plus_sat', ndbc_plus_sat),
                            ('ndbc_only', ndbc_only),
                            ('local_only', local_only),
                            ('no_do', no_do)]:
            target_col = f'onset_{lead}d'
            X_tr, y_tr = train[feats], train[target_col]
            X_te, y_te = test[feats], test[target_col]

            pw = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1)
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                scale_pos_weight=pw, min_child_weight=5, subsample=0.8,
                colsample_bytree=0.7, gamma=2, reg_alpha=0.5, reg_lambda=2.0,
                eval_metric='aucpr', random_state=42, use_label_encoder=False)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

            y_prob = model.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob) if y_te.sum() > 0 else 0
            pr = average_precision_score(y_te, y_prob) if y_te.sum() > 0 else 0

            thresholds = np.arange(0.01, 0.95, 0.005)
            f1s = [f1_score(y_te, (y_prob >= t).astype(int), zero_division=0)
                   for t in thresholds]
            best_f1 = max(f1s)

            results[f'{lead}d_{name}'] = {
                'auc_roc': auc, 'auc_pr': pr, 'f1': best_f1}
            print(f"  {lead}d {name:>15}: AUC-ROC={auc:.3f} PR={pr:.3f} F1={best_f1:.3f}")

    return results


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MULTI-SOURCE HYPOXIA ONSET FORECAST PIPELINE")
    print("=" * 70)
    print(f"Data discovery + access: AQUAVIEW STAC API ({AQUAVIEW_API})")
    print(f"API docs: {AQUAVIEW_API}/api.html")
    print(f"Sources: NDBC (via THREDDS) + WOD (via S3) — all from AQUAVIEW")

    # Load or discover station registry via AQUAVIEW
    stations = load_station_registry()
    if not stations:
        print("\n*** No stations available. Run multistation_extract.py or check AQUAVIEW API. ***")
        return

    # Download all stations
    print(f"\n[1/6] Downloading station data (URLs from AQUAVIEW assets)...")
    station_dfs = {}
    for sid in stations:
        try:
            result = download_station_data(sid, stations[sid])
            if result is not None:
                station_dfs[sid] = result
            else:
                print(f"  {sid.upper()}: excluded (failed quality check)")
        except Exception as e:
            print(f"  {sid.upper()} download failed ({e}), skipping")

    # Auto-select primary station: the one with the most hypoxic days
    hyp_counts = {sid: int(sdf['hyp'].sum()) for sid, sdf in station_dfs.items()
                  if 'hyp' in sdf.columns}
    if not hyp_counts or max(hyp_counts.values()) == 0:
        print("\n*** No hypoxia detected at any station. Cannot train onset models. ***")
        return
    primary = max(hyp_counts, key=hyp_counts.get)
    primary_df = station_dfs[primary]
    cross_station_ids = [sid for sid in station_dfs if sid != primary]
    print(f"\n  Primary station (most hypoxia): {primary.upper()} "
          f"({hyp_counts[primary]} hypoxic days)")
    print(f"  Active stations: {len(station_dfs)} ({len(cross_station_ids)} cross-stations)")

    # Multi-station summary
    print("\n[2/6] Station summary...")
    for sid, sdf in station_dfs.items():
        do_min = sdf['o2'].min() if 'o2' in sdf.columns else float('nan')
        do_max = sdf['o2'].max() if 'o2' in sdf.columns else float('nan')
        hyp_days = int(sdf['hyp'].sum()) if 'hyp' in sdf.columns else 0
        date_range = f"{sdf['dt'].min().date()} to {sdf['dt'].max().date()}"
        print(f"  {sid.upper():>6s}: {len(sdf):>5d} days, "
              f"DO {do_min:.2f}–{do_max:.2f} mg/L, "
              f"{hyp_days} hyp days ({date_range})")

    # Load WOD features (if extraction script has been run)
    wod_features = None
    if WOD_FEATURES_FILE.exists():
        print(f"\n  Loading WOD cross-source features from {WOD_FEATURES_FILE}")
        wod_features = pd.read_csv(WOD_FEATURES_FILE, index_col=0, parse_dates=True)
        wod_coverage = wod_features['wod_n_profiles'].gt(0).mean() * 100
        print(f"  WOD coverage: {wod_coverage:.1f}% of days have offshore context")
    else:
        print(f"\n  WOD features not found ({WOD_FEATURES_FILE})")
        print("  Run multistation_extract.py first to generate cross-source features.")
        print("  Continuing with NDBC-only features...")

    # Load CoastWatch satellite features (if extraction script has been run)
    sat_features = None
    if SAT_FEATURES_FILE.exists():
        print(f"\n  Loading CoastWatch satellite features from {SAT_FEATURES_FILE}")
        sat_features = pd.read_csv(SAT_FEATURES_FILE)
        sat_features["dt"] = pd.to_datetime(sat_features["dt"])
        sat_cols = [c for c in sat_features.columns if c.startswith("sat_")]
        sat_valid = sat_features[sat_cols].notna().any(axis=1).sum()
        print(f"  Satellite variables: {', '.join(sat_cols)}")
        print(f"  Satellite coverage: {sat_valid} days with data")
    else:
        print(f"\n  Satellite features not found ({SAT_FEATURES_FILE})")
        print("  Run multistation_extract.py first to generate satellite features.")
        print("  Continuing without satellite context...")

    # Identify events at primary station
    print(f"\n[3/6] Identifying hypoxia events at {primary.upper()}...")
    primary_df['onset'] = ((primary_df['hyp'] == 1) &
                           (primary_df['hyp'].shift(1, fill_value=0) == 0)).astype(int)
    events = []
    start_idx = None
    for i in range(len(primary_df)):
        if primary_df.loc[i, 'onset'] == 1:
            start_idx = i
        if start_idx is not None and primary_df.loc[i, 'hyp'] == 1:
            if i == len(primary_df) - 1 or primary_df.loc[i + 1, 'hyp'] == 0:
                events.append({'start': start_idx, 'end': i,
                              'duration': i - start_idx + 1})
                start_idx = None
    print(f"  Found {len(events)} hypoxia events")
    for j, ev in enumerate(events[:20]):  # show first 20
        onset_date = primary_df.loc[ev['start'], 'dt'].date()
        print(f"    Event {j+1}: {onset_date}, duration {ev['duration']} days")
    if len(events) > 20:
        print(f"    ... and {len(events) - 20} more events")

    # Set train/test split: use 70/30 based on primary station's time range
    date_range = primary_df['dt'].max() - primary_df['dt'].min()
    split_date = primary_df['dt'].min() + date_range * 0.7
    split_date = pd.Timestamp(split_date.date())
    print(f"  Train/test split: {split_date.date()} (70/30 temporal)")

    # Drop sparse columns
    dropped = []
    for col in list(primary_df.columns):
        if col in ['dt', 'hyp', 'onset']:
            continue
        if primary_df[col].notna().mean() < 0.5:
            dropped.append(col)
            primary_df.drop(col, axis=1, inplace=True)
    if dropped:
        print(f"  Dropped sparse columns: {', '.join(dropped)}")

    # Build cross-station dict (only stations that passed quality check)
    cross_dfs = {sid: station_dfs[sid] for sid in cross_station_ids}

    # Feature engineering
    print(f"\n[4/6] Engineering features ({len(cross_dfs)} cross-stations"
          f"{' + WOD' if wod_features is not None else ''}"
          f"{' + satellite' if sat_features is not None else ''})...")
    primary_df = engineer_features(primary_df, cross_station_dfs=cross_dfs,
                                   wod_features_df=wod_features,
                                   sat_df=sat_features)
    print(f"  Total features: {len(primary_df.columns)}")

    # Train models
    print("\n[5/6] Training onset models (5 feature sets × 4 lead times)...")
    results = train_onset_models(primary_df, events, cross_station_ids,
                                 split_date=split_date)

    # Save
    print("\n[6/6] Saving results...")
    with open('multistation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to multistation_results.json")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Lead':<6s} {'Feature Set':<18s} {'AUC-ROC':>8s} {'AUC-PR':>8s} {'F1':>6s}")
    print("-" * 50)
    for key, vals in sorted(results.items()):
        parts = key.split('_', 1)
        lead = parts[0]
        name = parts[1]
        print(f"{lead:<6s} {name:<18s} {vals['auc_roc']:>8.3f} "
              f"{vals['auc_pr']:>8.3f} {vals['f1']:>6.3f}")


if __name__ == '__main__':
    main()
