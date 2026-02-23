#!/usr/bin/env python3
"""
Live Hypoxia Prediction Update
===============================
Fetches current NDBC realtime data and CoastWatch satellite observations,
engineers features, runs pre-trained XGBoost models, and writes predictions
to docs/data/latest.json for the static dashboard.

Uses AQUAVIEW's STAC API to discover which stations have active realtime
ocean data feeds, avoiding 404 errors on decommissioned stations.

Designed to run via GitHub Actions cron (every 6 hours) or manually.

Usage:
    python update_predictions.py
"""
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import joblib
import requests
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "docs" / "data"
CACHE_DIR = ROOT / ".cache"
STATION_REGISTRY = ROOT / "docs" / "data" / "station_registry.json"

HYPOXIA_THRESHOLD = 2.0
LEAD_TIMES = [1, 3, 5, 7]

# AQUAVIEW STAC API
AQUAVIEW_API = "https://aquaview-sfeos-1025757962819.us-east1.run.app"

# ─── AQUAVIEW Discovery ──────────────────────────────────────────────────────

def discover_realtime_stations(bbox="-98,25,-80,31"):
    """
    Query AQUAVIEW STAC API to discover which NDBC stations in the study area
    have active realtime ocean feeds (rt_ocean asset).

    Returns dict: {station_id: {name, lat, lon, rt_url, has_rt_ocean, has_rt_txt}}
    """
    print("  Querying AQUAVIEW for NDBC stations with realtime feeds...")
    stations = {}

    try:
        # Search NDBC stations in the study area using POST (STAC standard)
        url = f"{AQUAVIEW_API}/search"
        west, south, east, north = [float(x) for x in bbox.split(',')]

        # First try POST (STAC standard for search)
        body = {
            'collections': ['NDBC'],
            'bbox': [west, south, east, north],
            'limit': 100,
        }
        try:
            r = requests.post(url, json=body, timeout=30)
            r.raise_for_status()
        except Exception:
            # Fall back to GET if POST not supported
            params = {
                'collections': 'NDBC',
                'bbox': bbox,
                'limit': 100,
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()

        data = r.json()

        # Handle both STAC FeatureCollection and custom response formats
        items = data.get('features', [])
        if not items:
            items = data.get('items', [])

        # Paginate to get all stations
        next_token = data.get('next_token') or data.get('token')
        page = 1
        while next_token and len(items) < 500:
            page += 1
            try:
                body['token'] = next_token
                r = requests.post(url, json=body, timeout=30)
                r.raise_for_status()
                data = r.json()
                new_items = data.get('features', []) or data.get('items', [])
                if not new_items:
                    break
                items.extend(new_items)
                next_token = data.get('next_token') or data.get('token')
            except Exception:
                break

        for item in items:
            props = item.get('properties', {})
            assets = item.get('assets', {})
            geom = item.get('geometry', {})
            coords = geom.get('coordinates', [None, None])

            # Extract station ID from the item ID (e.g., ndbc_gbhm6 -> gbhm6)
            item_id = item.get('id', '')
            sid = item_id.replace('ndbc_', '').lower()

            has_rt_ocean = 'rt_ocean' in assets
            has_rt_txt = 'rt_txt' in assets

            stations[sid] = {
                'name': props.get('title', sid.upper()),
                'lat': coords[1] if len(coords) > 1 else None,
                'lon': coords[0] if len(coords) > 0 else None,
                'has_rt_ocean': has_rt_ocean,
                'has_rt_txt': has_rt_txt,
                'rt_ocean_url': assets['rt_ocean']['href'] if has_rt_ocean else None,
                'rt_txt_url': assets['rt_txt']['href'] if has_rt_txt else None,
                'variables': props.get('aquaview:variables', []),
            }

        rt_ocean_count = sum(1 for s in stations.values() if s['has_rt_ocean'])
        print(f"  Found {len(stations)} NDBC stations, "
              f"{rt_ocean_count} with realtime ocean feeds")

    except Exception as e:
        print(f"  AQUAVIEW discovery failed ({e}), using fallback station list")

    return stations


# ─── NDBC Realtime Parsing ───────────────────────────────────────────────────

# Column mapping for NDBC .ocean realtime files
# Format: #YY MM DD hh mm DEPTH OTMP COND SAL O2% O2PPM CLCON TURB PH EH
OCEAN_COLS = ['YY', 'MM', 'DD', 'hh', 'mm',
              'DEPTH', 'OTMP', 'COND', 'SAL', 'O2PCT', 'O2PPM',
              'CLCON', 'TURB', 'PH', 'EH']

# Map NDBC realtime columns to our internal names
REALTIME_MAP = {
    'OTMP': 'wt',     # water temperature
    'SAL': 'sal',     # salinity
    'O2PPM': 'o2',    # dissolved oxygen mg/L
    'O2PCT': 'os',    # O2 saturation %
    'DEPTH': 'dep',   # depth
}


NDBC_HEADERS = {
    'User-Agent': 'HypoxiaDashboard/1.0 (research; github.com/aquaview-hypoxia-demo)',
    'Accept': 'text/plain, text/html, */*',
}


def fetch_ndbc_realtime(station_id, url=None, max_retries=2):
    """
    Fetch and parse NDBC realtime ocean data (~45 days).
    Returns a daily-aggregated DataFrame or None on failure.

    If url is provided, uses that directly (from AQUAVIEW asset).
    Otherwise constructs the standard .ocean URL.
    """
    if url is None:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id.upper()}.ocean"

    print(f"  {station_id.upper()}: fetching {url.split('/')[-1]}...", end=" ", flush=True)

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30, headers=NDBC_HEADERS)
            if r.status_code == 404:
                print("404 (no feed)")
                return None
            if r.status_code == 403:
                print(f"403 (blocked)")
                return None
            r.raise_for_status()
            df = parse_ndbc_ocean_text(r.text, station_id)
            if df is not None and len(df) > 0:
                print(f"OK ({len(df)} days)")
            else:
                lines = r.text.strip().split('\n')
                print(f"OK but no parseable data ({len(r.text)} bytes, {len(lines)} lines)")
            return df
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"connection error, retry in {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"connection failed: {type(e).__name__}")
                return None
        except requests.exceptions.HTTPError as e:
            if '404' in str(e):
                print("404 (no feed)")
                return None
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"HTTP {r.status_code}, retry in {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"HTTP error: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"{type(e).__name__}, retry in {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"failed: {type(e).__name__}: {e}")
                return None


def parse_ndbc_ocean_text(text, station_id):
    """Parse NDBC .ocean realtime text into daily-aggregated DataFrame."""
    lines = text.strip().split('\n')

    # Skip header lines (start with #)
    data_lines = []
    for line in lines:
        if line.startswith('#'):
            continue
        data_lines.append(line)

    if not data_lines:
        return None

    rows = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 6:  # need at least datetime + 1 value
            continue
        try:
            row = {}
            # Parse datetime
            yy = int(parts[0])
            mm = int(parts[1])
            dd = int(parts[2])
            hh = int(parts[3])
            mn = int(parts[4])
            row['time'] = pd.Timestamp(year=yy, month=mm, day=dd,
                                       hour=hh, minute=mn)

            # Parse values — NDBC uses 'MM' for missing data
            for i, col in enumerate(OCEAN_COLS[5:], start=5):
                if i < len(parts):
                    raw = parts[i]
                    if raw == 'MM' or raw == 'mm':
                        row[col] = np.nan
                    else:
                        try:
                            val = float(raw)
                            if val in (99.0, 999.0, 9999.0) or val > 900:
                                val = np.nan
                            row[col] = val
                        except ValueError:
                            row[col] = np.nan
                else:
                    row[col] = np.nan

            rows.append(row)
        except (ValueError, IndexError):
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df['date'] = df['time'].dt.date

    # Rename to internal column names
    for ndbc_col, our_col in REALTIME_MAP.items():
        if ndbc_col in df.columns:
            df[our_col] = df[ndbc_col]

    # Apply physical range limits
    if 'o2' in df.columns:
        df.loc[df['o2'] > 25, 'o2'] = np.nan
        df.loc[df['o2'] < 0, 'o2'] = np.nan
    if 'wt' in df.columns:
        df.loc[df['wt'] > 50, 'wt'] = np.nan
    if 'sal' in df.columns:
        df.loc[df['sal'] > 45, 'sal'] = np.nan

    # Daily aggregation
    daily_parts = [df.groupby('date')['time'].count().rename('n')]
    for col in ['wt', 'sal', 'o2', 'os', 'dep']:
        if col in df.columns:
            if col == 'wt':
                daily_parts.append(df.groupby('date')[col].mean().rename('wm'))
                daily_parts.append(df.groupby('date')[col].min().rename('wn'))
                daily_parts.append(df.groupby('date')[col].max().rename('wx'))
            elif col == 'o2':
                daily_parts.append(df.groupby('date')[col].mean().rename('o2'))
                daily_parts.append(df.groupby('date')[col].min().rename('o2n'))
                daily_parts.append(df.groupby('date')[col].max().rename('o2x'))
                daily_parts.append(df.groupby('date')[col].std().rename('o2sd'))
            else:
                daily_parts.append(df.groupby('date')[col].mean().rename(col))

    daily = pd.concat(daily_parts, axis=1).reset_index()
    daily['dt'] = pd.to_datetime(daily['date'])
    daily['hyp'] = (daily['o2'] <= HYPOXIA_THRESHOLD).astype(int) if 'o2' in daily.columns else 0
    daily = daily.drop(columns=['date']).sort_values('dt').reset_index(drop=True)

    return daily


# ─── CoastWatch Satellite Data ───────────────────────────────────────────────
#
# Dataset IDs discovered via AQUAVIEW COASTWATCH collection and confirmed
# against the working multistation_extract.py pipeline.
#
# Primary: Multiparameter Eddy Index (0.25°, daily) — provides SST + chl-a
#   AQUAVIEW ID: noaacweddymesiplusdaily
#   Variables: sst, chla, ssh, sss, eke
#   Coverage: 2020-05-05 to present, global
#
# Fallbacks (individual products, 4km):
#   SST:  noaacwSNPPACSPOSSTL3GCDaily  (var: sea_surface_temperature)
#   Chl:  noaacwS3BOLCIchlaDaily        (var: chlor_a)
#

ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap/griddap"

# Primary multi-param dataset (matches original pipeline)
EDDY_DATASET = 'noaacweddymesiplusdaily'
EDDY_VARS = ['sst', 'chla']  # we only need SST and chl-a for the dashboard

# Fallback individual datasets
FALLBACK_SST = {'dataset': 'noaacwSNPPACSPOSSTL3GCDaily',
                'variable': 'sea_surface_temperature'}
FALLBACK_CHL = {'dataset': 'noaacwS3BOLCIchlaDaily',
                'variable': 'chlor_a'}

# Study area bounding box (same as original pipeline)
BBOX = {'south': 25.0, 'north': 31.0, 'west': -98.0, 'east': -80.0}


def _parse_erddap_csv(text, col_map):
    """
    Parse ERDDAP CSV response into a DataFrame.
    col_map: dict mapping CSV column name → our internal name.
    Returns DataFrame with 'dt' + mapped columns, or None.
    """
    from io import StringIO
    lines = text.strip().split('\n')
    if len(lines) <= 2:
        return None
    # Line 0 = header names, line 1 = units, lines 2+ = data
    header = lines[0].split(',')
    data_lines = lines[2:]
    if not data_lines:
        return None

    rows = []
    for line in data_lines:
        parts = line.split(',')
        if len(parts) < len(header):
            continue
        row = {}
        for i, col_name in enumerate(header):
            col_name = col_name.strip()
            if col_name == 'time':
                row['time'] = parts[i].strip()
            elif col_name in col_map:
                raw = parts[i].strip()
                try:
                    row[col_map[col_name]] = float(raw) if raw != 'NaN' else np.nan
                except ValueError:
                    row[col_map[col_name]] = np.nan
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if 'time' not in df.columns:
        return None
    df['dt'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['dt'])
    df['date'] = df['dt'].dt.date
    # Daily spatial mean
    value_cols = [c for c in df.columns if c not in ('time', 'dt', 'date')]
    daily = df.groupby('date')[value_cols].mean().reset_index()
    daily['dt'] = pd.to_datetime(daily['date'])
    return daily[['dt'] + value_cols]


def fetch_satellite_data(days_back=45):
    """
    Fetch satellite data from CoastWatch ERDDAP for the study area.

    Strategy (matches original pipeline):
    1. Try the multi-param eddy dataset (SST + chl-a in one request, 0.25°)
    2. Fall back to individual VIIRS SST and OLCI chl-a datasets (4km)
    """
    start_date = datetime.utcnow() - timedelta(days=days_back)
    start_str = start_date.strftime('%Y-%m-%dT00:00:00Z')

    # Use ERDDAP's "last" keyword for the end time — avoids 404 errors
    # when our "now" timestamp exceeds the dataset's latest available date
    # (NRT products often lag 1-18 days behind real time).
    end_kw = 'last'

    # Snap bbox to 0.25° grid (matching original pipeline for eddy dataset)
    lat_s = round(BBOX['south'] * 4) / 4
    lat_n = round(BBOX['north'] * 4) / 4
    lon_w = round(BBOX['west'] * 4) / 4
    lon_e = round(BBOX['east'] * 4) / 4

    # ── Try primary: multi-param eddy dataset (3 dims: time, lat, lon) ──
    dims = (f"[({start_str}):1:({end_kw})]"
            f"[({lat_s}):1:({lat_n})]"
            f"[({lon_w}):1:({lon_e})]")
    query_vars = ",".join(v + dims for v in EDDY_VARS)
    url = f"{ERDDAP_BASE}/{EDDY_DATASET}.csv?{query_vars}"

    try:
        print(f"  Trying eddy multi-param dataset ({EDDY_DATASET})...")
        r = requests.get(url, timeout=120, headers=NDBC_HEADERS)
        r.raise_for_status()
        col_map = {'sst': 'sat_sst', 'chla': 'sat_chl'}
        daily = _parse_erddap_csv(r.text, col_map)
        if daily is not None and len(daily) > 0:
            print(f"  Eddy dataset: {len(daily)} days (SST + chl-a)")
            return daily
        else:
            print(f"  Eddy dataset returned empty data")
    except Exception as e:
        print(f"  Eddy dataset failed ({e})")

    # ── Fallback: individual datasets (4 dims: time, altitude, lat, lon) ──
    # VIIRS SST and OLCI chl-a have an altitude dimension that must be included.
    # The original pipeline used [(0.0)] for altitude.
    print(f"  Falling back to individual satellite datasets...")
    results = {}

    for label, config, out_col in [
        ('SST', FALLBACK_SST, 'sat_sst'),
        ('Chl-a', FALLBACK_CHL, 'sat_chl'),
    ]:
        dataset = config['dataset']
        variable = config['variable']
        fb_url = (f"{ERDDAP_BASE}/{dataset}.csv?"
                  f"{variable}"
                  f"[({start_str}):1:({end_kw})]"
                  f"[(0.0)]"
                  f"[({BBOX['south']}):1:({BBOX['north']})]"
                  f"[({BBOX['west']}):1:({BBOX['east']})]")
        try:
            r = requests.get(fb_url, timeout=60, headers=NDBC_HEADERS)
            r.raise_for_status()
            col_map = {variable: out_col}
            daily = _parse_erddap_csv(r.text, col_map)
            if daily is not None and len(daily) > 0:
                results[out_col] = daily
                print(f"  {label}: {len(daily)} days from {dataset}")
            else:
                print(f"  {label}: empty data from {dataset}")
        except Exception as e:
            print(f"  {label}: failed ({e})")

    if not results:
        return None

    merged = None
    for var_name, df in results.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='dt', how='outer')

    return merged.sort_values('dt').reset_index(drop=True) if merged is not None else None


# ─── Feature Engineering (must match training) ──────────────────────────────
# Import from train_models.py to ensure consistency
sys.path.insert(0, str(Path(__file__).parent))
from train_models import engineer_features


# ─── Load Models & Predict ───────────────────────────────────────────────────
def load_station_model(station_id):
    """Load trained model artifacts for a station."""
    meta_path = MODEL_DIR / f"{station_id}_meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    features_path = MODEL_DIR / f"{station_id}_features.json"
    with open(features_path) as f:
        feature_names = json.load(f)

    models = {}
    for lead in meta['lead_times']:
        model_path = MODEL_DIR / f"{station_id}_{lead}d.pkl"
        if model_path.exists():
            models[lead] = joblib.load(model_path)

    return {
        'meta': meta,
        'feature_names': feature_names,
        'models': models,
    }


def predict_station(station_id, station_df, cross_dfs, sat_df, model_info):
    """Run inference for a single station, returning predictions dict."""
    feature_names = model_info['feature_names']
    models = model_info['models']

    # Engineer features (same logic as training)
    cross_station_ids = model_info['meta']['cross_station_ids']
    filtered_cross = {k: v for k, v in cross_dfs.items()
                      if k in cross_station_ids}

    df = engineer_features(station_df, cross_station_dfs=filtered_cross,
                           sat_df=sat_df)

    # Ensure feature alignment
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0

    # Use only the latest row for prediction
    latest = df.iloc[[-1]]
    X = latest[feature_names]

    predictions = {}
    for lead, model in models.items():
        try:
            prob = float(model.predict_proba(X)[:, 1][0])
            risk = 'low' if prob < 0.3 else ('moderate' if prob < 0.6 else 'high')
            predictions[f'{lead}d'] = {
                'prob': round(prob, 4),
                'risk': risk,
            }
        except Exception as e:
            print(f"  {station_id.upper()} {lead}d prediction failed: {e}")

    return predictions


# ─── Station Registry ────────────────────────────────────────────────────────
def load_or_build_registry():
    """Load station registry or build from model metadata."""
    if STATION_REGISTRY.exists():
        with open(STATION_REGISTRY) as f:
            return json.load(f)

    # Build from model metadata files
    registry = {}
    for meta_file in MODEL_DIR.glob("*_meta.json"):
        sid = meta_file.stem.replace("_meta", "")
        with open(meta_file) as f:
            meta = json.load(f)
        registry[sid] = {
            'id': sid,
            'name': sid.upper(),
            'has_model': True,
        }

    return registry


# ─── Main Update Loop ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HYPOXIA PREDICTION UPDATE")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    # Check for trained models
    station_meta_path = MODEL_DIR / "station_meta.json"
    if not station_meta_path.exists():
        print("No trained models found. Run train_models.py first.")
        sys.exit(1)

    with open(station_meta_path) as f:
        station_meta = json.load(f)

    model_stations = list(station_meta.keys())
    print(f"\nModels available for: {', '.join(s.upper() for s in model_stations)}")

    # Load TimesFM reference data (benchmark metrics, model_type assignments)
    tfm_ref_path = MODEL_DIR / "timesfm_reference.json"
    tfm_ref = {}
    tfm_model_info = {}
    if tfm_ref_path.exists():
        with open(tfm_ref_path) as f:
            tfm_data = json.load(f)
        tfm_ref = tfm_data.get('stations', {})
        tfm_model_info = tfm_data.get('model_info', {})
        tfm_stations = [sid for sid, v in tfm_ref.items() if v.get('timesfm_benchmark')]
        xgb_stations = [sid for sid in model_stations if sid not in tfm_stations or not tfm_ref.get(sid, {}).get('timesfm_benchmark')]
        print(f"  TimesFM reference: {len(tfm_stations)} TFM stations, {len(xgb_stations)} XGBoost stations")
    else:
        print("  No TimesFM reference data found (models/timesfm_reference.json)")

    # ── Step 0: Discover realtime feeds via AQUAVIEW ──
    print(f"\n[0/4] Discovering realtime feeds via AQUAVIEW...")
    aquaview_stations = discover_realtime_stations()

    # Build the full station list: model stations + AQUAVIEW stations with rt_ocean
    all_station_ids = list(set(
        model_stations +
        [sid for sid, info in aquaview_stations.items() if info['has_rt_ocean']]
    ))

    # Also load the station registry for additional metadata
    registry_file = ROOT / "src" / "station_data" / "station_registry.json"
    if registry_file.exists():
        with open(registry_file) as f:
            registry = json.load(f)
        for sid in registry:
            if sid not in all_station_ids:
                all_station_ids.append(sid)
    else:
        registry = {}

    # Determine which stations to actually fetch (only those with rt_ocean)
    stations_with_rt = {
        sid for sid, info in aquaview_stations.items()
        if info['has_rt_ocean']
    }

    # Only fetch stations with confirmed rt_ocean assets.
    # Stations not found in AQUAVIEW or without rt_ocean are skipped --
    # blindly trying them just produces 404 errors.
    fetch_station_ids = []
    skip_station_ids = []
    for sid in all_station_ids:
        if sid in stations_with_rt:
            fetch_station_ids.append(sid)
        else:
            skip_station_ids.append(sid)

    if skip_station_ids:
        print(f"  Skipping {len(skip_station_ids)} stations without realtime ocean feeds: "
              f"{', '.join(s.upper() for s in skip_station_ids[:5])}"
              f"{'...' if len(skip_station_ids) > 5 else ''}")

    # Fetch NDBC realtime data (only for stations with known feeds)
    print(f"\n[1/4] Fetching NDBC realtime data ({len(fetch_station_ids)} stations)...")
    station_dfs = {}
    for sid in fetch_station_ids:
        # Use AQUAVIEW-provided URL if available
        rt_url = None
        if sid in aquaview_stations and aquaview_stations[sid]['rt_ocean_url']:
            rt_url = aquaview_stations[sid]['rt_ocean_url']

        df = fetch_ndbc_realtime(sid, url=rt_url)
        if df is not None and len(df) > 0:
            station_dfs[sid] = df
            do_val = df['o2'].iloc[-1] if 'o2' in df.columns else None
            if do_val is not None and pd.notna(do_val):
                print(f"           latest DO={do_val:.2f} mg/L")

    if not station_dfs:
        print("\nNo station data fetched.")
        if not stations_with_rt:
            print("No stations with realtime ocean feeds found in AQUAVIEW.")
            print("The dashboard will show historical model metadata only.")
        else:
            print("Check network connectivity.")

    # Fetch satellite data
    print(f"\n[2/4] Fetching satellite data from CoastWatch ERDDAP...")
    sat_df = fetch_satellite_data(days_back=45)
    if sat_df is not None:
        print(f"  Satellite data: {len(sat_df)} days")
    else:
        print("  No satellite data available (continuing without)")

    # Load models and run predictions
    print(f"\n[3/4] Running predictions...")
    station_results = []

    for sid in all_station_ids:
        has_model = sid in model_stations
        has_data = sid in station_dfs
        has_rt = sid in stations_with_rt

        # Determine status
        if has_data:
            status = 'active'
        elif has_rt:
            status = 'fetch_failed'  # has feed but fetch failed
        elif has_model:
            status = 'historical_only'  # trained model but no realtime feed
        else:
            status = 'no_data'

        entry = {
            'id': sid.upper(),
            'name': sid.upper(),  # will be enriched from registry/AQUAVIEW
            'has_model': has_model,
            'has_realtime': has_rt,
            'status': status,
        }

        if has_data:
            df = station_dfs[sid]
            latest = df.iloc[-1]
            entry['current'] = {
                'do': round(float(latest.get('o2', np.nan)), 2) if pd.notna(latest.get('o2')) else None,
                'temp': round(float(latest.get('wm', np.nan)), 1) if pd.notna(latest.get('wm')) else None,
                'sal': round(float(latest.get('sal', np.nan)), 1) if pd.notna(latest.get('sal')) else None,
            }
            entry['last_observation'] = str(df['dt'].max().date())
            entry['days_of_data'] = len(df)

            # Check if currently hypoxic
            if entry['current']['do'] is not None:
                entry['currently_hypoxic'] = entry['current']['do'] <= HYPOXIA_THRESHOLD

            if has_model and len(df) >= 14:
                model_info = load_station_model(sid)
                if model_info:
                    cross_dfs = {k: v for k, v in station_dfs.items() if k != sid}
                    predictions = predict_station(sid, df, cross_dfs,
                                                  sat_df, model_info)
                    entry['predictions'] = predictions
                    entry['metrics'] = station_meta[sid].get('metrics', {})
            elif has_model and len(df) < 14:
                entry['status'] = 'insufficient_data'
                entry['predictions'] = None
        else:
            entry['current'] = None
            entry['predictions'] = None

        # Add model_type and TimesFM benchmark data from reference
        sid_lower = sid if sid == sid.lower() else sid.lower()
        if sid_lower in tfm_ref:
            ref = tfm_ref[sid_lower]
            if ref.get('timesfm_benchmark'):
                entry['model_type'] = 'timesfm'
                entry['timesfm_benchmark'] = ref['timesfm_benchmark']
                if ref.get('n_days'):
                    entry['n_days'] = ref['n_days']
                if ref.get('n_hyp_days'):
                    entry['n_hyp_days'] = ref['n_hyp_days']
            elif has_model:
                entry['model_type'] = 'xgboost'
        elif has_model:
            entry['model_type'] = 'xgboost'

        station_results.append(entry)

    # Enrich with registry + AQUAVIEW metadata (names, coordinates)
    for entry in station_results:
        sid = entry['id'].lower()
        # First try local registry
        if sid in registry:
            info = registry[sid]
            entry['name'] = info.get('name', entry['name'])
            entry['lat'] = info.get('lat')
            entry['lon'] = info.get('lon')
        # Then fill gaps from AQUAVIEW
        if sid in aquaview_stations:
            aq = aquaview_stations[sid]
            if 'name' not in entry or entry['name'] == sid.upper():
                # Clean up AQUAVIEW title (remove "NDBC Station XXXX - " prefix)
                title = aq['name']
                if ' - ' in title:
                    title = title.split(' - ', 1)[1]
                entry['name'] = title
            if 'lat' not in entry or entry.get('lat') is None:
                entry['lat'] = aq['lat']
            if 'lon' not in entry or entry.get('lon') is None:
                entry['lon'] = aq['lon']

    # Build output JSON
    print(f"\n[4/4] Writing predictions...")
    output = {
        'updated': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'update_interval_hours': 6,
        'stations': station_results,
        'model_info': tfm_model_info if tfm_model_info else {
            'xgboost': {
                'description': 'Station-specific XGBoost classifier with engineered features',
                'features': 'DO lags, rolling stats, satellite SST/chlorophyll, cross-station deltas',
            },
            'timesfm': {
                'description': 'Google TimesFM foundation model for zero-shot DO forecasting',
                'approach': 'Direct DO time series forecasting, threshold at 2.0 mg/L',
            },
        },
        'summary': {
            'total_stations': len(station_results),
            'stations_with_models': sum(1 for s in station_results if s['has_model']),
            'stations_with_realtime': sum(1 for s in station_results if s.get('has_realtime')),
            'stations_with_data': sum(1 for s in station_results if s['status'] == 'active'),
            'stations_with_predictions': sum(1 for s in station_results if s.get('predictions')),
            'stations_with_warnings': sum(
                1 for s in station_results
                if s.get('predictions') and any(
                    p.get('risk') == 'high'
                    for p in s['predictions'].values()
                )
            ),
        },
    }

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "latest.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Written: {output_path}")

    # Archive daily snapshot
    history_dir = OUTPUT_DIR / "history"
    history_dir.mkdir(exist_ok=True)
    today = datetime.utcnow().strftime('%Y-%m-%d')
    archive_path = history_dir / f"predictions-{today}.json"
    with open(archive_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Archived: {archive_path}")

    # Clean old archives (keep 45 days)
    cutoff = datetime.utcnow() - timedelta(days=45)
    for old_file in history_dir.glob("predictions-*.json"):
        try:
            file_date = datetime.strptime(old_file.stem.split("predictions-")[1],
                                          "%Y-%m-%d")
            if file_date < cutoff:
                old_file.unlink()
                print(f"  Cleaned: {old_file.name}")
        except (ValueError, IndexError):
            pass

    # Summary
    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    sm = output['summary']
    print(f"  {sm['total_stations']} stations total, "
          f"{sm['stations_with_realtime']} with realtime feeds, "
          f"{sm['stations_with_data']} reporting data, "
          f"{sm['stations_with_predictions']} with predictions")
    print("=" * 60)
    for s in station_results:
        status = s['status']
        pred_str = ""
        if s.get('predictions'):
            probs = [f"{k}={v['prob']:.0%}" for k, v in s['predictions'].items()]
            pred_str = " | ".join(probs)
        cur = s.get('current') or {}
        do_str = f"DO={cur['do']}" if cur.get('do') else ""
        rt_flag = "RT" if s.get('has_realtime') else "  "
        model_flag = "M" if s.get('has_model') else " "
        print(f"  {s['id']:>8s} [{rt_flag}{model_flag}] {status:<18s} {do_str:<12s} {pred_str}")


if __name__ == '__main__':
    main()
