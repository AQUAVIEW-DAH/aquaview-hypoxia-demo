#!/usr/bin/env python3
"""
Per-Station Hypoxia Onset Model Training
=========================================
Trains XGBoost onset prediction models for every NDBC station that has
enough hypoxic events. Saves model artifacts to ../models/ for use by
update_predictions.py and the GitHub Actions cron job.

Requires station data to already be downloaded (run multistation_extract.py
first, or point STATION_DATA_DIR at an existing extraction).

Usage:
    python train_models.py                     # default: station_data/
    python train_models.py --data-dir /path    # custom data directory
"""
import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
HYPOXIA_THRESHOLD = 2.0   # mg/L
LEAD_TIMES = [1, 3, 5, 7]
MIN_ONSET_EVENTS = 15     # minimum events to train a station model
MIN_DAYS = 365            # minimum days of data

NDBC_VARS = {'wt': 'water_temperature', 'sal': 'salinity',
             'o2': 'dissolved_oxygen', 'os': 'o2_saturation',
             'dep': 'depth'}

MODEL_DIR = Path(__file__).parent.parent / "models"
STATION_DATA_DIR = Path("station_data")

# ─── Feature Engineering (shared with update_predictions.py) ─────────────────
def engineer_features(df, cross_station_dfs=None, sat_df=None):
    """
    Engineer features from daily station data. Replicates the logic in
    hypoxia_forecast_pipeline.py so that training and inference use
    identical feature sets.
    """
    df = df.copy()

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

    # Cross-station features
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

            for col in ['wm', 'sal', 'o2']:
                src = f'{prefix}_{col}'
                if col in df.columns and src in df.columns:
                    df[f'delta_{col}_{prefix}'] = df[col] - df[src]

            for col in [c for c in df.columns if c.startswith(f'{prefix}_o2')]:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_roll7'] = df[col].rolling(7, min_periods=3).mean()

    # Satellite features
    if sat_df is not None and len(sat_df) > 0:
        sat_merge = sat_df.copy()
        sat_merge["dt"] = pd.to_datetime(sat_merge["dt"])
        df = df.merge(sat_merge, on="dt", how="left")

        sat_cols = [c for c in sat_merge.columns if c.startswith("sat_")]
        for col in sat_cols:
            if col in df.columns:
                for lag in [1, 3, 7]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
                df[f'{col}_roll7'] = df[col].rolling(7, min_periods=3).mean()
                df[f'{col}_diff3'] = df[col].diff(3)
                df[f'{col}_diff7'] = df[col].diff(7)

        if 'sat_sst' in df.columns and 'wm' in df.columns:
            df['sst_insitu_delta'] = df['sat_sst'] - df['wm']

    # Fill remaining NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill().fillna(0)

    return df


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_station_data(data_dir):
    """Load pre-extracted daily station data from CSV files."""
    data_dir = Path(data_dir)
    station_dfs = {}

    # Look for station CSV files
    for f in sorted(data_dir.glob("*_daily.csv")):
        sid = f.stem.replace("_daily", "")
        df = pd.read_csv(f, parse_dates=['dt'])
        if 'o2' in df.columns and df['o2'].notna().sum() > 30:
            if 'hyp' not in df.columns:
                df['hyp'] = (df['o2'] <= HYPOXIA_THRESHOLD).astype(int)
            station_dfs[sid] = df
            print(f"  {sid.upper():>8s}: {len(df):>5d} days, "
                  f"{df['hyp'].sum()} hypoxic")

    if not station_dfs:
        # Fallback: try loading from OPeNDAP via station registry
        print("  No CSV files found. Trying OPeNDAP via station registry...")
        station_dfs = load_from_opendap(data_dir)

    return station_dfs


def load_from_opendap(data_dir):
    """Load station data from OPeNDAP (reuses pipeline download logic)."""
    registry_file = data_dir / "station_registry.json"
    if not registry_file.exists():
        print(f"  No station registry at {registry_file}")
        return {}

    with open(registry_file) as f:
        registry = json.load(f)

    station_dfs = {}
    for sid, info in registry.items():
        try:
            all_daily = []
            for url in info.get('opendap_urls', []):
                opendap_url = url.replace("/thredds/fileServer/",
                                          "/thredds/dodsC/")
                ds = xr.open_dataset(opendap_url)
                df_year = pd.DataFrame({'time': ds['time'].values})
                for local_name, remote_name in NDBC_VARS.items():
                    if remote_name in ds:
                        vals = ds[remote_name].values.flatten()
                        vals = vals[:len(df_year)].astype(float)
                        vals[(vals > 900) | (vals < -10)] = np.nan
                        if remote_name == "dissolved_oxygen":
                            vals[vals > 25] = np.nan
                        elif remote_name == "water_temperature":
                            vals[vals > 50] = np.nan
                        elif remote_name == "salinity":
                            vals[vals > 45] = np.nan
                        df_year[local_name] = vals
                    else:
                        df_year[local_name] = np.nan
                ds.close()

                if 'o2' not in df_year.columns or df_year['o2'].dropna().empty:
                    continue

                df_year['time'] = pd.to_datetime(df_year['time'])
                df_year['date'] = df_year['time'].dt.date

                agg_dict = {'time': 'count'}
                for col, ops in [('wt', [('wm', 'mean'), ('wn', 'min'), ('wx', 'max')]),
                                 ('sal', [('sal', 'mean')]),
                                 ('o2', [('o2', 'mean'), ('o2n', 'min'),
                                         ('o2x', 'max'), ('o2sd', 'std')]),
                                 ('os', [('os', 'mean')]),
                                 ('dep', [('dep', 'mean')])]:
                    if col in df_year.columns:
                        for out_name, func in ops:
                            agg_dict[col] = func  # simplified
                grouped = df_year.groupby('date')
                daily_parts = []
                daily_parts.append(grouped['time'].count().rename('n'))
                for col in ['wt', 'sal', 'o2', 'os', 'dep']:
                    if col in df_year.columns:
                        if col == 'wt':
                            daily_parts.append(grouped[col].mean().rename('wm'))
                            daily_parts.append(grouped[col].min().rename('wn'))
                            daily_parts.append(grouped[col].max().rename('wx'))
                        elif col == 'o2':
                            daily_parts.append(grouped[col].mean().rename('o2'))
                            daily_parts.append(grouped[col].min().rename('o2n'))
                            daily_parts.append(grouped[col].max().rename('o2x'))
                            daily_parts.append(grouped[col].std().rename('o2sd'))
                        else:
                            daily_parts.append(grouped[col].mean().rename(col))

                daily = pd.concat(daily_parts, axis=1).reset_index()
                daily['dt'] = pd.to_datetime(daily['date'])
                daily['hyp'] = (daily['o2'] <= HYPOXIA_THRESHOLD).astype(int)
                all_daily.append(daily)

            if all_daily:
                result = pd.concat(all_daily).sort_values('dt').reset_index(drop=True)
                result = result.drop(columns=['date'], errors='ignore')
                if result['o2'].notna().sum() > 30 and result['o2'].std() > 0.01:
                    station_dfs[sid] = result
                    print(f"  {sid.upper():>8s}: {len(result):>5d} days, "
                          f"{result['hyp'].sum()} hypoxic (OPeNDAP)")
        except Exception as e:
            print(f"  {sid.upper()}: failed ({e})")

    return station_dfs


# ─── Event Detection ─────────────────────────────────────────────────────────
def detect_onset_events(df):
    """Detect hypoxia onset events (transitions from normal to hypoxic)."""
    df = df.copy()
    df['onset'] = ((df['hyp'] == 1) &
                   (df['hyp'].shift(1, fill_value=0) == 0)).astype(int)
    events = []
    start_idx = None
    for i in range(len(df)):
        if df.iloc[i]['onset'] == 1:
            start_idx = i
        if start_idx is not None and df.iloc[i]['hyp'] == 1:
            if i == len(df) - 1 or df.iloc[i + 1]['hyp'] == 0:
                events.append({'start': start_idx, 'end': i,
                              'duration': i - start_idx + 1})
                start_idx = None
    return events


# ─── Model Training ──────────────────────────────────────────────────────────
def train_station_models(station_id, df, events, cross_dfs, sat_df=None):
    """Train onset models for a single station at all lead times."""
    print(f"\n{'='*60}")
    print(f"Training models for {station_id.upper()}")
    print(f"{'='*60}")
    print(f"  Data: {len(df)} days, {df['hyp'].sum()} hypoxic days, "
          f"{len(events)} onset events")

    # Feature engineering
    cross_station_ids = list(cross_dfs.keys())
    df = engineer_features(df, cross_station_dfs=cross_dfs, sat_df=sat_df)

    # Create onset targets
    for lead in LEAD_TIMES:
        target = pd.Series(0, index=df.index)
        for e in events:
            onset_idx = e['start']
            for d in range(1, lead + 1):
                if onset_idx - d >= 0:
                    target.iloc[onset_idx - d] = 1
        df[f'onset_{lead}d'] = target

    # Train only on non-hypoxic days
    df_valid = df[df['hyp'] == 0].copy()

    exclude = {'dt', 'hyp', 'onset'} | {f'onset_{l}d' for l in LEAD_TIMES}
    all_features = [c for c in df_valid.columns if c not in exclude
                    and df_valid[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Temporal split at 70%
    date_range = df['dt'].max() - df['dt'].min()
    split_date = df['dt'].min() + date_range * 0.7
    split_date = pd.Timestamp(split_date.date())

    train = df_valid[df_valid['dt'] < split_date].dropna(subset=all_features)
    test = df_valid[df_valid['dt'] >= split_date].dropna(subset=all_features)

    print(f"  Features: {len(all_features)}")
    print(f"  Train: {len(train)} days (before {split_date.date()})")
    print(f"  Test:  {len(test)} days (after {split_date.date()})")

    if len(train) < 100 or len(test) < 30:
        print(f"  SKIP: insufficient data for train/test split")
        return None

    models = {}
    metrics = {}

    for lead in LEAD_TIMES:
        target_col = f'onset_{lead}d'
        X_tr, y_tr = train[all_features], train[target_col]
        X_te, y_te = test[all_features], test[target_col]

        if y_tr.sum() < 5:
            print(f"  {lead}d: SKIP (only {y_tr.sum()} positive training samples)")
            continue

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

        models[lead] = model
        metrics[lead] = {'auc_roc': round(auc, 4),
                         'auc_pr': round(pr, 4),
                         'f1': round(best_f1, 4)}
        print(f"  {lead}d: AUC-ROC={auc:.3f}  PR-AUC={pr:.3f}  F1={best_f1:.3f}")

    if not models:
        print(f"  No models trained for {station_id.upper()}")
        return None

    return {
        'models': models,
        'metrics': metrics,
        'feature_names': all_features,
        'cross_station_ids': cross_station_ids,
        'n_events': len(events),
        'n_days': len(df),
        'date_range': [str(df['dt'].min().date()), str(df['dt'].max().date())],
        'split_date': str(split_date.date()),
    }


# ─── Save Models ─────────────────────────────────────────────────────────────
def save_models(station_id, result):
    """Save trained models and metadata to models/ directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for lead, model in result['models'].items():
        model_path = MODEL_DIR / f"{station_id}_{lead}d.pkl"
        joblib.dump(model, model_path)

    # Save feature names (same for all lead times of a station)
    features_path = MODEL_DIR / f"{station_id}_features.json"
    with open(features_path, 'w') as f:
        json.dump(result['feature_names'], f)

    # Save cross-station IDs (needed for feature engineering at inference)
    meta = {
        'station_id': station_id,
        'cross_station_ids': result['cross_station_ids'],
        'metrics': result['metrics'],
        'n_events': result['n_events'],
        'n_days': result['n_days'],
        'date_range': result['date_range'],
        'split_date': result['split_date'],
        'lead_times': [l for l in result['models'].keys()],
    }
    meta_path = MODEL_DIR / f"{station_id}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {len(result['models'])} models → {MODEL_DIR}/")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train per-station hypoxia models')
    parser.add_argument('--data-dir', type=str, default='station_data',
                        help='Directory with station data (CSVs or registry)')
    parser.add_argument('--min-events', type=int, default=MIN_ONSET_EVENTS,
                        help=f'Minimum onset events to train (default: {MIN_ONSET_EVENTS})')
    args = parser.parse_args()

    print("=" * 60)
    print("PER-STATION HYPOXIA ONSET MODEL TRAINING")
    print("=" * 60)

    # Load all station data
    print("\n[1/4] Loading station data...")
    station_dfs = load_station_data(args.data_dir)
    if not station_dfs:
        print("No station data found. Run multistation_extract.py first.")
        sys.exit(1)
    print(f"  Loaded {len(station_dfs)} stations")

    # Load satellite data if available
    sat_df = None
    sat_file = Path(args.data_dir) / "satellite_merged.csv"
    if sat_file.exists():
        print(f"\n  Loading satellite features from {sat_file}")
        sat_df = pd.read_csv(sat_file)
        sat_df["dt"] = pd.to_datetime(sat_df["dt"])
        print(f"  Satellite: {len(sat_df)} days")

    # Identify which stations have enough hypoxic events
    print("\n[2/4] Identifying trainable stations...")
    trainable = {}
    for sid, df in station_dfs.items():
        events = detect_onset_events(df)
        if len(events) >= args.min_events and len(df) >= MIN_DAYS:
            trainable[sid] = events
            print(f"  {sid.upper():>8s}: {len(events):>3d} events ✓")
        else:
            reason = (f"{len(events)} events" if len(events) < args.min_events
                      else f"{len(df)} days")
            print(f"  {sid.upper():>8s}: {reason} — skipping")

    if not trainable:
        print("\nNo stations have enough onset events for training.")
        sys.exit(1)

    print(f"\n  Training models for {len(trainable)} station(s)")

    # Train models for each qualifying station
    print("\n[3/4] Training models...")
    all_meta = {}
    for sid, events in trainable.items():
        df = station_dfs[sid].copy()
        cross_dfs = {k: v for k, v in station_dfs.items() if k != sid}

        result = train_station_models(sid, df, events, cross_dfs, sat_df=sat_df)
        if result is not None:
            save_models(sid, result)
            all_meta[sid] = {
                'metrics': result['metrics'],
                'n_events': result['n_events'],
                'n_days': result['n_days'],
                'date_range': result['date_range'],
                'lead_times': [l for l in result['models'].keys()],
            }

    # Save combined metadata
    print("\n[4/4] Saving station metadata...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    combined_meta_path = MODEL_DIR / "station_meta.json"
    with open(combined_meta_path, 'w') as f:
        json.dump(all_meta, f, indent=2)
    print(f"  Saved combined metadata → {combined_meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for sid, meta in all_meta.items():
        leads = meta['lead_times']
        aucs = []
        for l in leads:
            # metrics may be keyed by int or str depending on context
            m = meta['metrics'].get(l) or meta['metrics'].get(str(l))
            if m:
                aucs.append(m['auc_roc'])
        best_auc = max(aucs) if aucs else 0
        print(f"  {sid.upper():>8s}: {meta['n_events']} events, "
              f"leads {leads}, best AUC-ROC {best_auc:.3f}")
    print(f"\nModels saved to {MODEL_DIR.resolve()}/")
    print("Run update_predictions.py to generate live predictions.")


if __name__ == '__main__':
    main()
