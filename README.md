# Predicting Coastal Hypoxia With AQUAVIEW

Multi-source hypoxia onset prediction for the northern Gulf of Mexico, using [AQUAVIEW](https://aquaview.org) as the unified discovery and data access layer.

**[Read the blog post →](https://aquaview-dah.github.io/aquaview-hypoxia-demo/)** · **[Live Dashboard →](https://aquaview-dah.github.io/aquaview-hypoxia-demo/dashboard.html)**

## What This Does

An XGBoost model predicts when new hypoxia events (dissolved oxygen < 2 mg/L) will begin at coastal monitoring stations, 1 to 7 days in advance. The pipeline pulls data from three different ocean data systems through a single catalog API:

- **NDBC** — 12 fixed monitoring stations with dissolved oxygen, temperature, and salinity (via OPeNDAP/THREDDS)
- **CoastWatch** — Satellite SST, chlorophyll-a, sea surface height, salinity, and eddy kinetic energy (via ERDDAP)
- **WOD** — World Ocean Database offshore oxygen profiles (via S3)

AQUAVIEW's STAC API handles discovery and provides download URLs for all three systems. The scripts go from catalog query to model output without constructing data access URLs manually.

## Key Results

At the primary station (GBHM6, Bayou Heron MS), the satellite-enriched model outperforms station-only models at longer lead times:

| Lead Time | Cross-Source (AUC-ROC / AUC-PR) | NDBC-Only | Local-Only |
|-----------|--------------------------------|-----------|------------|
| 1 day     | 0.917 / 0.398                  | 0.921 / 0.419 | 0.919 / 0.424 |
| 5 days    | 0.866 / 0.610                  | 0.847 / 0.555 | 0.843 / 0.541 |
| 7 days    | 0.853 / 0.642                  | 0.842 / 0.590 | 0.843 / 0.608 |

Satellite features (SST anomalies, eddy kinetic energy) add the most value at 5–7 day lead times, where basin-scale signals arrive at the coast days later.

## Dashboard

The live dashboard at [aquaview-dah.github.io/aquaview-hypoxia-demo/dashboard.html](https://aquaview-dah.github.io/aquaview-hypoxia-demo/dashboard.html) shows current conditions and onset predictions for all stations. It updates every 6 hours via GitHub Actions:

1. A cron job fetches realtime NDBC data and CoastWatch satellite observations
2. Features are computed using the same pipeline as training
3. Pre-trained per-station XGBoost models generate onset probabilities at 1, 3, 5, and 7 day lead times
4. Predictions are written to `docs/data/latest.json` and the dashboard reads that file

Stations with fewer than 15 historical onset events show current conditions only (no predictions). This is a research demo, not an operational forecast.

### Running the Dashboard Locally

```bash
# Train per-station models (one-time, after data extraction)
cd src
python train_models.py

# Generate predictions from current NDBC/satellite data
python update_predictions.py
```

## Repository Structure

```
aquaview-hypoxia-demo/
├── .github/workflows/
│   └── update-predictions.yml   # Cron job: fetch data + predict every 6h
├── docs/
│   ├── index.html               # Blog post (GitHub Pages)
│   ├── dashboard.html           # Live prediction dashboard
│   └── data/
│       └── latest.json          # Current predictions (auto-updated)
├── models/                      # Trained XGBoost models (per-station)
├── src/
│   ├── multistation_extract.py  # AQUAVIEW discovery + data download
│   ├── hypoxia_forecast_pipeline.py  # Feature engineering + model eval
│   ├── train_models.py          # Per-station model training
│   └── update_predictions.py    # Realtime fetch + inference
├── blog_post_draft.md
├── requirements.txt
└── README.md
```

## Running the Analysis

### Prerequisites

Python 3.9+ and the following packages:

```bash
pip install -r requirements.txt
```

### Step 1: Extract Data

```bash
cd src
python multistation_extract.py
```

This queries AQUAVIEW to discover stations and satellite products, then downloads data from NDBC (OPeNDAP), CoastWatch (ERDDAP), and WOD (S3). Output goes to `station_data/`.

The extraction takes 5–10 minutes depending on network speed and ERDDAP server responsiveness. No API keys are needed — all data sources are public.

### Step 2: Run the Forecast Pipeline

```bash
python hypoxia_forecast_pipeline.py
```

This loads the extracted CSVs, engineers features (lags, rolling means, cross-station deltas, satellite anomalies), trains XGBoost models at 1/3/5/7-day lead times across multiple feature sets, and writes results to `multistation_results.json`.

### Step 3: Train Per-Station Models (for dashboard)

```bash
python train_models.py
```

Trains and saves XGBoost models for every station with enough hypoxic events. Models are saved to `models/` as joblib files.

### Step 4: Generate Live Predictions

```bash
python update_predictions.py
```

Fetches current NDBC realtime data and CoastWatch satellite observations, runs the trained models, and writes predictions to `docs/data/latest.json`.

## How AQUAVIEW Is Used

The extraction script makes three types of AQUAVIEW API calls:

1. **Search** — `GET /search?collections=NDBC&bbox=-91,28,-85,31` discovers stations and datasets matching spatial/temporal criteria
2. **Item lookup** — `GET /collections/{collection}/items/{id}` retrieves full metadata and asset URLs for each dataset
3. **Download** — Asset URLs point to the actual data: OPeNDAP endpoints for NDBC, ERDDAP CSV endpoints for CoastWatch, S3 paths for WOD

The script doesn't construct download URLs. It uses whatever URL the catalog provides in each item's `assets` field.

## Notes

- WOD annual files are multi-gigabyte global datasets. The extraction script discovers them through AQUAVIEW and has the S3 URLs, but downloading them requires byte-range requests or `s3fs`. This analysis ran without WOD data — the cross-source results reflect NDBC + satellite only.
- CoastWatch ERDDAP subsetting can be finicky. The script handles per-variable dimension constraints, hidden altitude axes, and coordinate snapping, but ERDDAP server availability varies.
- The blog post and interactive map are served from `docs/index.html` via GitHub Pages. The dashboard is at `docs/dashboard.html`.

## License

MIT

## Citation

If you use this analysis or the AQUAVIEW catalog:

> AQUAVIEW STAC Catalog. Data from NOAA/NWS National Data Buoy Center, NOAA/NCEI World Ocean Database, and NOAA/NESDIS CoastWatch.
