#!/usr/bin/env python3
"""
Multi-Station Hypoxia Data Extraction Pipeline
===============================================
Discovers and downloads ocean water quality data from MULTIPLE sources
entirely via AQUAVIEW (aquaview.ai) — a unified STAC catalog for ocean data.

AQUAVIEW is used for BOTH discovery AND access:
  1. Search → find stations/cruises matching our criteria
  2. Get Item → retrieve asset URLs (THREDDS, S3) for each dataset
  3. Download → fetch data from the AQUAVIEW-provided URLs

Sources accessed through AQUAVIEW:
  1. NDBC THREDDS (12 fixed monitoring stations with dissolved oxygen)
  2. WOD (World Ocean Database) — Argo float oxygen profiles via S3
  3. CoastWatch ERDDAP — satellite SST, chlorophyll-a, SSH, SSS, EKE

Run this script in Cursor or any Python 3.9+ environment.
Requires: pip install xarray netCDF4 pandas numpy requests
Optional: pip install s3fs  (for WOD Argo data)

Output: CSV files per station + summary JSON for blog post
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import xarray as xr
import requests

warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Output directory ────────────────────────────────────────────────────────
OUT_DIR = Path("station_data")
OUT_DIR.mkdir(exist_ok=True)

# ─── AQUAVIEW STAC API ───────────────────────────────────────────────────────
# Docs: https://aquaview-sfeos-1025757962819.us-east1.run.app/api.html
AQUAVIEW_API = "https://aquaview-sfeos-1025757962819.us-east1.run.app"

# ─── Study Area ──────────────────────────────────────────────────────────────
BBOX = {"west": -98.0, "south": 25.0, "east": -80.0, "north": 31.0}

# CoastWatch ERDDAP base (fallback when AQUAVIEW asset URL is not usable)
COASTWATCH_ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap/griddap"
HYPOXIA_THRESHOLD = 2.0  # mg/L

# Variables to extract from NDBC ocean files
NDBC_VARS = {
    "water_temperature": "wt",
    "salinity": "sal",
    "dissolved_oxygen": "o2",
    "o2_saturation": "os",
    "depth": "dep",
}
NDBC_OPTIONAL_VARS = {
    "chlorophyll_concentration": "chl",
    "turbidity": "turb",
    "water_ph": "ph",
    "conductivity": "cond",
}

# WOD years to search (must match what's available in AQUAVIEW WOD assets)
WOD_YEARS = [2020, 2021, 2022, 2023, 2024]


# ═════════════════════════════════════════════════════════════════════════════
# AQUAVIEW API Client
# ═════════════════════════════════════════════════════════════════════════════

def aquaview_search(collection=None, q=None, bbox=None, limit=100, paginate_all=False):
    """
    Search AQUAVIEW STAC catalog.

    API: GET /search
    Params: collections, bbox, q, limit
    Returns: (numberMatched, [features])

    If paginate_all=True, follows next_token to retrieve ALL matching items
    (not just the first page).
    """
    params = {"limit": limit}
    if q:
        params["q"] = q
    if bbox:
        params["bbox"] = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
    if collection:
        params["collections"] = collection

    url = f"{AQUAVIEW_API}/search"
    all_features = []
    n_matched = 0
    page = 1

    while True:
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  AQUAVIEW search failed: {e}")
            return n_matched, all_features

        n_matched = data.get("numberMatched", 0)
        features = data.get("features", [])
        all_features.extend(features)
        print(f"  Page {page}: {len(features)} items (total so far: {len(all_features)}/{n_matched})")

        if not paginate_all:
            break

        next_token = data.get("next", None)
        if next_token is None:
            # Also check context.next or links
            links = data.get("links", [])
            next_link = [l for l in links if l.get("rel") == "next"]
            if next_link:
                # Parse token from next link URL
                import urllib.parse
                parsed = urllib.parse.urlparse(next_link[0]["href"])
                qs = urllib.parse.parse_qs(parsed.query)
                next_token = qs.get("token", [None])[0]

        if not next_token or len(features) == 0:
            break

        params["token"] = next_token
        page += 1

    return n_matched, all_features


def aquaview_get_item(collection_id, item_id):
    """
    Get a single item with full details and asset download URLs.

    API: GET /collections/{collection_id}/items/{item_id}
    Returns: Full STAC item with geometry, properties, and assets
    """
    url = f"{AQUAVIEW_API}/collections/{collection_id}/items/{item_id}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  AQUAVIEW get_item failed for {item_id}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: AQUAVIEW Discovery + Station Registry
# ═════════════════════════════════════════════════════════════════════════════

def discover_ndbc_stations():
    """
    Use AQUAVIEW to discover NDBC stations with dissolved oxygen in our study area.

    Flow:
      1. Search AQUAVIEW (collection=NDBC, bbox=study area, paginate_all=True)
      2. Filter results for stations with dissolved_oxygen in aquaview:variables
      3. Get item details for each → extract OPeNDAP asset URLs
      4. Return a station registry with all metadata and data URLs from AQUAVIEW

    Fallback: If AQUAVIEW is unreachable, loads from station_data/aquaview_discovery_cache.json
    (pre-built via AQUAVIEW MCP tools).
    """
    print("\n[AQUAVIEW] Searching for NDBC stations with dissolved oxygen...")
    print(f"  API: GET {AQUAVIEW_API}/search")
    print(f"  Params: collections=NDBC, bbox={BBOX}")

    n_total, features = aquaview_search(
        collection="NDBC", bbox=BBOX, limit=100, paginate_all=True
    )
    print(f"  Total NDBC stations in study area: {n_total}")

    # Filter for stations with dissolved_oxygen
    do_features = []
    for f in features:
        props = f.get("properties", {})
        avars = props.get("aquaview:variables", [])
        if "dissolved_oxygen" in avars:
            do_features.append(f)

    print(f"  Stations with dissolved_oxygen: {len(do_features)}")

    # If AQUAVIEW returned nothing, try loading from cache
    if len(do_features) == 0:
        cache_path = OUT_DIR / "aquaview_discovery_cache.json"
        if cache_path.exists():
            print(f"\n  [FALLBACK] Loading station list from {cache_path}")
            with open(cache_path) as fh:
                cache = json.load(fh)
            print(f"  Cache from {cache.get('generated', '?')}: "
                  f"{cache['total_stations']} DO stations")
            # Build station registry from cache (no AQUAVIEW get_item needed)
            stations = {}
            for s in cache["stations"]:
                sid = s["id"]
                # Build OPeNDAP URL from known NDBC THREDDS pattern
                opendap_base = (f"https://dods.ndbc.noaa.gov/thredds/dodsC/"
                                f"data/ocean/{sid}/{sid}o9999.nc")
                stations[sid] = {
                    "name": sid.upper(),
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "institution": "NDBC",
                    "aquaview_item_id": f"ndbc_{sid}",
                    "aquaview_variables": s.get("variables", []),
                    "opendap_urls": [opendap_base],
                    "aquaview_source_url": "",
                    "has_rt_ocean": s.get("has_rt_ocean", False),
                }
                print(f"    {sid.upper()}: ({s['lat']:.3f}°N, {s['lon']:.3f}°W)"
                      f"{' [rt_ocean]' if s.get('has_rt_ocean') else ''}")
            return stations, cache["total_stations"]
        else:
            print("  *** No cache file found. Check AQUAVIEW API connectivity. ***")

    # Now get full item details for each DO station → extract asset URLs
    stations = {}
    for f in do_features:
        item_id = f["id"]
        print(f"\n  [AQUAVIEW] GET /collections/NDBC/items/{item_id}")
        item = aquaview_get_item("NDBC", item_id)
        if item is None:
            continue

        props = item.get("properties", {})
        coords = item.get("geometry", {}).get("coordinates", [0, 0])
        assets = item.get("assets", {})

        # Extract station ID from AQUAVIEW item ID (e.g., "ndbc_dpha1" → "dpha1")
        sid = item_id.replace("ndbc_", "")

        # Extract OPeNDAP URLs from AQUAVIEW assets
        # AQUAVIEW provides fileServer URLs; we convert to dodsC for xarray OPeNDAP
        opendap_urls = []
        has_rt = "rt_ocean" in assets
        for asset_key, asset in assets.items():
            if asset_key.startswith("od_ocean_"):
                href = asset["href"]
                # Convert: .../thredds/fileServer/... → .../thredds/dodsC/...
                opendap_url = href.replace("/thredds/fileServer/", "/thredds/dodsC/")
                opendap_urls.append(opendap_url)

        if not opendap_urls:
            print(f"    {sid}: no ocean data assets in AQUAVIEW, skipping")
            continue

        # Parse lat/lon from AQUAVIEW geometry
        if isinstance(coords[0], (int, float)):
            lon, lat = coords[0], coords[1]
        else:
            lon, lat = coords[0][0], coords[1][0]

        stations[sid] = {
            "name": props.get("title", sid).replace("NDBC Station ", "").replace(f"{sid.upper()} - ", ""),
            "lat": lat,
            "lon": lon,
            "institution": props.get("aquaview:institution", ""),
            "aquaview_item_id": item_id,
            "aquaview_variables": props.get("aquaview:variables", []),
            "opendap_urls": sorted(opendap_urls),  # from AQUAVIEW assets
            "aquaview_source_url": props.get("aquaview:source_url", ""),
            "has_rt_ocean": has_rt,
        }
        print(f"    {sid.upper()}: {stations[sid]['name']} "
              f"({lat:.3f}°N, {lon:.3f}°W) — "
              f"{len(opendap_urls)} data files from AQUAVIEW assets"
              f"{' [rt_ocean]' if has_rt else ''}")

    return stations, n_total


def discover_wod_cruises():
    """
    Use AQUAVIEW to discover WOD cruise datasets with oxygen profiles.

    Flow:
      1. Search AQUAVIEW (collection=WOD, q=Oxygen, bbox=study area)
      2. For each matching cruise, get item → extract S3 asset URLs
      3. Return cruise registry with data URLs from AQUAVIEW
    """
    print("\n[AQUAVIEW] Searching for WOD cruise datasets with Oxygen...")
    print(f"  API: GET {AQUAVIEW_API}/search")
    print(f"  Params: collections=WOD, q=Oxygen, bbox={BBOX}")

    n_total, features = aquaview_search(collection="WOD", q="Oxygen", bbox=BBOX, limit=100)
    print(f"  WOD datasets with Oxygen in study area: {n_total}")

    # Classify by instrument type
    wod_types = {}
    total_profiles = 0
    cruises = []
    for f in features:
        props = f.get("properties", {})
        itype = props.get("aquaview:wod_instrument_type", "unknown")
        pcount = props.get("aquaview:profile_count", 0)
        wod_types[itype] = wod_types.get(itype, 0) + 1
        total_profiles += pcount

        # Get asset URLs from AQUAVIEW search results (assets are included)
        assets = f.get("assets", {})
        s3_urls = {}
        for ak, av in assets.items():
            if ak.startswith("nc_"):
                year = ak.replace("nc_", "")
                s3_urls[year] = av["href"]

        cruises.append({
            "id": f["id"],
            "type": itype,
            "profile_count": pcount,
            "title": props.get("title", ""),
            "s3_urls": s3_urls,
        })

    for itype, count in sorted(wod_types.items(), key=lambda x: -x[1]):
        label = {"pfl": "Profiling Float (Argo)", "ctd": "CTD", "osd": "Bottle",
                 "gld": "Glider", "mrb": "Moored Buoy"}.get(itype, itype)
        print(f"    {label}: {count} cruises")
    print(f"  Total oxygen profiles (first 100 cruises): {total_profiles:,}")

    return cruises, n_total, wod_types


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: NDBC Data Download (using AQUAVIEW-discovered URLs)
# ═════════════════════════════════════════════════════════════════════════════

def download_ndbc_station(station_id, config):
    """
    Download and aggregate one NDBC station to daily CSV.
    Data URLs come from AQUAVIEW assets (OPeNDAP endpoints).
    """
    print(f"\n  [{station_id.upper()}] {config['name']}...")
    print(f"    Source: AQUAVIEW item {config['aquaview_item_id']}")
    print(f"    Data files: {len(config['opendap_urls'])} NetCDF via OPeNDAP")
    frames = []

    for url in config["opendap_urls"]:
        # Extract year label from URL for logging
        fname = url.split("/")[-1]
        try:
            ds = xr.open_dataset(url, decode_times=True)
        except Exception as e:
            print(f"    {fname}: SKIP ({e})")
            continue

        # Build dataframe from available variables
        try:
            time_vals = pd.to_datetime(ds["time"].values)
        except Exception:
            print(f"    {fname}: SKIP (no valid time)")
            ds.close()
            continue

        df = pd.DataFrame({"time": time_vals})

        for nc_var, local_name in {**NDBC_VARS, **NDBC_OPTIONAL_VARS}.items():
            if nc_var in ds:
                try:
                    vals = ds[nc_var].values.flatten()[:len(df)]
                    vals = vals.astype(float)
                    # Filter fill values (NDBC uses >900 AND 99.0 as fills)
                    vals[(vals > 900) | (vals < -10)] = np.nan
                    # Apply realistic range limits per variable
                    if nc_var == "dissolved_oxygen":
                        vals[vals > 25] = np.nan   # DO cannot exceed ~20 mg/L
                    elif nc_var == "o2_saturation":
                        vals[vals > 200] = np.nan  # O2 sat rarely exceeds 150%
                    elif nc_var == "water_temperature":
                        vals[vals > 50] = np.nan
                    elif nc_var == "salinity":
                        vals[vals > 45] = np.nan
                    df[local_name] = vals
                except Exception:
                    pass

        ds.close()

        if "o2" not in df.columns or df["o2"].notna().sum() == 0:
            print(f"    {fname}: SKIP (no DO data)")
            continue

        # Daily aggregation
        df["date"] = df["time"].dt.date
        agg_dict = {}
        for col in df.columns:
            if col in ("time", "date"):
                continue
            agg_dict[col] = "mean"
        agg_dict["time"] = "count"  # observation count

        daily = df.groupby("date").agg(agg_dict).rename(columns={"time": "n"})
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "dt"
        frames.append(daily)

        n_obs = len(df)
        n_days = len(daily)
        do_min = df["o2"].min()
        do_max = df["o2"].max()
        print(f"    {fname}: {n_obs:,} obs → {n_days} days, "
              f"DO {do_min:.2f}–{do_max:.2f} mg/L")

    if not frames:
        print(f"    *** NO DATA for {station_id.upper()} ***")
        return None

    result = pd.concat(frames).sort_index()

    # Quality check: reject stations with insufficient valid DO data
    if "o2" in result.columns:
        valid_do = result["o2"].dropna()
        if len(valid_do) < 30:
            print(f"    *** REJECTED {station_id.upper()}: only {len(valid_do)} valid DO days (need ≥30) ***")
            return None
        if valid_do.std() < 0.01:
            print(f"    *** REJECTED {station_id.upper()}: DO has no variance (likely all fill values) ***")
            return None
        pct_valid = len(valid_do) / len(result) * 100
        if pct_valid < 10:
            print(f"    *** REJECTED {station_id.upper()}: only {pct_valid:.1f}% of days have valid DO ***")
            return None

    # Add hypoxia flag
    if "o2" in result.columns:
        result["hyp"] = (result["o2"] < HYPOXIA_THRESHOLD).astype(int)
    else:
        result["hyp"] = 0

    # Save CSV
    csv_path = OUT_DIR / f"{station_id}_daily.csv"
    result.to_csv(csv_path)

    return result


def download_all_ndbc(stations):
    """Download all AQUAVIEW-discovered NDBC stations."""
    print("\n" + "=" * 70)
    print("NDBC DATA DOWNLOAD")
    print("All data URLs discovered via AQUAVIEW STAC API assets")
    print("=" * 70)

    station_data = {}
    for sid, config in stations.items():
        df = download_ndbc_station(sid, config)
        if df is not None:
            station_data[sid] = df

    return station_data


# ═════════════════════════════════════════════════════════════════════════════
# PART 3: WOD Argo Float Data (cross-source, URLs from AQUAVIEW)
# ═════════════════════════════════════════════════════════════════════════════

def download_wod_profiles(wod_cruises):
    """
    Download WOD profile data using S3 URLs discovered through AQUAVIEW.
    Falls back gracefully if files are too large or unavailable.
    """
    print("\n" + "=" * 70)
    print("WOD DATA DOWNLOAD (cross-source)")
    print("All data URLs discovered via AQUAVIEW STAC API assets")
    print("=" * 70)

    # Collect unique S3 URLs from AQUAVIEW-discovered cruises
    # Group by (instrument_type, year) → S3 URL
    s3_urls_by_year = {}
    for cruise in wod_cruises:
        for year_str, url in cruise.get("s3_urls", {}).items():
            try:
                year = int(year_str)
                if year in WOD_YEARS:
                    key = (cruise["type"], year)
                    s3_urls_by_year[key] = url
            except ValueError:
                continue

    # Prefer CTD, then PFL (Argo)
    urls_to_try = []
    for year in WOD_YEARS:
        if ("ctd", year) in s3_urls_by_year:
            urls_to_try.append(("ctd", year, s3_urls_by_year[("ctd", year)]))
        elif ("pfl", year) in s3_urls_by_year:
            urls_to_try.append(("pfl", year, s3_urls_by_year[("pfl", year)]))
        else:
            # Fallback: construct URL from known pattern (CTD first, then PFL)
            urls_to_try.append(("ctd", year,
                f"https://noaa-wod-pds.s3.amazonaws.com/{year}/wod_ctd_{year}.nc"))

    all_profiles = []

    for itype, year, url in urls_to_try:
        print(f"\n  [AQUAVIEW-sourced] WOD {itype.upper()} {year}: {url}")
        try:
            ds = xr.open_dataset(url, decode_times=True)

            # Filter to bounding box
            lat = ds["lat"].values
            lon = ds["lon"].values
            mask = ((lat >= BBOX["south"]) & (lat <= BBOX["north"]) &
                    (lon >= BBOX["west"]) & (lon <= BBOX["east"]))

            n_in_bbox = mask.sum()
            if n_in_bbox == 0:
                print(f"    No profiles in study bbox for {year}")
                ds.close()
                continue

            print(f"    Found {n_in_bbox} profiles in study area (of {len(lat)} total)")

            # Extract profile-level summary data
            cast_indices = np.where(mask)[0]
            for idx in cast_indices[:50]:  # limit to 50 profiles per year
                try:
                    profile = {
                        "year": year,
                        "source_type": itype,
                        "lat": float(lat[idx]),
                        "lon": float(lon[idx]),
                    }
                    if "time" in ds:
                        profile["time"] = str(pd.Timestamp(ds["time"].values[idx]))
                    if "Oxygen" in ds:
                        o2_vals = ds["Oxygen"].values
                        # WOD uses ragged arrays; get profile oxygen
                        if "Oxygen_row_size" in ds:
                            row_sizes = ds["Oxygen_row_size"].values
                            start = int(row_sizes[:idx].sum())
                            count = int(row_sizes[idx])
                            o2_profile = o2_vals[start:start+count]
                            o2_profile = o2_profile[o2_profile < 900]  # filter fills
                            if len(o2_profile) > 0:
                                profile["o2_surface"] = float(o2_profile[0])
                                profile["o2_min"] = float(np.nanmin(o2_profile))
                                profile["o2_mean"] = float(np.nanmean(o2_profile))
                                profile["o2_n_depths"] = len(o2_profile)
                        elif len(o2_vals.shape) == 2:
                            o2_profile = o2_vals[idx, :]
                            o2_profile = o2_profile[~np.isnan(o2_profile)]
                            o2_profile = o2_profile[o2_profile < 900]
                            if len(o2_profile) > 0:
                                profile["o2_surface"] = float(o2_profile[0])
                                profile["o2_min"] = float(np.nanmin(o2_profile))
                                profile["o2_mean"] = float(np.nanmean(o2_profile))
                                profile["o2_n_depths"] = len(o2_profile)

                    if "Temperature" in ds:
                        t_vals = ds["Temperature"].values
                        if "Temperature_row_size" in ds:
                            row_sizes = ds["Temperature_row_size"].values
                            start = int(row_sizes[:idx].sum())
                            count = int(row_sizes[idx])
                            t_profile = t_vals[start:start+count]
                            t_profile = t_profile[t_profile < 900]
                            if len(t_profile) > 0:
                                profile["temp_surface"] = float(t_profile[0])

                    all_profiles.append(profile)
                except Exception:
                    continue

            ds.close()

        except Exception as e:
            print(f"    WOD {itype.upper()} {year} failed: {e}")
            # Try PFL (Argo) as fallback if CTD failed
            if itype == "ctd":
                pfl_url = f"https://noaa-wod-pds.s3.amazonaws.com/{year}/wod_pfl_{year}.nc"
                print(f"    Trying PFL fallback: {pfl_url}")
                try:
                    ds = xr.open_dataset(pfl_url, decode_times=True)
                    lat = ds["lat"].values
                    lon = ds["lon"].values
                    mask = ((lat >= BBOX["south"]) & (lat <= BBOX["north"]) &
                            (lon >= BBOX["west"]) & (lon <= BBOX["east"]))
                    n_in_bbox = mask.sum()
                    print(f"    PFL: {n_in_bbox} profiles in study area (of {len(lat)} total)")
                    if n_in_bbox > 0:
                        cast_indices = np.where(mask)[0]
                        for idx in cast_indices[:20]:
                            try:
                                profile = {
                                    "year": year, "source_type": "pfl",
                                    "lat": float(lat[idx]), "lon": float(lon[idx]),
                                }
                                if "time" in ds:
                                    profile["time"] = str(pd.Timestamp(ds["time"].values[idx]))
                                all_profiles.append(profile)
                            except Exception:
                                continue
                    ds.close()
                except Exception as e2:
                    print(f"    PFL fallback also failed: {e2}")
                    print(f"    (WOD S3 files may be too large for direct access)")

    if all_profiles:
        wod_df = pd.DataFrame(all_profiles)
        csv_path = OUT_DIR / "wod_profiles.csv"
        wod_df.to_csv(csv_path, index=False)
        print(f"\n  Saved {len(wod_df)} WOD profiles to {csv_path}")
        return wod_df
    else:
        print("\n  No WOD profiles extracted.")
        print("  NOTE: WOD annual files (e.g. wod_pfl_2022.nc) are multi-GB global datasets.")
        print("  AQUAVIEW correctly provides these S3 URLs, but they're too large for")
        print("  direct HTTP/xarray access. Production use would require s3fs with")
        print("  byte-range requests, or NCEI's per-cruise download service.")
        print("  AQUAVIEW discovery stats from WOD are still available for the blog.")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PART 3b: AQUAVIEW → CoastWatch Satellite Data (ERDDAP)
# ═════════════════════════════════════════════════════════════════════════════

def discover_coastwatch_datasets():
    """
    Use AQUAVIEW to discover CoastWatch satellite products in the study area.

    Returns ERDDAP asset URLs for:
      - Daily satellite SST (VIIRS, 4km)
      - Daily chlorophyll-a (Sentinel-3 OLCI, 4km)
      - Multi-param eddy index (SST, SSS, chla, SSH, EKE; 0.25°)

    These are gridded satellite products hosted on CoastWatch ERDDAP.
    AQUAVIEW provides the ERDDAP base URLs in each item's assets field,
    and we construct subsetting queries to extract data for our study area.
    """
    print("\n" + "-" * 60)
    print("AQUAVIEW → CoastWatch Satellite Discovery")
    print("-" * 60)

    # ── Step 1: Search AQUAVIEW for CoastWatch SST datasets ─────────────
    bbox_str = f"{BBOX['west']},{BBOX['south']},{BBOX['east']},{BBOX['north']}"

    print(f"\n  AQUAVIEW search: collections=COASTWATCH, q=sea surface temperature, bbox={bbox_str}")
    search_url = f"{AQUAVIEW_API}/search"
    resp = requests.get(search_url, params={
        "collections": "COASTWATCH",
        "q": "sea surface temperature",
        "bbox": bbox_str,
        "limit": 20,
    }, timeout=30)
    resp.raise_for_status()
    sst_results = resp.json()
    n_sst = sst_results.get("numberMatched", sst_results.get("context", {}).get("matched", 0))
    print(f"    → {n_sst} SST datasets found in CoastWatch")

    # ── Step 2: Search for chlorophyll datasets ─────────────────────────
    print(f"\n  AQUAVIEW search: collections=COASTWATCH, q=chlor, bbox={bbox_str}")
    resp = requests.get(search_url, params={
        "collections": "COASTWATCH",
        "q": "chlor",
        "bbox": bbox_str,
        "limit": 20,
    }, timeout=30)
    resp.raise_for_status()
    chl_results = resp.json()
    n_chl = chl_results.get("numberMatched", chl_results.get("context", {}).get("matched", 0))
    print(f"    → {n_chl} chlorophyll datasets found in CoastWatch")

    # ── Step 3: Get specific items via AQUAVIEW for asset URLs ──────────
    datasets = {}

    # Multi-param eddy index: best single product (SST+SSS+chla+SSH+EKE)
    print(f"\n  AQUAVIEW item lookup: COASTWATCH/noaacweddymesiplusdaily")
    item_url = f"{AQUAVIEW_API}/collections/COASTWATCH/items/noaacweddymesiplusdaily"
    resp = requests.get(item_url, timeout=30)
    if resp.status_code == 200:
        item = resp.json()
        csv_url = item.get("assets", {}).get("csv", {}).get("href")
        if csv_url:
            datasets["eddy_multi"] = {
                "aquaview_id": "noaacweddymesiplusdaily",
                "title": item["properties"]["title"],
                "erddap_csv": csv_url,
                "variables": ["sst", "chla", "ssh", "sss", "eke"],
                "resolution": "0.25°",
                "start": item["properties"].get("start_datetime", ""),
                "end": item["properties"].get("end_datetime", ""),
            }
            print(f"    → Asset URL: {csv_url}")
            print(f"    → Variables: SST, chlorophyll-a, SSH, SSS, eddy kinetic energy")

    # Daily SST (VIIRS, 4km) as backup/higher-res SST
    print(f"\n  AQUAVIEW item lookup: COASTWATCH/noaacwSNPPACSPOSSTL3GCDaily")
    item_url = f"{AQUAVIEW_API}/collections/COASTWATCH/items/noaacwSNPPACSPOSSTL3GCDaily"
    resp = requests.get(item_url, timeout=30)
    if resp.status_code == 200:
        item = resp.json()
        csv_url = item.get("assets", {}).get("csv", {}).get("href")
        if csv_url:
            datasets["sst_viirs"] = {
                "aquaview_id": "noaacwSNPPACSPOSSTL3GCDaily",
                "title": item["properties"]["title"],
                "erddap_csv": csv_url,
                "variables": ["sea_surface_temperature"],
                "resolution": "4km",
                "start": item["properties"].get("start_datetime", ""),
                "end": item["properties"].get("end_datetime", ""),
            }
            print(f"    → Asset URL: {csv_url}")

    # Daily chlorophyll-a (Sentinel-3B OLCI, global 4km)
    print(f"\n  AQUAVIEW item lookup: COASTWATCH/noaacwS3BOLCIchlaDaily")
    item_url = f"{AQUAVIEW_API}/collections/COASTWATCH/items/noaacwS3BOLCIchlaDaily"
    resp = requests.get(item_url, timeout=30)
    if resp.status_code == 200:
        item = resp.json()
        csv_url = item.get("assets", {}).get("csv", {}).get("href")
        if csv_url:
            datasets["chla_olci"] = {
                "aquaview_id": "noaacwS3BOLCIchlaDaily",
                "title": item["properties"]["title"],
                "erddap_csv": csv_url,
                "variables": ["chlor_a"],
                "resolution": "4km",
                "start": item["properties"].get("start_datetime", ""),
                "end": item["properties"].get("end_datetime", ""),
            }
            print(f"    → Asset URL: {csv_url}")

    print(f"\n  CoastWatch discovery complete: {len(datasets)} datasets with ERDDAP asset URLs")

    # Save satellite registry
    sat_path = OUT_DIR / "satellite_registry.json"
    with open(sat_path, "w") as f:
        json.dump(datasets, f, indent=2)
    print(f"  Satellite registry saved to: {sat_path}")

    return datasets, n_sst + n_chl


def download_coastwatch_satellite(datasets):
    """
    Download satellite data from CoastWatch ERDDAP using OPeNDAP (preferred) or CSV.

    Uses OPeNDAP + xarray for more reliable downloads through proxies; falls back
    to CSV with retries. Requests are chunked by 6 months to avoid timeouts.
    """
    print("\n" + "-" * 60)
    print("Downloading CoastWatch Satellite Data (OPeNDAP/ERDDAP)")
    print("-" * 60)

    if not datasets:
        print("  No CoastWatch datasets discovered.")
        return None

    all_satellite = {}
    MAX_RETRIES = 3
    RETRY_DELAYS = (5, 15, 45)  # seconds

    def _erddap_base(dataset_id, ext="dods"):
        """Base URL for CoastWatch ERDDAP (OPeNDAP or CSV)."""
        return f"{COASTWATCH_ERDDAP_BASE}/{dataset_id}.{ext}"

    def _fetch_with_retries(fetcher, label):
        """Try fetcher up to MAX_RETRIES with exponential backoff."""
        for attempt in range(MAX_RETRIES):
            try:
                result = fetcher()
                if result is not None:
                    return result
            except Exception as e:
                print(f"    Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
        return None

    # ── Eddy multi-param (0.25°, time-chunked) ─────────────────────────────
    lat_s = round(BBOX["south"] * 4) / 4
    lat_n = round(BBOX["north"] * 4) / 4
    lon_w = round(BBOX["west"] * 4) / 4
    lon_e = round(BBOX["east"] * 4) / 4

    if "eddy_multi" in datasets:
        ds_cfg = datasets["eddy_multi"]
        print(f"\n  Eddy multi-param: {ds_cfg['title'][:55]}...")
        print(f"    Method: OPeNDAP (6‑month chunks) + retries")
        print(f"    Bbox: lat [{lat_s},{lat_n}], lon [{lon_w},{lon_e}]")

        chunks = []
        t_end_str = ds_cfg.get("end", "2026-01-31T00:00:00Z")[:10]
        done = False
        # 6-month chunks: May–Oct, Nov–Apr
        for year in range(2020, 2027):
            if done:
                break
            for start_mo, end_mo, end_yr in [(5, 11, 0), (11, 5, 1)]:
                t_start = f"{year}-{start_mo:02d}-01T00:00:00Z"
                t_stop = f"{year + end_yr}-{end_mo:02d}-01T00:00:00Z"
                if t_stop[:10] > t_end_str:
                    done = True
                    break
                label = f"{t_start[:7]}..{t_stop[:7]}"
                print(f"    Chunk {label}...", end=" ", flush=True)

                def _fetch_chunk():
                    dims = f"[({t_start}):1:({t_stop})][({lat_s}):1:({lat_n})][({lon_w}):1:({lon_e})]"
                    q = ",".join(v + dims for v in ["sst", "chla", "ssh", "sss", "eke"])
                    # Try OPeNDAP first (often works better through proxies)
                    for ext in ("dods", "csv"):
                        url = _erddap_base("noaacweddymesiplusdaily", ext) + "?" + quote(q, safe=",")
                        try:
                            if ext == "dods":
                                nx = xr.open_dataset(url, decode_times=True)
                                df = nx.to_dataframe().reset_index()
                                nx.close()
                            else:
                                r = requests.get(url, timeout=180)
                                if r.status_code != 200:
                                    continue
                                from io import StringIO
                                df = pd.read_csv(StringIO(r.text), skiprows=[1])
                            return df if df is not None and len(df) > 0 else None
                        except Exception:
                            continue
                    return None

                chunk_df = _fetch_with_retries(_fetch_chunk, label)
                if chunk_df is not None and len(chunk_df) > 0:
                    chunks.append(chunk_df)
                    print(f"{len(chunk_df)} rows")
                else:
                    print("failed")

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df["date"] = df["time"].dt.date
            daily = df.groupby("date").agg({"sst": "mean", "chla": "mean", "ssh": "mean", "sss": "mean", "eke": "mean"}).reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            daily.columns = ["dt", "sat_sst", "sat_chla", "sat_ssh", "sat_sss", "sat_eke"]
            all_satellite["eddy_multi"] = daily
            csv_path = OUT_DIR / "satellite_eddy_multi.csv"
            daily.to_csv(csv_path, index=False)
            print(f"    → {len(daily)} daily averages saved to {csv_path}")

    # ── VIIRS SST & OLCI chlorophyll: use OPeNDAP, single small bbox ───────
    center_lat, center_lon = 30.3, -88.1
    lat_lo, lat_hi = center_lat - 0.5, center_lat + 0.5
    lon_lo, lon_hi = center_lon - 0.5, center_lon + 0.5

    if "sst_viirs" in datasets:
        ds_cfg = datasets["sst_viirs"]
        end = ds_cfg.get("end", "2026-01-13T12:00:00Z")
        print(f"\n  VIIRS SST: {ds_cfg['title'][:55]}...")
        print(f"    Method: OPeNDAP, point extract, 6‑month chunks")

        def _fetch_sst():
            q = f"sea_surface_temperature[(2020-01-15T12:00:00Z):1:({end})][(0.0)][({lat_lo}):4:({lat_hi})][({lon_lo}):4:({lon_hi})]"
            for ext in ("dods", "csv"):
                url = _erddap_base("noaacwSNPPACSPOSSTL3GCDaily", ext) + "?" + quote(q, safe=",")
                try:
                    if ext == "dods":
                        nx = xr.open_dataset(url, decode_times=True)
                        df = nx.to_dataframe().reset_index()
                        nx.close()
                    else:
                        r = requests.get(url, timeout=180)
                        if r.status_code != 200:
                            continue
                        from io import StringIO
                        df = pd.read_csv(StringIO(r.text), skiprows=[1])
                    return df if df is not None and len(df) > 0 else None
                except Exception:
                    continue
            return None

        df = _fetch_with_retries(_fetch_sst, "sst_viirs")
        if df is not None and len(df) > 0:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            daily = df.groupby(df["time"].dt.date)["sea_surface_temperature"].mean().reset_index()
            daily.columns = ["dt", "sat_sst_hires"]
            daily["dt"] = pd.to_datetime(daily["dt"])
            all_satellite["sst_viirs"] = daily
            daily.to_csv(OUT_DIR / "satellite_sst_viirs.csv", index=False)
            print(f"    → {len(daily)} daily averages")

    if "chla_olci" in datasets:
        ds_cfg = datasets["chla_olci"]
        end = ds_cfg.get("end", "2026-02-07T12:00:00Z")
        print(f"\n  OLCI chlorophyll: {ds_cfg['title'][:55]}...")
        print(f"    Method: OPeNDAP, point extract, 6‑month chunks")

        def _fetch_chla():
            q = f"chlor_a[(2020-01-01T12:00:00Z):1:({end})][(0.0)][({lat_lo}):4:({lat_hi})][({lon_lo}):4:({lon_hi})]"
            for ext in ("dods", "csv"):
                url = _erddap_base("noaacwS3BOLCIchlaDaily", ext) + "?" + quote(q, safe=",")
                try:
                    if ext == "dods":
                        nx = xr.open_dataset(url, decode_times=True)
                        df = nx.to_dataframe().reset_index()
                        nx.close()
                    else:
                        r = requests.get(url, timeout=180)
                        if r.status_code != 200:
                            continue
                        from io import StringIO
                        df = pd.read_csv(StringIO(r.text), skiprows=[1])
                    return df if df is not None and len(df) > 0 else None
                except Exception:
                    continue
            return None

        df = _fetch_with_retries(_fetch_chla, "chla_olci")
        if df is not None and len(df) > 0:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            daily = df.groupby(df["time"].dt.date)["chlor_a"].mean().reset_index()
            daily.columns = ["dt", "sat_chla_hires"]
            daily["dt"] = pd.to_datetime(daily["dt"])
            all_satellite["chla_olci"] = daily
            daily.to_csv(OUT_DIR / "satellite_chla_olci.csv", index=False)
            print(f"    → {len(daily)} daily averages")

    if all_satellite:
        # Merge all satellite products into one daily dataframe
        merged = None
        for key, df in all_satellite.items():
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="dt", how="outer")
        merged = merged.sort_values("dt").reset_index(drop=True)

        csv_path = OUT_DIR / "satellite_merged.csv"
        merged.to_csv(csv_path, index=False)
        print(f"\n  Merged satellite data: {len(merged)} days, {len(merged.columns)-1} variables")
        print(f"  Saved to: {csv_path}")
        return merged
    else:
        print("\n  No satellite data downloaded.")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PART 4: Cross-Source Feature Engineering + Summary
# ═════════════════════════════════════════════════════════════════════════════

def compute_wod_features(station_data, stations, wod_df, target_station="dpha1"):
    """
    Create daily cross-source features from WOD profiles for the target station.

    For each day in the target station's time series, find the most recent
    WOD profile within MAX_DIST_KM and MAX_AGE_DAYS, and use its oxygen,
    temperature, etc. as "offshore context" features.

    Scientific rationale: Offshore deepwater oxygen conditions (measured by
    Argo floats) can precede coastal hypoxia events by days to weeks as
    water masses advect inshore. These features capture basin-scale signals
    that fixed coastal stations cannot.
    """
    MAX_DIST_KM = 200
    MAX_AGE_DAYS = 30

    if wod_df is None or len(wod_df) == 0:
        print("\n  No WOD data available for cross-source features.")
        return None

    if target_station not in station_data:
        return None

    target = station_data[target_station]
    target_lat = stations[target_station]["lat"]
    target_lon = stations[target_station]["lon"]

    print(f"\n  Computing WOD cross-source features for {target_station.upper()}...")

    # Parse WOD times
    wod = wod_df.copy()
    if "time" in wod.columns:
        wod["dt"] = pd.to_datetime(wod["time"], errors="coerce")
    else:
        wod["dt"] = pd.to_datetime(wod["year"].astype(str) + "-07-01")

    # Filter WOD to profiles with oxygen data
    if "o2_surface" in wod.columns:
        wod_with_o2 = wod.dropna(subset=["o2_surface"])
    else:
        print("    No oxygen values in WOD profiles; using location features only.")
        wod_with_o2 = wod

    # Compute distance from target station (approximate)
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon/2)**2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    wod_with_o2 = wod_with_o2.copy()
    wod_with_o2["dist_km"] = wod_with_o2.apply(
        lambda r: haversine_km(target_lat, target_lon, r["lat"], r["lon"]), axis=1
    )
    wod_nearby = wod_with_o2[wod_with_o2["dist_km"] <= MAX_DIST_KM].copy()

    if len(wod_nearby) == 0:
        print(f"    No WOD profiles within {MAX_DIST_KM} km of {target_station.upper()}")
        return None

    print(f"    {len(wod_nearby)} WOD profiles within {MAX_DIST_KM} km")
    wod_nearby = wod_nearby.sort_values("dt")

    # For each day, find most recent WOD profile within MAX_AGE_DAYS
    features = []
    for dt in target.index:
        age = (dt - wod_nearby["dt"]).dt.days
        valid = wod_nearby[(age >= 0) & (age <= MAX_AGE_DAYS)]
        if len(valid) == 0:
            features.append({
                "dt": dt,
                "wod_o2_surface": np.nan,
                "wod_o2_min": np.nan,
                "wod_o2_mean": np.nan,
                "wod_temp_surface": np.nan,
                "wod_dist_km": np.nan,
                "wod_age_days": np.nan,
                "wod_n_profiles": 0,
            })
        else:
            latest = valid.iloc[-1]
            feat = {
                "dt": dt,
                "wod_dist_km": latest.get("dist_km", np.nan),
                "wod_age_days": (dt - latest["dt"]).days,
                "wod_n_profiles": len(valid),
            }
            for col in ["o2_surface", "o2_min", "o2_mean", "temp_surface"]:
                feat[f"wod_{col}"] = latest.get(col, np.nan)

            for col in ["o2_surface", "o2_min"]:
                if col in valid.columns:
                    feat[f"wod_{col}_mean30d"] = valid[col].mean()

            features.append(feat)

    wod_features = pd.DataFrame(features).set_index("dt")
    coverage = wod_features["wod_n_profiles"].gt(0).mean() * 100
    print(f"    WOD feature coverage: {coverage:.1f}% of target days")

    csv_path = OUT_DIR / "wod_features.csv"
    wod_features.to_csv(csv_path)
    print(f"    Saved to {csv_path}")

    return wod_features


def compute_summary(station_data, stations, discovery_stats, wod_df=None):
    """Compute comprehensive summary for blog post."""
    print("\n" + "=" * 70)
    print("SUMMARY FOR BLOG POST")
    print("=" * 70)

    summary = {"discovery": discovery_stats, "stations": {}, "network": {}}

    # Per-station stats
    print(f"\n{'Station':<8s} {'Name':<35s} {'Days':>6s} {'DO range':>14s} "
          f"{'Hyp days':>9s} {'Events':>7s} {'Years':>12s}")
    print("-" * 100)

    total_days = 0
    total_hyp_days = 0
    total_events = 0
    all_do_values = []
    stations_with_hypoxia = []

    for sid, df in sorted(station_data.items()):
        config = stations[sid]
        n_days = len(df)
        total_days += n_days

        # DO stats
        if "o2" in df.columns:
            do_min = df["o2"].min()
            do_max = df["o2"].max()
            do_mean = df["o2"].mean()
            all_do_values.extend(df["o2"].dropna().tolist())
        else:
            do_min = do_max = do_mean = float("nan")

        # Hypoxia
        hyp_days = int(df["hyp"].sum()) if "hyp" in df.columns else 0
        total_hyp_days += hyp_days

        # Count events
        if "hyp" in df.columns:
            onset = ((df["hyp"] == 1) &
                     (df["hyp"].shift(1, fill_value=0) == 0))
            n_events = int(onset.sum())
        else:
            n_events = 0
        total_events += n_events

        if hyp_days > 0:
            stations_with_hypoxia.append(sid.upper())

        date_min = df.index.min().strftime("%Y")
        date_max = df.index.max().strftime("%Y")

        print(f"{sid.upper():<8s} {config['name']:<35s} {n_days:>6,d} "
              f"{do_min:>6.2f}–{do_max:<5.2f} {hyp_days:>9d} {n_events:>7d} "
              f"{date_min}–{date_max}")

        summary["stations"][sid] = {
            "name": config["name"],
            "lat": config["lat"], "lon": config["lon"],
            "aquaview_item_id": config["aquaview_item_id"],
            "n_data_files": len(config["opendap_urls"]),
            "n_days": n_days,
            "do_min": round(do_min, 3),
            "do_max": round(do_max, 3),
            "do_mean": round(do_mean, 3),
            "hyp_days": hyp_days,
            "n_events": n_events,
            "date_range": f"{date_min}–{date_max}",
        }

    print("-" * 100)
    print(f"{'TOTAL':<8s} {'':35s} {total_days:>6,d} {'':>14s} "
          f"{total_hyp_days:>9d} {total_events:>7d}")

    summary["network"] = {
        "n_stations": len(station_data),
        "total_station_days": total_days,
        "total_hyp_days": total_hyp_days,
        "total_events": total_events,
        "stations_with_hypoxia": stations_with_hypoxia,
        "do_range": [round(min(all_do_values), 3), round(max(all_do_values), 3)]
                     if all_do_values else [None, None],
        "do_mean": round(np.mean(all_do_values), 3) if all_do_values else None,
    }

    # Cross-station correlations (monthly DO)
    print("\n  Cross-station monthly DO correlations with DPHA1:")
    if "dpha1" in station_data and "o2" in station_data["dpha1"].columns:
        dpha1_monthly = station_data["dpha1"]["o2"].resample("ME").mean()
        correlations = {}
        for sid, df in station_data.items():
            if sid == "dpha1" or "o2" not in df.columns:
                continue
            other_monthly = df["o2"].resample("ME").mean()
            merged = pd.concat([dpha1_monthly, other_monthly], axis=1,
                              keys=["dpha1", sid]).dropna()
            if len(merged) >= 6:
                r = merged["dpha1"].corr(merged[sid])
                correlations[sid] = round(r, 3)
                print(f"    DPHA1 vs {sid.upper()}: r = {r:.3f} (n={len(merged)} months)")
        summary["network"]["do_correlations_with_dpha1"] = correlations

    # WOD summary
    if wod_df is not None and len(wod_df) > 0:
        summary["wod"] = {
            "n_profiles": len(wod_df),
            "years": sorted(wod_df["year"].unique().tolist()),
        }
        if "o2_surface" in wod_df.columns:
            o2_surf = wod_df["o2_surface"].dropna()
            if len(o2_surf) > 0:
                summary["wod"]["o2_surface_mean"] = round(o2_surf.mean(), 3)
                summary["wod"]["o2_surface_min"] = round(o2_surf.min(), 3)
        print(f"\n  WOD profiles extracted: {len(wod_df)}")

    # Blog-ready stats
    ds = discovery_stats
    print("\n" + "=" * 70)
    print("BLOG-READY STATISTICS")
    print("=" * 70)
    print(f"""
  Data Discovery (via AQUAVIEW STAC API):
    • {ds.get('ndbc_total', '?')} NDBC datasets in study area
    • {ds.get('ndbc_with_do', '?')} with dissolved oxygen sensors
    • {ds.get('wod_total', '?')} WOD cruise datasets with oxygen profiles
    • {ds.get('total_datasets', '?')} total datasets across {len(ds.get('sources', []))} sources

  Data Access (via AQUAVIEW asset URLs):
    • {sum(len(s['opendap_urls']) for s in stations.values())} NDBC NetCDF files accessed via OPeNDAP
    • WOD profiles accessed via NOAA S3 (URLs from AQUAVIEW assets)

  Station Network:
    • {len(station_data)} fixed monitoring stations
    • {total_days:,} total station-days of observations
    • {len(stations_with_hypoxia)} station(s) experienced hypoxia: {', '.join(stations_with_hypoxia)}
    • {total_events} hypoxia onset events detected
    • {total_hyp_days} total hypoxic days

  Environmental Gradient:
    • Longitude span: {min(s['lon'] for s in stations.values()):.3f}°W to {max(s['lon'] for s in stations.values()):.3f}°W
    • Latitude span: {min(s['lat'] for s in stations.values()):.3f}°N to {max(s['lat'] for s in stations.values()):.3f}°N
    • Environments: coastal bay, estuarine, barrier island, open shelf
""")

    # Save summary JSON
    json_path = OUT_DIR / "extraction_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Full summary saved to: {json_path}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MULTI-SOURCE HYPOXIA DATA EXTRACTION                          ║")
    print("║  Powered by AQUAVIEW STAC API (aquaview.ai)                    ║")
    print("║  Discovery + Access: All via AQUAVIEW                          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\nAQUAVIEW API: {AQUAVIEW_API}")
    print(f"API Docs: {AQUAVIEW_API}/api.html")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # ── Step 1: AQUAVIEW Discovery ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: AQUAVIEW DATA DISCOVERY")
    print("=" * 70)

    stations, n_ndbc = discover_ndbc_stations()
    wod_cruises, n_wod, wod_types = discover_wod_cruises()

    discovery_stats = {
        "api_base": AQUAVIEW_API,
        "bbox": BBOX,
        "ndbc_total": n_ndbc,
        "ndbc_with_do": len(stations),
        "wod_total": n_wod,
        "wod_types": wod_types,
        "total_datasets": n_ndbc + n_wod,
        "sources": ["NDBC", "WOD"],
    }

    print(f"\n  AQUAVIEW unified discovery: {n_ndbc + n_wod} datasets across 2 sources")
    print(f"  Selected {len(stations)} NDBC stations + {len(wod_cruises)} WOD cruises")

    if not stations:
        print("\n*** No NDBC stations found via AQUAVIEW. Check API connectivity. ***")
        sys.exit(1)

    # ── Step 2: Download NDBC stations (URLs from AQUAVIEW assets) ──────────
    station_data = download_all_ndbc(stations)

    if not station_data:
        print("\n*** No station data downloaded. Check network connectivity. ***")
        sys.exit(1)

    # ── Step 3: Download WOD cross-source data (URLs from AQUAVIEW assets) ──
    wod_df = download_wod_profiles(wod_cruises)

    # ── Step 3b: AQUAVIEW → CoastWatch Satellite Data ─────────────────────
    print("\n" + "=" * 70)
    print("STEP 3b: COASTWATCH SATELLITE DATA (via AQUAVIEW)")
    print("=" * 70)

    sat_datasets, n_sat_discovered = discover_coastwatch_datasets()
    sat_df = download_coastwatch_satellite(sat_datasets)

    discovery_stats["coastwatch_datasets_discovered"] = n_sat_discovered
    discovery_stats["coastwatch_downloaded"] = len(sat_datasets) if sat_datasets else 0
    discovery_stats["sources"] = ["NDBC", "WOD", "COASTWATCH"]

    # ── Step 4: Compute cross-source features (WOD → DPHA1) ────────────────
    wod_features = compute_wod_features(station_data, stations, wod_df,
                                        target_station="dpha1")

    # ── Step 5: Summary ────────────────────────────────────────────────────
    summary = compute_summary(station_data, stations, discovery_stats, wod_df)

    # ── Save station registry (for pipeline script) ─────────────────────────
    registry = {}
    for sid, config in stations.items():
        if sid in station_data:
            registry[sid] = {
                "name": config["name"],
                "lat": config["lat"],
                "lon": config["lon"],
                "aquaview_item_id": config["aquaview_item_id"],
                "opendap_urls": config["opendap_urls"],
            }
    registry_path = OUT_DIR / "station_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\n  Station registry saved to: {registry_path}")

    print("\n" + "=" * 70)
    print("DONE — Files saved to:", OUT_DIR.resolve())
    print("=" * 70)
    for f in sorted(OUT_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<30s} {size_kb:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
