# Predicting Coastal Hypoxia Across 12 Stations and 3 Data Systems With One Catalog

What does it actually look like to build a multi-source ocean analysis from scratch?

Hypoxia, dissolved oxygen dropping below 2 mg/L, kills coastal ecosystems fast. Fish flee, shellfish die, and the damage can cascade for months. Monitoring networks catch it after it starts. The question we wanted to answer: can we predict when a new hypoxia event will *begin*, days before it happens, using data from a network of stations, satellite products, and offshore profiles?

This post walks through building that system for the northern Gulf of Mexico. Along the way, we used [AQUAVIEW](https://aquaview.ai) as the data discovery and access layer, and we'll be honest about what that did and didn't change.

## The Hard Problem: Onset, Not Persistence

Most hypoxia forecasting predicts whether DO will be low tomorrow given that it's already low today. That's persistence forecasting, and DO autocorrelation does the heavy lifting.

The harder question is: when will a *new* event begin? Predicting the transition from normal oxygen to hypoxic conditions is what coastal managers actually need, and it's what we built.

## Finding the Data

The first step in any ocean analysis is figuring out what data exists, where it lives, and how to get it. This is usually the most tedious part. You're browsing NOAA pages, clicking through THREDDS directories, guessing at URL patterns, and hoping variable names are consistent across stations.

We used AQUAVIEW's STAC API to shortcut this. Three sets of API calls did the core discovery work:

```
GET /search?collections=NDBC&bbox=-91,28,-85,31
→ 72 NDBC stations in the study area

GET /search?collections=WOD&q=Oxygen&bbox=-91,28,-85,31
→ 1,770 WOD cruise datasets with oxygen profiles

GET /search?collections=COASTWATCH&q=sea surface temperature&bbox=-91,28,-85,31
→ 25 satellite SST datasets
GET /search?collections=COASTWATCH&q=chlor&bbox=-91,28,-85,31
→ 8 satellite chlorophyll datasets
```

Three different data systems. Three different hosting infrastructures. Same API.

From the NDBC results, we filtered for stations where `aquaview:variables` includes `dissolved_oxygen`, which narrowed 72 stations down to 14 with DO sensors listed. Then for each of those 14, a `GET /collections/NDBC/items/{id}` call returned the full item metadata including OPeNDAP asset URLs we'd use to download data. No URL construction, no guessing. The download links came straight from the catalog.

Two of those 14 turned out to have garbage data: every dissolved oxygen reading was the fill value 99.0, meaning the sensor was listed but never produced real measurements. Quality filtering dropped them, leaving 12 stations with valid dissolved oxygen records.

The CoastWatch discovery was the most interesting addition. AQUAVIEW returned satellite product items with ERDDAP endpoints in their `assets` field. One product, the Multiparameter Eddy Significance Index, bundles daily satellite SST, sea surface salinity, chlorophyll-a, sea surface height, and eddy kinetic energy into a single gridded dataset at 0.25° resolution. Because ERDDAP supports server-side spatial subsetting, we could request just the grid cells covering our study area rather than downloading the entire global product. The ERDDAP CSV URL came from `GET /collections/COASTWATCH/items/noaacweddymesiplusdaily`, and the subsetting query pulled daily values for only our bounding box.

That's three data access patterns: NDBC stations via OPeNDAP (THREDDS), WOD profiles via S3, and CoastWatch satellite grids via ERDDAP. Each has its own URL structure, its own subsetting conventions, and its own metadata schema. AQUAVIEW abstracts all of that behind `assets.csv.href` and `assets.dods.href`.

## The Station Network

AQUAVIEW discovered 12 usable NDBC stations with dissolved oxygen sensors across the northern Gulf, spanning from Apalachicola, FL to Grand Bay, MS:

| Station | Location | Days | Hypoxic Days | Events |
|---------|----------|------|-------------|--------|
| GBHM6 | Bayou Heron, Grand Bay, MS | 2,463 | 469 | 100 |
| KATA1 | Katrina Cut, AL | 2,704 | 110 | 43 |
| BSCA1 | Bon Secour, AL | 3,576 | 96 | 47 |
| CRTA1 | Cedar Point, AL | 3,362 | 48 | 11 |
| MBLA1 | Middle Bay Lighthouse, AL | 2,365 | 48 | 22 |
| MHPA1 | Meaher Park, AL | 4,717 | 26 | 3 |
| GDQM6 | Bangs Lake, Grand Bay, MS | 1,958 | 17 | 4 |
| DPHA1 | Dauphin Island Sea Lab, AL | 4,228 | 17 | 5 |
| WKQA1 | Fish River, Weeks Bay, AL | 1,566 | 4 | 2 |
| PPTA1 | Perdido Pass, AL | 3,006 | 0 | 0 |
| GBCM6 | Bayou Cumbest, Grand Bay, MS | 480 | 0 | 0 |
| ADBF1 | Dry Bar, Apalachicola, FL | 71 | 0 | 0 |

That's roughly 30,500 station-days of observations, 237 hypoxia onset events, and 835 total hypoxic days across 9 stations. GBHM6 alone accounts for more than half of all hypoxic days. Hypoxia is extremely localized even within this relatively small network.

Beyond the in-situ stations, AQUAVIEW discovered 1,770 WOD cruise datasets with oxygen profiles and 33 CoastWatch satellite products in the study area. The satellite data provides daily basin-scale context: sea surface temperature (warm water holds less dissolved oxygen), chlorophyll-a (algal blooms deplete oxygen when they die), sea surface height and eddy kinetic energy (physical drivers of mixing and stratification).

## What the Data Shows

The station correlations tell an interesting story. Once fill values were properly filtered, most stations showed strong monthly DO correlations with DPHA1: Cedar Point (r=0.901), Bon Secour (r=0.895), Perdido Pass (r=0.883). The Grand Bay stations correlate well too (GDQM6: r=0.793, GBHM6: r=0.786). Nearly every station in the network tracks a similar seasonal oxygen cycle.

This matters for modeling because it means the cross-station features carry real signal. A drop in DO at Cedar Point today likely reflects conditions that will show up at Dauphin Island soon. The network isn't just more data; it's spatial context that a single-station model can't capture.

## The Modeling Approach

We train XGBoost classifiers to predict hypoxia onset at 1, 3, 5, and 7-day lead times using five feature sets that test what information actually matters:

- Cross-source (NDBC + satellite + WOD): everything, including local conditions, neighboring stations, satellite-derived SST/chlorophyll/SSH, and offshore profiles
- NDBC + satellite: multi-station features plus satellite context, without WOD
- NDBC network: multi-station features only
- Local-only: just the target station's own measurements
- No-DO: all features except dissolved oxygen, testing whether temperature, salinity, satellite SST, and seasonality alone carry predictive signal

The onset target labels the N days *before* each hypoxia event as positive. Only non-hypoxic days are used for training and evaluation, so the model must predict the *transition*, not just persistence.

## Results

The model performance tells us something about what information matters at different prediction horizons.

| Lead | Cross-source (319 feat) | NDBC only (283) | Local only (173) | No DO (182) |
|------|------------------------|-----------------|------------------|-------------|
| 1 day | 0.917 / 0.398 | 0.921 / 0.419 | 0.919 / 0.424 | 0.807 / 0.196 |
| 3 day | 0.873 / 0.441 | 0.872 / 0.460 | 0.866 / 0.450 | 0.828 / 0.368 |
| 5 day | 0.866 / 0.610 | 0.847 / 0.555 | 0.843 / 0.541 | 0.853 / 0.581 |
| 7 day | 0.853 / 0.642 | 0.842 / 0.590 | 0.843 / 0.608 | 0.850 / 0.671 |

*(AUC-ROC / AUC-PR for each feature set)*

At one-day lead, local station data alone (AUC-ROC 0.919) performs essentially the same as the full multi-source model (0.917). When you're predicting 24 hours out, the target station's own dissolved oxygen trend contains most of the signal. Adding satellite or network features doesn't help and slightly hurts from noise.

The picture changes at five and seven-day lead times. At five days, the satellite-enriched model (cross-source: 0.866 AUC-ROC, 0.610 AUC-PR) outperforms NDBC-only (0.847, 0.555). At seven days, the gap persists: 0.853/0.642 vs 0.842/0.590. The satellite features, particularly SST anomalies and eddy kinetic energy from CoastWatch's Multiparameter Eddy Significance Index, provide basin-scale context that arrives at the coast days later. That's 36 additional engineered features from 1,438 days of daily satellite coverage, discovered and accessed through AQUAVIEW's CoastWatch collection.

The no-DO set is revealing. At one day, removing dissolved oxygen measurements drops AUC-ROC from 0.921 to 0.807. At seven days, no-DO (0.850) nearly matches the full model (0.853). Environmental context alone carries almost all the predictive signal at longer horizons. Direct DO measurement matters most for the shortest warning, which makes intuitive sense: the oxygen is already dropping.

One honest note: cross-source and NDBC-plus-satellite are identical in these results because WOD offshore profiles couldn't be accessed. The WOD annual files on S3 are multi-gigabyte global datasets that require byte-range requests rather than direct HTTP access. AQUAVIEW correctly discovered 1,770 WOD cruise datasets and provided their S3 URLs, but the download infrastructure wasn't in place for this analysis. The cross-source signal here comes entirely from the satellite features.

## What AQUAVIEW Did and Didn't Do

Let's be specific about this.

AQUAVIEW helped with discovery across three different data systems. The question "what oceanographic data exists in this bounding box, across in-situ stations, cruise profiles, and satellite products" is answered in seconds. Each AQUAVIEW item's `assets` field contains the actual download URLs: OPeNDAP endpoints for NDBC, S3 paths for WOD, ERDDAP CSV endpoints for CoastWatch. The script goes directly from catalog query to data download without any URL reverse-engineering.

This is where the multi-source story becomes concrete. NDBC stations are hosted on THREDDS and accessed via OPeNDAP. WOD profiles live on S3. CoastWatch satellite grids use ERDDAP with spatial subsetting via query parameters. These are three fundamentally different data access patterns. Without a unifying catalog, you'd need to know that each system exists, learn how to search it, and figure out its URL conventions. With AQUAVIEW, it's three `GET /search` calls followed by `GET /collections/{id}/items/{item_id}` for each dataset you want. The asset URLs come back in a consistent format regardless of the backend.

We also found stations we didn't know about. The original analysis plan called for 9 stations based on prior knowledge of the region. AQUAVIEW's search returned 14 DO-equipped stations, 5 more than expected (Katrina Cut, Cedar Point, Bon Secour, and two offshore buoys). Two of those turned out to have bad data, but the other three added real value. KATA1 alone contributed 110 hypoxic days and 43 onset events.

What AQUAVIEW didn't do: the science. Knowing that satellite SST anomalies and chlorophyll concentrations serve as hypoxia precursors, designing the lag features that capture how those signals propagate from open water to the coast, filtering fill values: that's domain knowledge and engineering work. A catalog can point you to the data. It can't tell you what to do with it.

A practical limitation we hit: WOD annual files (like `wod_pfl_2022.nc`) are multi-gigabyte global datasets. AQUAVIEW correctly provides their S3 URLs, but these files are too large for direct HTTP access with xarray. Production use would need byte-range requests via s3fs or NCEI's per-cruise download service. The CoastWatch ERDDAP datasets required more iteration. AQUAVIEW provided the ERDDAP base URLs and variable metadata, but ERDDAP's griddap subsetting syntax is finicky: each variable needs its own dimension constraints, some datasets have hidden altitude dimensions, and coordinate values must align with the actual grid. Getting the queries right took several rounds. Once the syntax was correct, the server-side subsetting worked well and downloads were small. The catalog handled discovery and access URLs; the subsetting details were ours to figure out.

## Where This Is Heading

This analysis was built with an AI agent (Claude) handling the data discovery and pipeline construction. That's increasingly how ocean science gets done: researchers directing AI tools to find, access, and integrate data on their behalf. In that world, the value of a unified catalog changes.

An agent can brute-force its way through THREDDS directories and figure out WOD's S3 structure by trial and error. But that's fragile. It depends on page layouts not changing, URL patterns being guessable, and the agent already knowing which systems to check. It definitely can't guess that CoastWatch ERDDAP happens to have a multi-parameter eddy index product with daily chlorophyll in the same grid as SST. A structured STAC API with consistent metadata and asset URLs is what makes agent-driven data access *reliable* rather than merely possible.

And this scales with the catalog. AQUAVIEW currently spans 24 collections. As that grows to include more IOOS regional networks, satellite products, and international holdings, no agent can independently track all those access patterns. The catalog becomes the only practical entry point. For researchers who increasingly rely on AI tools to handle data wrangling, a unified catalog goes from convenience to infrastructure.

The pipeline itself is fully reproducible:

```bash
pip install -r requirements.txt
python multistation_extract.py        # AQUAVIEW discovery → data download
python hypoxia_forecast_pipeline.py   # Feature engineering → model training
```

The extraction script logs every AQUAVIEW API call it makes, from search queries to item lookups to asset URLs, so you can see exactly how the catalog is used.

## Key Takeaways

1. Onset prediction is feasible at multi-day lead times, with AUC-ROC above 0.85 at seven days out. Predicting when new hypoxia events will begin, not just whether low DO persists, is the operationally useful problem, and it's tractable with standard ML methods applied to multi-source data.

2. Satellite features matter most when lead time matters most. At one-day lead, local DO dominates. At five to seven days, satellite-derived SST, chlorophyll, and eddy energy improve AUC-PR by 5+ points over station-only models. AQUAVIEW's discovery of the CoastWatch eddy index product, a dataset we wouldn't have found through NDBC alone, is what made that comparison possible.

3. Hypoxia is hyperlocal. Even with 12 stations spanning 300+ km of coastline, GBHM6 alone accounts for 56% of all hypoxic days. Cross-station correlations are mostly strong (0.76 to 0.90), but the distribution of hypoxia itself is extremely uneven. A catalog that lets you quickly discover *all* available stations in a region is valuable precisely because you can't assume which ones matter in advance.

4. As agents handle more of the workflow, unified catalogs become infrastructure. This entire analysis, from data discovery across three systems to trained models, was built in a single session by an AI agent using AQUAVIEW's STAC API as the entry point. A structured catalog with consistent metadata and direct asset URLs is what makes that agent workflow reliable at scale.

---

*Data discovered and accessed via [AQUAVIEW](https://aquaview.ai) from NOAA NDBC, World Ocean Database, and CoastWatch ERDDAP. Inspired by [Rajasekaran et al. (2025)](https://arxiv.org/abs/2602.05178), "Benchmarking Artificial Intelligence Models for Daily Coastal Hypoxia Forecasting."*
