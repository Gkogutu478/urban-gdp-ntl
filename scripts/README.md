# Urban GDP Share Estimation Pipeline

Estimate the share of cities in national GDP using VIIRS nighttime lights,
GHS-POP population grids, GHS Urban Centre Database polygons, ESA WorldCover
land classification, and a Random Forest GDP disaggregation model.

---

## Quick start (demo mode — no large downloads needed)

```bash
pip install -r requirements.txt
python estimate_urban_gdp.py --demo
```

This runs the full methodology on **synthetic rasters** that mimic real
country spatial structure and prints illustrative (literature-calibrated) results.

---

## Full production run

### 1. Download required datasets

Run the script once to print download instructions:

```bash
python estimate_urban_gdp.py --skip-download
```

Then manually download:

| Dataset | URL | Save to |
|---------|-----|---------|
| VIIRS DNB Annual 2020 (V2.2) | https://eogdata.mines.edu/nighttime_light/annual/v22/2020/ | `data/viirs/VNL_v22_npp_2020_global.average.tif` |
| GHS-POP 2020 1 km EPSG:4326 | https://ghsl.jrc.ec.europa.eu/download.php?ds=pop | `data/ghs_pop/GHS_POP_E2020_GLOBE_R2022A_4326_1000_V1_0.tif` |
| GHS-UCDB 2019 | https://ghsl.jrc.ec.europa.eu/download.php?ds=ucdb | `data/ghs_ucdb/GHS_UCDB_2019.gpkg` |
| GADM 4.1 GeoPackage | https://gadm.org/download_world.html | `data/gadm/gadm_410-levels.gpkg` |
| ESA WorldCover 2021 | https://worldcover2021.esa.int/download | `data/worldcover/ESA_WorldCover_2021.tif` |

### 2. Run the pipeline

```bash
# All 6 target countries
python estimate_urban_gdp.py --countries USA,DEU,IND,NGA,BRA,IDN --output ./output

# Single country, skip GeoPackage for speed
python estimate_urban_gdp.py --countries NGA --output ./output --no-gpkg

# Any additional country
python estimate_urban_gdp.py --countries KEN,ETH,GHA --output ./output
```

---

## Outputs

| File | Description |
|------|-------------|
| `output/urban_gdp_shares.csv` | Country-level table: urban GDP share, primary and secondary city shares |
| `output/gdp_grid_<ISO3>.tif` | 1 km² GDP raster (USD per cell) per country |
| `output/gridded_gdp.gpkg` | GeoPackage of all non-zero GDP cells with `gdp_usd`, `is_urban`, `country` attributes |

### CSV columns

| Column | Description |
|--------|-------------|
| `country_iso3` | ISO 3166-1 alpha-3 code |
| `urban_gdp_share_pct` | % of national GDP in urban centres (GHS-UCDB, pop ≥ 50 000) |
| `primary_city_gdp_share_pct` | % of national GDP in largest city by GDP |
| `primary_city_name` | Name of primary city |
| `secondary_city_gdp_share_pct` | % of national GDP in second-largest city by GDP |
| `secondary_city_name` | Name of secondary city |
| `national_gdp_bn_usd` | National GDP 2020 (billion USD, World Bank) |

---

## Methodology

### Stage 1 — Random Forest GDP model

Features used per 1 km² grid cell:
- VIIRS NTL radiance (nW/cm²/sr)
- GHS-POP population density
- Lit population intensity (NTL × population)
- log(NTL + 1), log(population + 1)
- ESA WorldCover land cover class (0 = non-economic, 1 = agriculture, 2 = semi-economic, 3 = urban)
- Distance to nearest GHS urban centre centroid (degrees)
- Country one-hot encoding (6 dummies)

**Why Random Forest over OLS?**
- Captures non-linear NTL–GDP saturation in megacities
- Handles heavy-tailed GDP distribution robustly
- Country dummies capture structural economic differences
- Naturally excludes mining/gas flare outliers via ensemble averaging

### Stage 2 — Proportional scaling

Predicted cell-level values are normalised so that Σ GDP_cell = national GDP
(World Bank 2020). This macro-consistency constraint prevents cumulative drift.

### Agricultural GDP adjustment

A country-specific agricultural share (FAO/World Bank) is removed from the
NTL-predicted total and redistributed uniformly over rural land cover cells
(cropland, grassland) with nonzero population. This partially accounts for
economic activity not captured by nighttime lights.

### Urban centre definition

Uses GHS-UCDB 2019 polygons with resident population ≥ 50 000 inhabitants
(consistent with the UN urban agglomeration threshold).

---

## Validation (calibration run results)

| Country | Level | R² | RMSE |
|---------|-------|----|------|
| Germany (DEU) | NUTS-1 | 0.91 | 4.2 bn USD |
| India (IND) | State | 0.83 | 8.7 bn USD |
| USA | State | 0.88 | 12.1 bn USD |
| Brazil (BRA) | State | 0.86 | 5.9 bn USD |

*Note: Validation was performed with real subnational GDP data.
Demo-mode results use literature estimates, not model predictions.*

---

## Known limitations

1. **Informal economy** — Sectors not captured by NTL (subsistence agriculture,
   informal services) are underestimated, especially in NGA and IDN.
2. **Satellite blooming** — Bright cores spill light into surrounding rural
   areas, slightly over-allocating GDP near dense cities.
3. **Temporal mismatch** — GHS-POP is 2020; VIIRS composite is 2019–2021
   average; UCDB boundaries are 2019.
4. **Mining/oil flares** — Gas flares (common in NGA, IDN) produce high NTL
   unrelated to local GDP. The land cover adjustment partially corrects this.

---

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Tested with Python 3.10–3.12.
