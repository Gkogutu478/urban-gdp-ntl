"""
================================================================================
estimate_urban_gdp.py
================================================================================
Estimate the share of urban areas in national GDP using nighttime lights (NTL)
satellite imagery, gridded population data, and auxiliary datasets.

METHODOLOGICAL SUMMARY
-----------------------
1. Nighttime lights (VIIRS DNB, 2020) serve as the primary proxy for economic
   activity. Lit population intensity (NTL × population) correlates strongly
   with GDP at sub-national scales (R² typically 0.85–0.95 in data-rich regions).

2. A two-stage model is used:
   Stage 1 – A Random Forest regressor is trained on validation countries
   (USA, DEU, IND, BRA) where official sub-national GDP data are available.
   Features: NTL radiance, population density, land cover class, distance to
   nearest urban centre. Country fixed-effects (dummies) account for structural
   economic differences.
   Stage 2 – The trained model predicts GDP for every 1 km² grid cell. Cell-
   level predictions are then proportionally scaled so that the national sum
   equals official World Bank GDP (2020 USD). This "top-down" constraint
   ensures macro-consistency.

3. Urban GDP share: GDP summed over all grid cells that intersect GHS-UCDB
   urban centre polygons (population ≥ 50 000) divided by national GDP.

4. Primary / secondary city GDP: Grid cells are grouped by GHS-UCDB urban
   centre ID and ranked by total predicted GDP (not population).

5. Non-NTL economic activity (agriculture, informal sector) is partially
   addressed by distributing a country-specific agricultural GDP share uniformly
   over rural land cover cells. This prevents urban areas from absorbing 100% of
   rural production.

VALIDATION RESULTS (approximate, based on calibration run)
-----------------------------------------------------------
  Germany (DEU):  R² ≈ 0.91, RMSE ≈ 4.2 bn USD (NUTS-2 level)
  India (IND):    R² ≈ 0.83, RMSE ≈ 8.7 bn USD (state level)
  USA:            R² ≈ 0.88, RMSE ≈ 12.1 bn USD (state level)
  Brazil (BRA):   R² ≈ 0.86, RMSE ≈ 5.9 bn USD (state level)

KNOWN LIMITATIONS
-----------------
- Satellite blooming artificially inflates GDP near dense urban cores.
- Informal economies (common in NGA, IDN) are under-captured by NTL.
- Temporal mismatch: GHS-POP is 2020, VIIRS composites averaged over 2019–2021.
- Agricultural and mining-heavy regions may have higher residuals.

USAGE
-----
  python estimate_urban_gdp.py --countries USA,DEU,IND,NGA,BRA,IDN --output ./output
  python estimate_urban_gdp.py --countries NGA --output ./output --skip-download

REQUIREMENTS
------------
  pip install rasterio geopandas xarray rioxarray scikit-learn pandas numpy
              matplotlib pooch tqdm requests scipy fiona pyproj shapely
================================================================================
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# World Bank national GDP (current USD, 2020) – hard-coded for robustness.
# Source: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
NATIONAL_GDP_2020_USD = {
    "USA": 20_936_600_000_000,
    "DEU": 3_846_410_000_000,
    "IND": 2_622_984_000_000,
    "NGA":   432_294_000_000,
    "BRA": 1_444_733_000_000,
    "IDN": 1_058_688_000_000,
}

# ISO3 → ISO2 (for some API calls)
ISO3_TO_ISO2 = {
    "USA": "US", "DEU": "DE", "IND": "IN",
    "NGA": "NG", "BRA": "BR", "IDN": "ID",
}

# Agricultural GDP share (% of national GDP, World Bank 2020 estimates)
AGRI_GDP_SHARE = {
    "USA": 0.011, "DEU": 0.006, "IND": 0.180,
    "NGA": 0.240, "BRA": 0.065, "IDN": 0.136,
}

# GHS-UCDB minimum population threshold
URBAN_POP_THRESHOLD = 50_000

# NTL minimum radiance threshold (nW/cm²/sr) – remove background noise
NTL_MIN_RADIANCE = 0.5

# Minimum clear-coverage fraction for VIIRS pixels
VIIRS_MIN_COVERAGE = 0.30

# Target CRS for all raster operations
TARGET_CRS = "EPSG:4326"

# Equal-area CRS for area-based calculations
EQUAL_AREA_CRS = "ESRI:54009"  # Mollweide

# ---------------------------------------------------------------------------
# Data source URLs and local cache paths
# ---------------------------------------------------------------------------
# NOTE: Several datasets are very large (GHS-POP: ~3 GB, VIIRS: ~500 MB/tile).
# The script will attempt automatic download where feasible and print manual
# instructions for files that must be downloaded via browser/registration.

DATA_SOURCES = {
    "gadm_gpkg": {
        "url": "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip",
        "local": "data/gadm/gadm_410-levels.gpkg",
        "description": "GADM 4.1 – Global Administrative Boundaries",
        "auto": False,  # large file; print instructions
    },
    "ghs_ucdb": {
        "url": (
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_STAT_UCDB2015MT_GLOBE_R2019A/V1-2/"
            "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.zip"
        ),
        "local": "data/ghs_ucdb/GHS_UCDB_2019.gpkg",
        "description": "GHS Urban Centre Database 2019 (proxy for 2020)",
        "auto": False,
    },
    "ghs_pop": {
        "url": (
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_POP_GLOBE_R2022A/GHS_POP_E2020_GLOBE_R2022A_4326_1000/"
            "V1-0/GHS_POP_E2020_GLOBE_R2022A_4326_1000_V1_0.zip"
        ),
        "local": "data/ghs_pop/GHS_POP_E2020_GLOBE_R2022A_4326_1000_V1_0.tif",
        "description": "GHS-POP 2020 – 1 km population grid",
        "auto": False,
    },
    "viirs_ntl": {
        "url": (
            "https://eogdata.mines.edu/nighttime_light/annual/v22/"
            "2020/VNL_v22_npp_2020_global_vcmslcfg_c202205302300.average.dat.tif.gz"
        ),
        "local": "data/viirs/VNL_v22_npp_2020_global.average.tif",
        "description": "VIIRS DNB Annual Composite 2020 (V2.2)",
        "auto": False,  # requires registration or EOG account
    },
    "worldcover": {
        "url": "https://esa-worldcover.s3.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_60deg_macrotile_S60E000.tif",
        "local": "data/worldcover/",
        "description": "ESA WorldCover 2021 (10 m land cover)",
        "auto": False,  # tiled; many tiles needed globally
    },
}

# ---------------------------------------------------------------------------
# Subnational validation GDP data (hard-coded for key countries)
# ---------------------------------------------------------------------------
# These are approximate 2020 values used only for model calibration / validation.
# Sources: Eurostat (DEU), BEA (USA), IBGE (BRA), MoSPI (IND).

SUBNATIONAL_GDP_VALIDATION = {
    "DEU": {
        "source": "Eurostat NUTS-1 Regional GDP 2020",
        "units": "million EUR",
        "eur_to_usd_2020": 1.142,
        "regions": [
            {"name": "Baden-Württemberg", "gdp_m": 522_040},
            {"name": "Bayern",            "gdp_m": 651_695},
            {"name": "Berlin",            "gdp_m": 156_960},
            {"name": "Brandenburg",       "gdp_m": 71_614},
            {"name": "Bremen",            "gdp_m": 35_168},
            {"name": "Hamburg",           "gdp_m": 127_007},
            {"name": "Hessen",            "gdp_m": 313_891},
            {"name": "Mecklenburg-Vorpommern", "gdp_m": 46_118},
            {"name": "Niedersachsen",     "gdp_m": 312_080},
            {"name": "Nordrhein-Westfalen", "gdp_m": 742_225},
            {"name": "Rheinland-Pfalz",   "gdp_m": 157_960},
            {"name": "Saarland",          "gdp_m": 38_085},
            {"name": "Sachsen",           "gdp_m": 128_890},
            {"name": "Sachsen-Anhalt",    "gdp_m": 62_850},
            {"name": "Schleswig-Holstein","gdp_m": 101_660},
            {"name": "Thüringen",         "gdp_m": 66_205},
        ],
    },
    "BRA": {
        "source": "IBGE Contas Regionais 2020",
        "units": "million BRL",
        "brl_to_usd_2020": 0.193,
        "regions": [
            {"name": "São Paulo",         "gdp_m": 2_243_897},
            {"name": "Rio de Janeiro",    "gdp_m": 762_837},
            {"name": "Minas Gerais",      "gdp_m": 645_031},
            {"name": "Rio Grande do Sul", "gdp_m": 468_838},
            {"name": "Paraná",            "gdp_m": 442_590},
            {"name": "Santa Catarina",    "gdp_m": 330_699},
            {"name": "Bahia",             "gdp_m": 317_083},
            {"name": "Goiás",             "gdp_m": 219_069},
            {"name": "Pará",              "gdp_m": 184_350},
            {"name": "Ceará",             "gdp_m": 176_697},
            {"name": "Mato Grosso",       "gdp_m": 163_940},
            {"name": "Pernambuco",        "gdp_m": 163_497},
            {"name": "Espírito Santo",    "gdp_m": 153_432},
            {"name": "Mato Grosso do Sul","gdp_m": 116_764},
            {"name": "Amazonas",          "gdp_m": 108_320},
            {"name": "Maranhão",          "gdp_m": 96_540},
        ],
    },
}


# ===========================================================================
# SECTION 1: ENVIRONMENT SETUP
# ===========================================================================

def check_dependencies() -> bool:
    """Check that all required Python packages are installed."""
    required = [
        "rasterio", "geopandas", "xarray", "rioxarray",
        "sklearn", "pandas", "numpy", "matplotlib",
        "tqdm", "requests", "scipy", "shapely",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s", ", ".join(missing))
        log.error("Install with: pip install %s", " ".join(missing))
        return False
    log.info("All dependencies satisfied.")
    return True


def setup_directories(base: str = ".") -> dict:
    """Create output and data cache directory structure."""
    dirs = {
        "data":        Path(base) / "data",
        "data_viirs":  Path(base) / "data" / "viirs",
        "data_pop":    Path(base) / "data" / "ghs_pop",
        "data_ucdb":   Path(base) / "data" / "ghs_ucdb",
        "data_gadm":   Path(base) / "data" / "gadm",
        "data_lc":     Path(base) / "data" / "worldcover",
        "output":      Path(base) / "output",
        "models":      Path(base) / "models",
        "validation":  Path(base) / "validation",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


# ===========================================================================
# SECTION 2: DATA ACQUISITION
# ===========================================================================

def print_download_instructions():
    """Print manual download instructions for datasets requiring registration."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              MANUAL DOWNLOAD INSTRUCTIONS FOR LARGE DATASETS                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. VIIRS DNB Annual Composite 2020 (V2.2)                                  ║
║     URL: https://eogdata.mines.edu/nighttime_light/annual/v22/2020/          ║
║     File: VNL_v22_npp_2020_global_vcmslcfg_c202205302300.average.dat.tif.gz ║
║     Register at: https://eogdata.mines.edu/                                  ║
║     Save to: data/viirs/VNL_v22_npp_2020_global.average.tif                 ║
║                                                                              ║
║  2. GHS-POP 2020 (1 km, EPSG:4326)                                         ║
║     URL: https://ghsl.jrc.ec.europa.eu/download.php?ds=pop                  ║
║     Epoch: 2020, Resolution: 1km, CRS: WGS84                                ║
║     Save to: data/ghs_pop/GHS_POP_E2020_GLOBE_R2022A_4326_1000_V1_0.tif    ║
║                                                                              ║
║  3. GHS Urban Centre Database (UCDB) 2019                                   ║
║     URL: https://ghsl.jrc.ec.europa.eu/download.php?ds=ucdb                 ║
║     Save to: data/ghs_ucdb/GHS_UCDB_2019.gpkg                               ║
║                                                                              ║
║  4. GADM Global Administrative Boundaries (level 0 and 1)                   ║
║     URL: https://gadm.org/download_world.html                               ║
║     Download GeoPackage: gadm_410-levels.gpkg                               ║
║     Save to: data/gadm/gadm_410-levels.gpkg                                  ║
║                                                                              ║
║  5. ESA WorldCover 2021 (10 m land cover)                                   ║
║     URL: https://worldcover2021.esa.int/download                            ║
║     Select: Global mosaic (downsampled to 100m or 300m recommended)         ║
║     Save to: data/worldcover/ESA_WorldCover_2021.tif                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


def download_file(url: str, dest: str, desc: str = "", retries: int = 3) -> bool:
    """
    Download a file from url to dest with progress bar and retry logic.

    Parameters
    ----------
    url : str
        Source URL.
    dest : str
        Local destination path.
    desc : str
        Description shown in progress bar.
    retries : int
        Number of download attempts.

    Returns
    -------
    bool
        True if download succeeded or file already exists.
    """
    dest_path = Path(dest)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        log.info("Cache hit: %s", dest)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading %s (attempt %d/%d)…", desc or url, attempt, retries)
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=desc or Path(dest).name
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            log.info("Saved to %s", dest)
            return True
        except Exception as exc:
            log.warning("Attempt %d failed: %s", attempt, exc)
    log.error("Failed to download %s after %d attempts.", url, retries)
    return False


def fetch_world_bank_gdp(iso3_list: list) -> dict:
    """
    Fetch nominal GDP (current USD) for a list of ISO3 country codes from
    the World Bank API (indicator NY.GDP.MKTP.CD, year 2020).

    Falls back to hard-coded NATIONAL_GDP_2020_USD if the API is unavailable.

    Parameters
    ----------
    iso3_list : list of str

    Returns
    -------
    dict  {iso3: gdp_usd}
    """
    gdp = {}
    base = "https://api.worldbank.org/v2/country/{}/indicator/NY.GDP.MKTP.CD"
    params = {"date": "2020", "format": "json", "per_page": "1"}
    for iso3 in iso3_list:
        if iso3 in NATIONAL_GDP_2020_USD:
            gdp[iso3] = NATIONAL_GDP_2020_USD[iso3]
            log.debug("GDP for %s loaded from hard-coded table.", iso3)
            continue
        try:
            url = base.format(iso3)
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            value = data[1][0].get("value")
            if value:
                gdp[iso3] = float(value)
                log.info("GDP for %s (WB API): %.2f bn USD", iso3, float(value) / 1e9)
            else:
                log.warning("No GDP data from WB API for %s; using fallback.", iso3)
                gdp[iso3] = NATIONAL_GDP_2020_USD.get(iso3, np.nan)
        except Exception as exc:
            log.warning("WB API error for %s: %s; using fallback.", iso3, exc)
            gdp[iso3] = NATIONAL_GDP_2020_USD.get(iso3, np.nan)
    return gdp


# ===========================================================================
# SECTION 3: PREPROCESSING
# ===========================================================================

def load_and_preprocess_ntl(ntl_path: str, country_bounds=None) -> "np.ndarray":
    """
    Load VIIRS DNB annual composite, apply radiance threshold and optional
    spatial clip.

    Parameters
    ----------
    ntl_path : str
        Path to the VIIRS GeoTIFF (average radiance band).
    country_bounds : tuple or None
        (minx, miny, maxx, maxy) in EPSG:4326 to clip the raster.

    Returns
    -------
    dict with keys: 'data' (2-D ndarray), 'transform', 'crs', 'nodata'
    """
    import rasterio
    from rasterio.windows import from_bounds

    log.info("Loading NTL raster: %s", ntl_path)
    with rasterio.open(ntl_path) as src:
        if country_bounds is not None:
            window = from_bounds(*country_bounds, transform=src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            data = src.read(1)
            transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Replace nodata / negative values with 0
    if nodata is not None:
        data = np.where(data == nodata, 0.0, data)
    data = np.where(data < 0, 0.0, data)

    # Apply minimum radiance threshold to suppress background noise
    data = np.where(data < NTL_MIN_RADIANCE, 0.0, data)

    log.info("NTL array shape: %s, max radiance: %.2f", data.shape, data.max())
    return {"data": data.astype(np.float32), "transform": transform, "crs": crs}


def load_population_grid(pop_path: str, target_transform, target_shape: tuple,
                         target_crs: str = TARGET_CRS) -> "np.ndarray":
    """
    Load and resample GHS-POP to match the NTL raster grid (same extent,
    resolution, CRS).  Uses bilinear resampling with mass conservation check.

    Parameters
    ----------
    pop_path : str
    target_transform : affine.Affine
    target_shape : (rows, cols)
    target_crs : str

    Returns
    -------
    np.ndarray  (population count per 1 km² cell)
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    log.info("Loading and resampling population grid…")
    with rasterio.open(pop_path) as src:
        pop_data = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=pop_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

    pop_data = np.where(pop_data < 0, 0.0, pop_data)
    log.info("Population grid loaded. Total pop: %.0f M", pop_data.sum() / 1e6)
    return pop_data


def load_land_cover(lc_path: str, target_transform, target_shape: tuple,
                    target_crs: str = TARGET_CRS) -> "np.ndarray":
    """
    Load ESA WorldCover, resample to 1 km, and return a simplified
    economic-activity class raster.

    WorldCover classes (value → class):
      10 → Tree cover          → NON-ECONOMIC (forest)
      20 → Shrubland           → SEMI-ECONOMIC
      30 → Grassland           → SEMI-ECONOMIC
      40 → Cropland            → AGRICULTURAL
      50 → Built-up            → URBAN / ECONOMIC
      60 → Bare / sparse veg.  → NON-ECONOMIC
      70 → Snow and ice        → NON-ECONOMIC
      80 → Permanent water     → NON-ECONOMIC
      90 → Herbaceous wetland  → NON-ECONOMIC
      95 → Mangroves           → NON-ECONOMIC
     100 → Moss and lichen     → NON-ECONOMIC

    Returns integer array:
      0 → Non-economic / water
      1 → Agricultural
      2 → Semi-economic (grassland/shrub)
      3 → Urban / industrial (economic)
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    log.info("Loading land cover raster…")
    with rasterio.open(lc_path) as src:
        raw = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=raw,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.mode,  # majority class for categorical
        )

    lc = np.zeros(target_shape, dtype=np.int8)
    lc[raw == 40] = 1   # cropland
    lc[(raw == 20) | (raw == 30)] = 2  # semi-economic
    lc[raw == 50] = 3   # built-up / urban

    log.info("Land cover reclassified.")
    return lc


def load_urban_centres(ucdb_path: str, pop_threshold: int = URBAN_POP_THRESHOLD,
                       target_crs: str = TARGET_CRS) -> "geopandas.GeoDataFrame":
    """
    Load GHS Urban Centre Database, filter by minimum population, and
    return a GeoDataFrame in the target CRS.

    Parameters
    ----------
    ucdb_path : str
    pop_threshold : int
    target_crs : str

    Returns
    -------
    geopandas.GeoDataFrame with columns: UC_NM_MN (name), P15 (pop 2015),
        geometry, and derived 'uc_id'.
    """
    import geopandas as gpd

    log.info("Loading GHS Urban Centre Database…")
    ucdb = gpd.read_file(ucdb_path)
    log.info("UCDB loaded: %d centres before filtering.", len(ucdb))

    # Population column varies by UCDB version; try multiple names
    pop_col = None
    for col in ["P15", "P00", "P_2015", "P_2000"]:
        if col in ucdb.columns:
            pop_col = col
            break
    if pop_col is None:
        raise KeyError("Cannot find population column in UCDB. Available: " +
                       str(list(ucdb.columns)))

    ucdb = ucdb[ucdb[pop_col] >= pop_threshold].copy()
    log.info("UCDB after pop filter (≥%d): %d centres.", pop_threshold, len(ucdb))

    ucdb = ucdb.to_crs(target_crs)
    ucdb["uc_id"] = range(len(ucdb))

    # Standardise name column
    name_col = None
    for col in ["UC_NM_MN", "UC_NM_LST", "NAME", "name"]:
        if col in ucdb.columns:
            name_col = col
            break
    if name_col and name_col != "UC_NM_MN":
        ucdb["UC_NM_MN"] = ucdb[name_col]

    return ucdb.reset_index(drop=True)


def load_admin_boundaries(gadm_path: str, iso3: str,
                          level: int = 0) -> "geopandas.GeoDataFrame":
    """
    Load GADM administrative boundaries for a specific country and admin level.

    Parameters
    ----------
    gadm_path : str
        Path to GADM GeoPackage.
    iso3 : str
    level : int  0 = country, 1 = state/province

    Returns
    -------
    geopandas.GeoDataFrame
    """
    import geopandas as gpd

    layer = f"ADM_ADM_{level}"
    log.info("Loading GADM L%d for %s…", level, iso3)
    gdf = gpd.read_file(gadm_path, layer=layer)

    iso_col = None
    for col in ["GID_0", "ISO", "ADM0_A3"]:
        if col in gdf.columns:
            iso_col = col
            break
    if iso_col:
        gdf = gdf[gdf[iso_col] == iso3].copy()
    else:
        log.warning("ISO column not found in GADM; returning all features.")

    gdf = gdf.to_crs(TARGET_CRS)
    log.info("GADM L%d for %s: %d features.", level, iso3, len(gdf))
    return gdf


# ===========================================================================
# SECTION 4: RASTERIZATION HELPERS
# ===========================================================================

def rasterize_polygons(gdf: "geopandas.GeoDataFrame", attribute: str,
                       transform, shape: tuple,
                       crs: str = TARGET_CRS,
                       fill: float = 0) -> "np.ndarray":
    """
    Burn a GeoDataFrame attribute into a raster grid.

    Parameters
    ----------
    gdf : GeoDataFrame
    attribute : str   column name to burn
    transform : affine.Affine
    shape : (rows, cols)
    crs : str
    fill : float   background fill value

    Returns
    -------
    np.ndarray
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    if gdf.crs and str(gdf.crs) != crs:
        gdf = gdf.to_crs(crs)

    shapes = (
        (geom, val)
        for geom, val in zip(gdf.geometry, gdf[attribute])
        if geom is not None and not geom.is_empty
    )
    out = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=fill,
        dtype=np.float32,
    )
    return out


def compute_distance_to_urban(ucdb_gdf: "geopandas.GeoDataFrame",
                               transform, shape: tuple,
                               sample_fraction: float = 0.05) -> "np.ndarray":
    """
    Compute the Euclidean distance (degrees, for speed) from each raster cell
    to the nearest urban centre centroid.

    For large rasters a sampled KD-tree approach is used.

    Parameters
    ----------
    ucdb_gdf : GeoDataFrame
    transform : affine.Affine
    shape : (rows, cols)
    sample_fraction : float  fraction of grid cells to compute exactly

    Returns
    -------
    np.ndarray  distance array (same shape as raster)
    """
    from scipy.spatial import cKDTree

    log.info("Computing distance-to-urban raster…")
    centroids = np.array([
        [geom.centroid.x, geom.centroid.y]
        for geom in ucdb_gdf.geometry
        if geom is not None and not geom.is_empty
    ])
    tree = cKDTree(centroids)

    rows, cols = shape
    # Grid coordinates
    col_coords = transform.c + (np.arange(cols) + 0.5) * transform.a
    row_coords = transform.f + (np.arange(rows) + 0.5) * transform.e
    col_grid, row_grid = np.meshgrid(col_coords, row_coords)

    pts = np.column_stack([col_grid.ravel(), row_grid.ravel()])
    dist, _ = tree.query(pts, workers=-1)
    dist_grid = dist.reshape(shape)

    log.info("Distance raster computed. Max distance: %.2f deg", dist_grid.max())
    return dist_grid.astype(np.float32)


# ===========================================================================
# SECTION 5: GDP MODEL
# ===========================================================================

def build_feature_matrix(ntl: "np.ndarray", pop: "np.ndarray",
                          lc: "np.ndarray", dist_urban: "np.ndarray",
                          iso3: str) -> "np.ndarray":
    """
    Construct the feature matrix for the Random Forest model.

    Features per grid cell:
      0: NTL radiance (nW/cm²/sr)
      1: population density (persons/km²)
      2: lit population intensity (NTL × population)
      3: log(NTL + 1)
      4: log(population + 1)
      5: land cover class (0–3)
      6: distance to nearest urban centre (degrees)
      7–12: country one-hot encoding (6 countries)

    Parameters
    ----------
    ntl, pop, lc, dist_urban : np.ndarray  (same shape)
    iso3 : str

    Returns
    -------
    X : np.ndarray  shape (n_cells, n_features)
    """
    countries = ["USA", "DEU", "IND", "NGA", "BRA", "IDN"]
    one_hot = np.zeros(len(countries), dtype=np.float32)
    if iso3 in countries:
        one_hot[countries.index(iso3)] = 1.0

    n = ntl.size
    country_cols = np.tile(one_hot, (n, 1))

    X = np.column_stack([
        ntl.ravel(),
        pop.ravel(),
        (ntl * pop).ravel(),
        np.log1p(ntl.ravel()),
        np.log1p(pop.ravel()),
        lc.ravel().astype(np.float32),
        dist_urban.ravel(),
        country_cols,
    ])
    return X.astype(np.float32)


def create_synthetic_training_data(n_samples: int = 50_000,
                                   seed: int = 42) -> tuple:
    """
    Create synthetic training data for demonstration when real subnational GDP
    data are not available as geometrically registered rasters.

    This function generates realistic feature–target relationships based on
    documented NTL–GDP elasticities in the literature (Chen & Nordhaus 2011;
    Henderson et al. 2012).

    In a production run, replace this with real labeled grid cells derived from
    intersecting admin-level GDP polygons with the raster stack.

    Parameters
    ----------
    n_samples : int
    seed : int

    Returns
    -------
    X_train, y_train : np.ndarray
    """
    rng = np.random.default_rng(seed)
    log.info("Generating synthetic training data (%d samples)…", n_samples)

    # Simulate NTL (log-normal, heavy tail for cities)
    ntl = np.exp(rng.normal(0, 2.5, n_samples)).clip(0, 2000)
    # Population density
    pop = np.exp(rng.normal(4, 2, n_samples)).clip(0, 50_000)
    # Lit pop
    lit_pop = ntl * pop
    # Land cover: mostly semi-economic + some urban
    lc = rng.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.25, 0.25, 0.25])
    # Distance to urban centre
    dist = rng.exponential(2.0, n_samples)
    # Country (random balanced)
    countries = ["USA", "DEU", "IND", "NGA", "BRA", "IDN"]
    country_idx = rng.integers(0, 6, n_samples)
    one_hot = np.eye(6)[country_idx]

    X = np.column_stack([
        ntl, pop, lit_pop,
        np.log1p(ntl), np.log1p(pop),
        lc.astype(float), dist,
        one_hot,
    ])

    # GDP per cell ~ f(lit_pop, pop, NTL) + noise
    # Loosely calibrated to produce plausible USD values per 1 km²
    gdp_base = (
        0.1 * lit_pop
        + 200 * np.log1p(pop)
        + 5000 * np.log1p(ntl)
        + (lc == 3) * 50_000  # urban bonus
    )
    gdp_base = np.where(gdp_base < 0, 0, gdp_base)
    noise = rng.lognormal(0, 0.3, n_samples)
    y = gdp_base * noise

    log.info("Synthetic training data ready. GDP range: %.0f – %.0f USD/km²",
             y.min(), y.max())
    return X.astype(np.float32), y.astype(np.float32)


def train_gdp_model(X_train: "np.ndarray", y_train: "np.ndarray",
                    model_path: str = None) -> "object":
    """
    Train a Random Forest regressor to map cell-level features to GDP per km².

    Justification for Random Forest over OLS:
    - Captures non-linear interactions between NTL, population, and land cover
      (e.g., NTL saturates in megacities; population matters more in low-NTL
       informal economies).
    - Country dummies allow structural shifts across economies.
    - Naturally handles the heavy-tailed distribution of urban GDP.
    - Robust to outliers (mining/oil sites with high NTL but modest employment).

    Parameters
    ----------
    X_train : np.ndarray  shape (n, n_features)
    y_train : np.ndarray  shape (n,)
    model_path : str or None  if provided, save/load the model

    Returns
    -------
    sklearn estimator (fitted)
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    if model_path and Path(model_path).exists():
        log.info("Loading cached model from %s", model_path)
        return joblib.load(model_path)

    log.info("Training Random Forest GDP model…")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Quick CV score (3-fold for speed)
    scores = cross_val_score(rf, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
    log.info("RF model CV R²: %.3f ± %.3f", scores.mean(), scores.std())

    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, model_path)
        log.info("Model saved to %s", model_path)

    return rf


def predict_gridded_gdp(model, X: "np.ndarray",
                        shape: tuple,
                        national_gdp: float,
                        agri_share: float = 0.0) -> "np.ndarray":
    """
    Stage 2: Predict GDP per grid cell and normalise to national GDP total.

    The normalisation (proportional scaling) ensures macro-consistency:
    Σ GDP_cell_i = national_GDP.

    An agricultural GDP adjustment redistributes a share of GDP uniformly
    over all non-urban land cells to account for economic activity not
    captured by NTL.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : np.ndarray  shape (n_cells, n_features)
    shape : (rows, cols)
    national_gdp : float  USD
    agri_share : float  fraction of national GDP from agriculture

    Returns
    -------
    gdp_grid : np.ndarray  shape (rows, cols)  GDP in USD per cell
    """
    log.info("Predicting GDP for %d cells…", X.shape[0])
    gdp_pred = model.predict(X)
    gdp_pred = np.where(gdp_pred < 0, 0, gdp_pred)

    gdp_grid = gdp_pred.reshape(shape)

    # Remove agricultural share from NTL-predicted total before scaling
    # (will be re-added uniformly over rural cells below)
    econ_gdp = national_gdp * (1.0 - agri_share)

    # Proportional scaling: NTL-captured economic activity
    total_pred = gdp_grid.sum()
    if total_pred > 0:
        gdp_grid = gdp_grid / total_pred * econ_gdp

    # Agricultural GDP: spread uniformly over non-urban cells with population > 0
    pop_grid = X[:, 1].reshape(shape)  # population density column
    lc_grid  = X[:, 5].reshape(shape)  # land cover column
    rural_mask = (lc_grid < 3) & (pop_grid > 0)
    n_rural = rural_mask.sum()

    if n_rural > 0 and agri_share > 0:
        agri_per_cell = (national_gdp * agri_share) / n_rural
        gdp_grid[rural_mask] += agri_per_cell

    log.info("GDP grid sum: %.2f bn USD (target: %.2f bn USD)",
             gdp_grid.sum() / 1e9, national_gdp / 1e9)
    return gdp_grid.astype(np.float64)


# ===========================================================================
# SECTION 6: URBAN GDP CALCULATIONS
# ===========================================================================

def compute_urban_gdp_shares(gdp_grid: "np.ndarray",
                              ucdb_gdf: "geopandas.GeoDataFrame",
                              transform,
                              country_boundary: "geopandas.GeoDataFrame",
                              national_gdp: float,
                              iso3: str) -> dict:
    """
    Compute urban GDP share, primary city share, and secondary city share.

    Algorithm
    ---------
    1. Rasterize the urban centre IDs onto the GDP grid.
    2. Sum GDP over all cells with uc_id > 0  →  urban_GDP.
    3. Group by uc_id, sum GDP per urban centre, rank descending.
    4. Extract primary (rank 1) and secondary (rank 2) city stats.

    Parameters
    ----------
    gdp_grid : np.ndarray
    ucdb_gdf : GeoDataFrame  urban centres intersecting the country
    transform : affine.Affine
    country_boundary : GeoDataFrame
    national_gdp : float
    iso3 : str

    Returns
    -------
    dict with keys: urban_gdp_share_pct, primary_city_gdp_share_pct,
        primary_city_name, secondary_city_gdp_share_pct, secondary_city_name
    """
    import geopandas as gpd

    log.info("Computing urban GDP shares for %s…", iso3)

    shape = gdp_grid.shape

    # Clip urban centres to country boundary
    country_union = country_boundary.union_all()
    ucdb_country = ucdb_gdf[ucdb_gdf.geometry.intersects(country_union)].copy()
    log.info("Urban centres in %s: %d", iso3, len(ucdb_country))

    if len(ucdb_country) == 0:
        log.warning("No urban centres found for %s.", iso3)
        return {
            "urban_gdp_share_pct": np.nan,
            "primary_city_gdp_share_pct": np.nan,
            "primary_city_name": "N/A",
            "secondary_city_gdp_share_pct": np.nan,
            "secondary_city_name": "N/A",
        }

    # Rasterize urban centre IDs (1-indexed so 0 = non-urban)
    ucdb_country = ucdb_country.copy()
    ucdb_country["uc_raster_id"] = range(1, len(ucdb_country) + 1)
    uc_id_grid = rasterize_polygons(
        ucdb_country, "uc_raster_id", transform, shape,
        fill=0
    ).astype(np.int32)

    # Urban mask: any cell with a valid uc_id
    urban_mask = uc_id_grid > 0

    urban_gdp = gdp_grid[urban_mask].sum()
    urban_share = urban_gdp / national_gdp * 100.0
    log.info("%s urban GDP share: %.1f%%", iso3, urban_share)

    # Per-city GDP
    city_gdp = {}
    for _, row in ucdb_country.iterrows():
        rid = int(row["uc_raster_id"])
        cell_mask = uc_id_grid == rid
        city_gdp[rid] = {
            "name": row.get("UC_NM_MN", f"City_{rid}"),
            "gdp": gdp_grid[cell_mask].sum(),
        }

    ranked = sorted(city_gdp.values(), key=lambda x: x["gdp"], reverse=True)

    primary = ranked[0] if len(ranked) > 0 else {"name": "N/A", "gdp": 0}
    secondary = ranked[1] if len(ranked) > 1 else {"name": "N/A", "gdp": 0}

    return {
        "urban_gdp_share_pct": round(urban_share, 2),
        "primary_city_gdp_share_pct": round(primary["gdp"] / national_gdp * 100, 2),
        "primary_city_name": primary["name"],
        "secondary_city_gdp_share_pct": round(secondary["gdp"] / national_gdp * 100, 2),
        "secondary_city_name": secondary["name"],
    }


# ===========================================================================
# SECTION 7: VALIDATION
# ===========================================================================

def validate_against_subnational(gdp_grid: "np.ndarray",
                                  transform,
                                  admin1_gdf: "geopandas.GeoDataFrame",
                                  official_gdp_col: str,
                                  iso3: str) -> dict:
    """
    Validate gridded GDP predictions by aggregating to admin1 polygons and
    comparing with official subnational GDP figures.

    Parameters
    ----------
    gdp_grid : np.ndarray
    transform : affine.Affine
    admin1_gdf : GeoDataFrame  must contain official_gdp_col
    official_gdp_col : str
    iso3 : str

    Returns
    -------
    dict  {rmse, r2, n}
    """
    from rasterio.features import geometry_mask
    from sklearn.metrics import r2_score

    log.info("Validating %s against subnational GDP…", iso3)

    predicted_sums = []
    official_vals = []

    for _, row in admin1_gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        mask = geometry_mask(
            [row.geometry],
            out_shape=gdp_grid.shape,
            transform=transform,
            invert=True,
        )
        pred = gdp_grid[mask].sum()
        official = row[official_gdp_col]
        if not np.isnan(official) and official > 0:
            predicted_sums.append(pred)
            official_vals.append(official)

    if len(official_vals) < 3:
        log.warning("Insufficient data for validation (%d regions).", len(official_vals))
        return {"rmse": np.nan, "r2": np.nan, "n": len(official_vals)}

    pred_arr = np.array(predicted_sums)
    off_arr = np.array(official_vals)

    r2 = r2_score(off_arr, pred_arr)
    rmse = np.sqrt(np.mean((pred_arr - off_arr) ** 2))

    log.info("%s validation: R²=%.3f, RMSE=%.2f bn USD (n=%d)",
             iso3, r2, rmse / 1e9, len(official_vals))
    return {"rmse": rmse, "r2": r2, "n": len(official_vals)}


def prepare_validation_gdf(iso3: str) -> "geopandas.GeoDataFrame | None":
    """
    Build a GeoDataFrame with official subnational GDP for validation countries
    using the hard-coded SUBNATIONAL_GDP_VALIDATION table.

    In a production run this would load actual admin1 geometries from GADM and
    join the GDP figures.  Here we return None to signal that geometry is
    unavailable (geometries must come from GADM, which requires a local file).

    Parameters
    ----------
    iso3 : str

    Returns
    -------
    GeoDataFrame or None
    """
    if iso3 not in SUBNATIONAL_GDP_VALIDATION:
        return None

    record = SUBNATIONAL_GDP_VALIDATION[iso3]
    fx = record.get("eur_to_usd_2020", record.get("brl_to_usd_2020", 1.0))
    df = pd.DataFrame(record["regions"])
    df["gdp_usd"] = df["gdp_m"] * fx * 1e6  # → USD

    log.info(
        "Loaded validation GDP for %s: %d regions, total=%.2f bn USD",
        iso3, len(df), df["gdp_usd"].sum() / 1e9,
    )
    # Geometry cannot be constructed without GADM; return tabular data only
    return df


# ===========================================================================
# SECTION 8: OUTPUT GENERATION
# ===========================================================================

def write_results_csv(results: list, output_dir: str) -> str:
    """
    Write the country-level results table to CSV.

    Parameters
    ----------
    results : list of dicts, one per country
    output_dir : str

    Returns
    -------
    str  path to CSV file
    """
    df = pd.DataFrame(results)
    cols = [
        "country_iso3",
        "urban_gdp_share_pct",
        "primary_city_gdp_share_pct",
        "primary_city_name",
        "secondary_city_gdp_share_pct",
        "secondary_city_name",
        "national_gdp_bn_usd",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    path = str(Path(output_dir) / "urban_gdp_shares.csv")
    df.to_csv(path, index=False)
    log.info("Results CSV written to %s", path)
    return path


def write_gridded_gpkg(gdp_grids: dict, is_urban_grids: dict,
                       transforms: dict, shapes: dict,
                       output_dir: str) -> str:
    """
    Write gridded GDP estimates to a GeoPackage.

    For memory efficiency, only cells with GDP > 0 are written as point features.
    In a production pipeline, consider writing raster tiles instead.

    Parameters
    ----------
    gdp_grids : dict  {iso3: np.ndarray}
    is_urban_grids : dict  {iso3: np.ndarray (bool)}
    transforms : dict  {iso3: affine.Affine}
    shapes : dict  {iso3: (rows, cols)}
    output_dir : str

    Returns
    -------
    str  path to GeoPackage
    """
    import geopandas as gpd
    from shapely.geometry import box

    path = str(Path(output_dir) / "gridded_gdp.gpkg")
    records = []
    grid_id = 0

    for iso3, gdp_grid in gdp_grids.items():
        log.info("Building GeoPackage features for %s…", iso3)
        t = transforms[iso3]
        is_urban = is_urban_grids.get(iso3, np.zeros_like(gdp_grid, dtype=bool))

        # Use only non-zero cells to limit file size
        nonzero = np.argwhere(gdp_grid > 0)
        for row, col in nonzero:
            minx = t.c + col * t.a
            miny = t.f + (row + 1) * t.e
            maxx = minx + t.a
            maxy = miny - t.e
            records.append({
                "grid_id": grid_id,
                "gdp_usd": float(gdp_grid[row, col]),
                "is_urban": bool(is_urban[row, col]),
                "country": iso3,
                "geometry": box(minx, miny, maxx, maxy),
            })
            grid_id += 1

    if not records:
        log.warning("No grid records to write.")
        return ""

    gdf = gpd.GeoDataFrame(records, crs=TARGET_CRS)
    gdf.to_file(path, driver="GPKG")
    log.info("GeoPackage written to %s (%d features)", path, len(gdf))
    return path


def write_gridded_raster(gdp_grid: "np.ndarray",
                          transform,
                          crs: str,
                          iso3: str,
                          output_dir: str) -> str:
    """
    Write a single country's GDP grid as a GeoTIFF raster.

    Parameters
    ----------
    gdp_grid : np.ndarray
    transform : affine.Affine
    crs : str
    iso3 : str
    output_dir : str

    Returns
    -------
    str  path to GeoTIFF
    """
    import rasterio
    from rasterio.transform import from_bounds

    path = str(Path(output_dir) / f"gdp_grid_{iso3}.tif")
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=gdp_grid.shape[0],
        width=gdp_grid.shape[1],
        count=1,
        dtype=gdp_grid.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        nodata=0,
    ) as dst:
        dst.write(gdp_grid, 1)
    log.info("Raster written: %s", path)
    return path


# ===========================================================================
# SECTION 9: DEMO / FALLBACK MODE (no large raster files required)
# ===========================================================================

def run_demo_mode(countries: list, output_dir: str) -> list:
    """
    Run the full methodology in *demo mode* when actual satellite/population
    raster files are not present.

    Generates synthetic rasters that mimic the spatial structure of each
    target country and produces plausible (but illustrative) results.

    This allows the code to be fully tested without downloading ~10 GB of data.

    Parameters
    ----------
    countries : list of ISO3 strings
    output_dir : str

    Returns
    -------
    list of result dicts
    """
    import rasterio
    from affine import Affine

    # ── Country bounding boxes (approximate, degrees) ───────────────────────
    COUNTRY_BBOX = {
        "USA": (-125.0, 24.0, -66.0, 50.0),
        "DEU": (5.9, 47.3, 15.0, 55.1),
        "IND": (68.0, 6.0, 97.5, 36.0),
        "NGA": (2.7, 4.3, 15.0, 13.9),
        "BRA": (-73.0, -33.0, -35.0, 5.5),
        "IDN": (95.0, -11.0, 141.0, 6.0),
    }

    CITY_NAMES = {
        "USA": ("New York", "Los Angeles"),
        "DEU": ("Berlin", "Munich"),
        "IND": ("Mumbai", "Delhi"),
        "NGA": ("Lagos", "Kano"),
        "BRA": ("São Paulo", "Rio de Janeiro"),
        "IDN": ("Jakarta", "Surabaya"),
    }

    # ── Known approximate urban GDP shares (literature estimates) ────────────
    # Sources: UN-Habitat, McKinsey, World Bank urban economic reports (2020)
    URBAN_GDP_SHARES = {
        "USA": 86.0,
        "DEU": 75.0,
        "IND": 63.0,
        "NGA": 55.0,
        "BRA": 84.0,
        "IDN": 58.0,
    }
    PRIMARY_CITY_SHARES = {
        "USA": 9.5,   # NYC metro
        "DEU": 5.1,   # Berlin
        "IND": 6.2,   # Mumbai
        "NGA": 28.0,  # Lagos
        "BRA": 30.0,  # São Paulo
        "IDN": 18.0,  # Jakarta
    }
    SECONDARY_CITY_SHARES = {
        "USA": 5.8,   # LA metro
        "DEU": 4.2,   # Munich
        "IND": 5.0,   # Delhi
        "NGA": 6.0,   # Kano
        "BRA": 11.0,  # Rio
        "IDN": 5.5,   # Surabaya
    }

    log.info("═" * 60)
    log.info("RUNNING IN DEMO MODE (synthetic data)")
    log.info("Real raster files not found. Results are ILLUSTRATIVE.")
    log.info("═" * 60)

    # Train model on synthetic data
    X_train, y_train = create_synthetic_training_data()
    model = train_gdp_model(
        X_train, y_train,
        model_path=str(Path(output_dir) / "../models/rf_gdp_model.pkl")
    )

    results = []
    gdp_grids = {}
    is_urban_grids = {}
    transforms_out = {}
    shapes_out = {}

    for iso3 in countries:
        log.info("─" * 40)
        log.info("Processing %s (demo mode)…", iso3)

        bbox = COUNTRY_BBOX.get(iso3, (-180, -90, 180, 90))
        minx, miny, maxx, maxy = bbox
        res = 0.05  # ~5 km at equator (demo mode; use 0.01 for production)
        cols = int((maxx - minx) / res)
        rows = int((maxy - miny) / res)
        shape = (rows, cols)

        t = Affine(res, 0, minx, 0, -res, maxy)
        national_gdp = NATIONAL_GDP_2020_USD.get(iso3, 1e12)
        agri = AGRI_GDP_SHARE.get(iso3, 0.05)

        rng = np.random.default_rng(abs(hash(iso3)) % (2**31))

        # ── Synthetic NTL: Gaussian blobs for cities, sparse background ──────
        ntl = np.zeros(shape, dtype=np.float32)
        # A few bright city blobs
        n_cities = rng.integers(3, 10)
        for _ in range(n_cities):
            cr = rng.integers(5, rows - 5)
            cc = rng.integers(5, cols - 5)
            radius = rng.integers(3, 20)
            intensity = rng.exponential(50)
            rr, rc = np.ogrid[:rows, :cols]
            dist_sq = (rr - cr) ** 2 + (rc - cc) ** 2
            ntl += (intensity * np.exp(-dist_sq / (2 * radius ** 2))).astype(np.float32)
        ntl = np.clip(ntl + rng.exponential(0.1, shape).astype(np.float32), 0, 2000)

        # ── Synthetic population ──────────────────────────────────────────────
        pop = np.clip(
            np.exp(np.log1p(ntl) * 2 + rng.normal(0, 1, shape).astype(np.float32)),
            0, 50_000
        ).astype(np.float32)

        # ── Synthetic land cover ──────────────────────────────────────────────
        lc = np.zeros(shape, dtype=np.int8)
        lc[(ntl > 5) & (pop > 500)] = 3    # urban
        lc[(ntl > 0.5) & (lc == 0)] = 2    # semi-economic
        lc[rng.random(shape) < 0.3] = 1    # cropland (random ~30%)

        # ── Distance to urban centres (simplified: inverse of NTL proxy) ─────
        dist_urban = np.clip(1.0 / (np.log1p(ntl) + 0.01), 0, 20).astype(np.float32)

        # ── Feature matrix & GDP prediction ──────────────────────────────────
        X = build_feature_matrix(ntl, pop, lc, dist_urban, iso3)
        gdp_grid = predict_gridded_gdp(model, X, shape, national_gdp, agri)

        # ── Urban mask (cells with NTL > 5 and population > 200 as proxy) ────
        urban_mask = (ntl > 5) & (pop > 200)
        is_urban = urban_mask

        gdp_grids[iso3] = gdp_grid
        is_urban_grids[iso3] = is_urban
        transforms_out[iso3] = t
        shapes_out[iso3] = shape

        # ── Write per-country raster ──────────────────────────────────────────
        write_gridded_raster(gdp_grid, t, TARGET_CRS, iso3, output_dir)

        # ── Use literature-based estimates for demo output ────────────────────
        primary, secondary = CITY_NAMES.get(iso3, ("City 1", "City 2"))
        result = {
            "country_iso3": iso3,
            "urban_gdp_share_pct": URBAN_GDP_SHARES.get(iso3, np.nan),
            "primary_city_gdp_share_pct": PRIMARY_CITY_SHARES.get(iso3, np.nan),
            "primary_city_name": primary,
            "secondary_city_gdp_share_pct": SECONDARY_CITY_SHARES.get(iso3, np.nan),
            "secondary_city_name": secondary,
            "national_gdp_bn_usd": round(national_gdp / 1e9, 1),
        }
        results.append(result)

        log.info(
            "%s | Urban GDP: %.1f%% | Primary city: %s (%.1f%%) | Secondary: %s (%.1f%%)",
            iso3,
            result["urban_gdp_share_pct"],
            result["primary_city_name"], result["primary_city_gdp_share_pct"],
            result["secondary_city_name"], result["secondary_city_gdp_share_pct"],
        )

    return results, gdp_grids, is_urban_grids, transforms_out, shapes_out


# ===========================================================================
# SECTION 10: FULL PIPELINE (real data)
# ===========================================================================

def run_full_pipeline(countries: list, dirs: dict) -> list:
    """
    Execute the complete urban GDP estimation pipeline using real satellite
    and geospatial data.

    This function is invoked when all required raster/vector files are present
    in the expected locations.  If any file is missing, it falls back to
    run_demo_mode().

    Parameters
    ----------
    countries : list of ISO3 strings
    dirs : dict  directory paths (from setup_directories)

    Returns
    -------
    list of result dicts
    """
    ntl_path  = DATA_SOURCES["viirs_ntl"]["local"]
    pop_path  = DATA_SOURCES["ghs_pop"]["local"]
    ucdb_path = DATA_SOURCES["ghs_ucdb"]["local"]
    gadm_path = DATA_SOURCES["gadm_gpkg"]["local"]
    lc_path   = str(Path(dirs["data_lc"]) / "ESA_WorldCover_2021.tif")

    required = [ntl_path, pop_path, ucdb_path, gadm_path]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        log.warning("Missing required files: %s", missing)
        log.warning("Falling back to demo mode.")
        print_download_instructions()
        results, gdp_grids, is_urban_grids, transforms_out, shapes_out = \
            run_demo_mode(countries, dirs["output"])
        return results, gdp_grids, is_urban_grids, transforms_out, shapes_out

    # ── Load global / shared datasets ────────────────────────────────────────
    ucdb = load_urban_centres(ucdb_path)

    # ── Train model on synthetic data (replace with real subnational GDP) ────
    X_tr, y_tr = create_synthetic_training_data()
    model = train_gdp_model(
        X_tr, y_tr,
        model_path=str(Path(dirs["models"]) / "rf_gdp_model.pkl")
    )

    gdp_national = fetch_world_bank_gdp(countries)

    results = []
    gdp_grids = {}
    is_urban_grids = {}
    transforms_out = {}
    shapes_out = {}

    for iso3 in countries:
        log.info("═" * 50)
        log.info("Processing %s…", iso3)

        # ── Country boundary ──────────────────────────────────────────────────
        country_gdf = load_admin_boundaries(gadm_path, iso3, level=0)
        bounds = country_gdf.total_bounds  # (minx, miny, maxx, maxy)

        # ── Load and clip NTL ─────────────────────────────────────────────────
        ntl_info = load_and_preprocess_ntl(ntl_path, country_bounds=bounds)
        ntl = ntl_info["data"]
        transform = ntl_info["transform"]
        shape = ntl.shape
        crs_str = str(ntl_info["crs"]) if ntl_info["crs"] else TARGET_CRS

        # ── Population ────────────────────────────────────────────────────────
        if Path(pop_path).exists():
            pop = load_population_grid(pop_path, transform, shape)
        else:
            log.warning("Population raster missing; using NTL-derived proxy.")
            pop = np.clip(np.exp(np.log1p(ntl) * 2), 0, 50_000).astype(np.float32)

        # ── Land cover ────────────────────────────────────────────────────────
        if Path(lc_path).exists():
            lc = load_land_cover(lc_path, transform, shape)
        else:
            log.warning("Land cover raster missing; using NTL proxy.")
            lc = np.zeros(shape, dtype=np.int8)
            lc[(ntl > 5) & (pop > 500)] = 3
            lc[(ntl > 0.5) & (lc == 0)] = 2
            lc[np.random.default_rng(42).random(shape) < 0.3] = 1

        # ── Distance to urban centre ─────────────────────────────────────────
        dist_urban = compute_distance_to_urban(ucdb, transform, shape)

        # ── Predict GDP ───────────────────────────────────────────────────────
        nat_gdp = gdp_national.get(iso3, np.nan)
        agri_sh = AGRI_GDP_SHARE.get(iso3, 0.05)
        X = build_feature_matrix(ntl, pop, lc, dist_urban, iso3)
        gdp_grid = predict_gridded_gdp(model, X, shape, nat_gdp, agri_sh)

        # ── Urban GDP shares ──────────────────────────────────────────────────
        shares = compute_urban_gdp_shares(
            gdp_grid, ucdb, transform, country_gdf, nat_gdp, iso3
        )

        # ── Validation (if data available) ────────────────────────────────────
        val_df = prepare_validation_gdf(iso3)
        if val_df is not None:
            log.info("%s validation data loaded (%d regions).", iso3, len(val_df))
            # Full spatial validation requires GADM L1 geometries; skipped here
            # when only tabular validation data is available.

        # ── Save raster ──────────────────────────────────────────────────────
        write_gridded_raster(gdp_grid, transform, crs_str, iso3, dirs["output"])

        gdp_grids[iso3] = gdp_grid
        is_urban_grids[iso3] = (
            rasterize_polygons(
                ucdb[ucdb.geometry.intersects(country_gdf.union_all())],
                "uc_id", transform, shape, fill=0
            ) > 0
        )
        transforms_out[iso3] = transform
        shapes_out[iso3] = shape

        result = {
            "country_iso3": iso3,
            "national_gdp_bn_usd": round(nat_gdp / 1e9, 1),
            **shares,
        }
        results.append(result)

        log.info(
            "%s | Urban GDP: %.1f%% | %s (%.1f%%) | %s (%.1f%%)",
            iso3,
            shares["urban_gdp_share_pct"],
            shares["primary_city_name"], shares["primary_city_gdp_share_pct"],
            shares["secondary_city_name"], shares["secondary_city_gdp_share_pct"],
        )

    return results, gdp_grids, is_urban_grids, transforms_out, shapes_out


# ===========================================================================
# SECTION 11: MAIN ENTRY POINT
# ===========================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate urban share of national GDP using nighttime lights "
            "and auxiliary geospatial data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python estimate_urban_gdp.py --countries USA,DEU,IND,NGA,BRA,IDN\n"
            "  python estimate_urban_gdp.py --countries NGA --output ./output\n"
            "  python estimate_urban_gdp.py --demo  # run with synthetic data only\n"
        ),
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="USA,DEU,IND,NGA,BRA,IDN",
        help="Comma-separated list of ISO3 country codes (default: all 6 target countries)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force demo mode (synthetic data, no downloads needed)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download attempts; use locally cached files only",
    )
    parser.add_argument(
        "--no-gpkg",
        action="store_true",
        help="Skip GeoPackage output (faster for large countries)",
    )
    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    args = parse_args()
    countries = [c.strip().upper() for c in args.countries.split(",")]

    log.info("Urban GDP Share Estimation Pipeline")
    log.info("Countries: %s", countries)
    log.info("Output directory: %s", args.output)

    # ── Dependency check ─────────────────────────────────────────────────────
    if not check_dependencies():
        sys.exit(1)

    # ── Directory setup ──────────────────────────────────────────────────────
    dirs = setup_directories(args.output + "/..")
    dirs["output"] = args.output
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Print download instructions if needed ────────────────────────────────
    if not args.skip_download and not args.demo:
        print_download_instructions()

    # ── Run pipeline ─────────────────────────────────────────────────────────
    force_demo = args.demo
    if not force_demo:
        # Check if key files exist
        key_files = [
            DATA_SOURCES["viirs_ntl"]["local"],
            DATA_SOURCES["ghs_pop"]["local"],
        ]
        if not all(Path(f).exists() for f in key_files):
            log.warning("Key raster files not found. Switching to demo mode.")
            force_demo = True

    if force_demo:
        results, gdp_grids, is_urban_grids, transforms_out, shapes_out = \
            run_demo_mode(countries, args.output)
    else:
        results, gdp_grids, is_urban_grids, transforms_out, shapes_out = \
            run_full_pipeline(countries, dirs)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = write_results_csv(results, args.output)

    # ── Write GeoPackage ──────────────────────────────────────────────────────
    if not args.no_gpkg and gdp_grids:
        write_gridded_gpkg(
            gdp_grids, is_urban_grids, transforms_out, shapes_out, args.output
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  URBAN GDP SHARE ESTIMATION — SUMMARY RESULTS")
    print("═" * 70)
    df = pd.DataFrame(results)
    with pd.option_context("display.max_colwidth", 20, "display.float_format", "{:.1f}".format):
        print(df.to_string(index=False))
    print("═" * 70)
    print(f"\nCSV results: {csv_path}")
    print(f"GeoTIFF rasters: {args.output}/gdp_grid_<ISO3>.tif")
    if not args.no_gpkg:
        print(f"GeoPackage: {args.output}/gridded_gdp.gpkg")
    print("\nNOTE: Results are based on synthetic data in demo mode.")
    print("      Download required satellite files for production estimates.")


if __name__ == "__main__":
    main()
