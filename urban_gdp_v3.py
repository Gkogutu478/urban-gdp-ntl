import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── World Bank national GDP 2020 (USD) ───────────────────
NATIONAL_GDP_2020 = {
    'USA': 20_936_600_000_000, 'DEU':  3_846_410_000_000,
    'IND':  2_622_984_000_000, 'NGA':    432_294_000_000,
    'BRA':  1_444_733_000_000, 'IDN':  1_058_688_000_000,
    'CHN': 14_722_800_000_000, 'GBR':  2_764_198_000_000,
    'FRA':  2_707_074_000_000, 'JPN':  5_057_759_000_000,
    'KEN':    101_014_000_000, 'ETH':    107_646_000_000,
    'ZAF':    419_015_000_000, 'MEX':  1_076_163_000_000,
    'TUR':    720_105_000_000, 'SAU':    700_118_000_000,
    'ARG':    383_067_000_000, 'PAK':    263_694_000_000,
    'BGD':    324_239_000_000, 'PHL':    361_489_000_000,
    'VNM':    271_158_000_000, 'EGY':    361_876_000_000,
    'COL':    270_158_000_000, 'THA':    499_723_000_000,
    'MYS':    336_664_000_000, 'AGO':     62_309_000_000,
    'GHA':     72_353_000_000, 'TZA':     63_177_000_000,
    'UGA':     37_396_000_000, 'CMR':     40_448_000_000,
    'AUS':  1_330_514_000_000, 'CAN':  1_643_408_000_000,
    'KOR':  1_637_896_000_000, 'ESP':  1_281_485_000_000,
    'ITA':  1_897_462_000_000, 'RUS':  1_483_498_000_000,
    'NLD':    910_045_000_000, 'CHE':    752_248_000_000,
    'POL':    596_075_000_000, 'SWE':    541_220_000_000,
    'BEL':    521_859_000_000, 'NOR':    366_022_000_000,
    'AUT':    433_258_000_000, 'ARE':    421_142_000_000,
    'IRN':    191_735_000_000, 'IRQ':    166_757_000_000,
    'MAR':    114_725_000_000, 'DZA':    145_164_000_000,
    'TUN':     39_361_000_000, 'LBY':     25_413_000_000,
    'SDN':     30_874_000_000, 'MOZ':     14_029_000_000,
    'ZMB':     18_110_000_000, 'ZWE':     18_638_000_000,
    'SEN':     24_910_000_000, 'CIV':     61_345_000_000,
    'MLI':     17_280_000_000, 'BFA':     17_999_000_000,
    'NER':     13_693_000_000, 'TCD':     11_778_000_000,
    'MDG':     13_721_000_000, 'RWA':     10_350_000_000,
    'MWI':      7_665_000_000, 'BEN':     15_653_000_000,
    'GMB':      1_978_000_000, 'SLE':      4_046_000_000,
    'LBR':      3_284_000_000, 'GIN':     15_698_000_000,
    'SOM':      7_627_000_000, 'ERI':      2_065_000_000,
    'COD':     48_994_000_000, 'CAF':      2_321_000_000,
    'GNQ':     10_018_000_000, 'GAB':     15_319_000_000,
    'COG':     12_267_000_000, 'SSD':      4_616_000_000,
    'VEN':     47_261_000_000, 'PER':    202_014_000_000,
    'CHL':    252_938_000_000, 'ECU':    107_436_000_000,
    'BOL':     36_925_000_000, 'PRY':     35_432_000_000,
    'URY':     53_627_000_000, 'GUY':      5_471_000_000,
    'SUR':      3_509_000_000, 'CRI':     61_772_000_000,
    'PAN':     52_938_000_000, 'GTM':     77_026_000_000,
    'HND':     23_826_000_000, 'SLV':     24_639_000_000,
    'NIC':     12_630_000_000, 'DOM':     78_845_000_000,
    'CUB':    107_352_000_000, 'HTI':      8_384_000_000,
    'JAM':     13_812_000_000, 'TTO':     21_693_000_000,
    'BGD':    324_239_000_000, 'LKA':     80_705_000_000,
    'MMR':     76_086_000_000, 'KHM':     26_730_000_000,
    'LAO':     18_748_000_000, 'NPL':     33_659_000_000,
    'AFG':     19_807_000_000, 'UZB':     57_921_000_000,
    'KAZ':    171_082_000_000, 'AZE':     42_607_000_000,
    'TKM':     45_231_000_000, 'GEO':     15_882_000_000,
    'ARM':     12_648_000_000, 'MNG':     13_997_000_000,
    'PRK':     16_000_000_000, 'TWN':    668_533_000_000,
    'HKG':    346_469_000_000, 'SGP':    340_001_000_000,
    'BRN':     12_054_000_000, 'PNG':     25_186_000_000,
    'FJI':      4_426_000_000, 'SLB':        760_000_000,
    'VUT':        914_000_000, 'WSM':        873_000_000,
    'ROU':    249_543_000_000, 'HUN':    155_808_000_000,
    'CZE':    245_349_000_000, 'SVK':    105_172_000_000,
    'BGR':     68_558_000_000, 'SRB':     53_347_000_000,
    'HRV':     57_203_000_000, 'SVN':     53_542_000_000,
    'BIH':     20_164_000_000, 'ALB':     15_279_000_000,
    'MKD':     12_267_000_000, 'MDA':     13_769_000_000,
    'BLR':     60_262_000_000, 'UKR':    155_582_000_000,
    'LTU':     56_461_000_000, 'LVA':     33_514_000_000,
    'EST':     30_618_000_000, 'FIN':    271_165_000_000,
    'DNK':    356_085_000_000, 'PRT':    228_539_000_000,
    'GRC':    188_836_000_000, 'ISR':    407_096_000_000,
    'JOR':     43_757_000_000, 'LBN':     33_376_000_000,
    'SYR':     21_000_000_000, 'YEM':     21_606_000_000,
    'OMN':     76_332_000_000, 'KWT':    105_942_000_000,
    'QAT':    144_425_000_000, 'BHR':     34_569_000_000,
}

print('='*70)
print('  URBAN GDP ESTIMATION v3 — CORRECTED URBAN SHARE CALCULATION')
print('='*70)

# ── Load and clean UCDB ───────────────────────────────────
print('\n[1] Loading UCDB...')
gdf = gpd.read_file(r'D:\urban_gdp_project\data\ghs_ucdb\GHS_UCDB_2019.gpkg')

df = pd.DataFrame({
    'city_name':    gdf['UC_NM_MN'],
    'country_name': gdf['CTR_MN_NM'],
    'iso3':         gdf['CTR_MN_ISO'],
    'region':       gdf['GRGN_L1'],
    'subregion':    gdf['GRGN_L2'],
    'area_km2':     gdf['AREA'],
    'pop_2015':     gdf['P15'],
    'pop_2000':     gdf['P00'],
    'pop_1990':     gdf['P90'],
    'ntl_mean':     gdf['NTL_AV'],
    'gdp_2015_usd': gdf['GDP15_SM'],
    'gdp_2000_usd': gdf['GDP00_SM'],
    'built_2015':   gdf['B15'],
    'built_2000':   gdf['B00'],
})

df = df.dropna(subset=['ntl_mean','pop_2015','gdp_2015_usd','iso3'])
df = df[df['pop_2015'] > 0]
df = df[df['gdp_2015_usd'] > 0]
print(f'    Cities loaded and cleaned: {len(df)}')

# ── Feature engineering (NO GDP-derived features as inputs) 
print('\n[2] Building features (causal only — no GDP leakage)...')
df['pop_density']     = df['pop_2015'] / (df['area_km2'] + 1)
df['built_share']     = df['built_2015'] / (df['area_km2'] + 1)
df['built_per_cap']   = df['built_2015'] / (df['pop_2015'] + 1)
df['log_ntl']         = np.log1p(df['ntl_mean'])
df['log_pop']         = np.log1p(df['pop_2015'])
df['log_area']        = np.log1p(df['area_km2'])
df['log_density']     = np.log1p(df['pop_density'])
df['log_built']       = np.log1p(df['built_share'])
df['ntl_sq']          = df['ntl_mean'] ** 2
df['ntl_x_pop']       = df['ntl_mean'] * df['log_pop']
df['ntl_x_built']     = df['ntl_mean'] * df['built_share']
df['pop_growth_9015'] = df['pop_2015'] / (df['pop_1990'] + 1)
df['pop_growth_0015'] = df['pop_2015'] / (df['pop_2000'] + 1)
df['built_growth']    = df['built_2015'] / (df['built_2000'] + 1)
df['ntl_per_km2']     = df['ntl_mean'] / (df['area_km2'] + 1)

le_iso = LabelEncoder()
le_reg = LabelEncoder()
df['iso3_enc']   = le_iso.fit_transform(df['iso3'].fillna('UNK'))
df['region_enc'] = le_reg.fit_transform(df['region'].fillna('UNK'))

FEATURES = [
    'ntl_mean',   'log_ntl',     'ntl_sq',      'ntl_per_km2',
    'ntl_x_pop',  'ntl_x_built',
    'pop_2015',   'log_pop',     'pop_density',  'log_density',
    'area_km2',   'log_area',
    'built_share','log_built',   'built_per_cap','built_growth',
    'pop_growth_9015', 'pop_growth_0015',
    'iso3_enc',   'region_enc',
]

X = df[FEATURES].replace([np.inf,-np.inf], 0).fillna(0).values.astype(np.float32)
y = np.log1p(df['gdp_2015_usd'].values)

# ── Train model ───────────────────────────────────────────
print('\n[3] Training Random Forest (no GDP leakage)...')
rf = RandomForestRegressor(
    n_estimators=500, max_depth=12,
    min_samples_leaf=5, max_features=0.5,
    n_jobs=-1, random_state=42,
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2', n_jobs=-1)
print(f'    5-fold CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')

rf.fit(X, y)
y_pred_is = rf.predict(X)
print(f'    In-sample R2: {r2_score(y, y_pred_is):.3f}')

fi = pd.Series(rf.feature_importances_, index=FEATURES)
print('\n    Top 8 features:')
for feat, imp in fi.sort_values(ascending=False).head(8).items():
    print(f'      {feat:<22} {imp:.4f}')

# ── Predict GDP share weights per city ────────────────────
print('\n[4] Computing city GDP weights...')
df['gdp_pred_log'] = rf.predict(X)
df['gdp_pred']     = np.expm1(df['gdp_pred_log'])

# ── KEY FIX: compute urban share correctly ────────────────
# Step A: for each country sum the UCDB GDP15 as the urban total
# Step B: divide by national GDP 2020 (scaled by growth) to get share
# Step C: allocate national GDP proportionally to each city

print('\n[5] Computing honest urban GDP shares...')
rows = []
df['gdp_final_usd'] = np.nan

for iso3, grp in df.groupby('iso3'):
    idx        = grp.index
    nat_gdp    = NATIONAL_GDP_2020.get(iso3, np.nan)

    # Sum of UCDB observed GDP for this country (2015 USD)
    ucdb_urban_sum_2015 = grp['gdp_2015_usd'].sum()

    # Approximate national GDP in 2015 using WB 2020 / 1.15 average growth
    if not np.isnan(nat_gdp):
        nat_gdp_2015  = nat_gdp / 1.15
        # Urban share = UCDB city sum / estimated national 2015 GDP
        urban_share   = min(ucdb_urban_sum_2015 / nat_gdp_2015, 0.98)
        # Urban GDP in 2020 = urban share x national GDP 2020
        urban_gdp_2020 = urban_share * nat_gdp
        # Distribute among cities proportionally to predicted weights
        total_w = grp['gdp_pred'].sum()
        if total_w > 0:
            df.loc[idx, 'gdp_final_usd'] = grp['gdp_pred'] / total_w * urban_gdp_2020
    else:
        # No WB anchor: use scaled predictions directly
        df.loc[idx, 'gdp_final_usd'] = grp['gdp_pred']
        urban_share  = np.nan
        nat_gdp      = np.nan
        urban_gdp_2020 = grp['gdp_pred'].sum()

    ranked = grp.sort_values('gdp_pred', ascending=False).reset_index(drop=True)
    p = ranked.iloc[0] if len(ranked) > 0 else None
    s = ranked.iloc[1] if len(ranked) > 1 else None
    t = ranked.iloc[2] if len(ranked) > 2 else None

    def city_share(city_row):
        if city_row is None or np.isnan(nat_gdp):
            return np.nan
        w = city_row['gdp_pred'] / grp['gdp_pred'].sum()
        return round(w * urban_share * 100, 2)

    rows.append({
        'country_iso3':                 iso3,
        'country_name':                 grp['country_name'].iloc[0],
        'region':                       grp['region'].iloc[0],
        'n_cities':                     len(grp),
        'national_gdp_2020_bn':         round(nat_gdp/1e9,1) if not np.isnan(nat_gdp) else np.nan,
        'urban_gdp_share_pct':          round(urban_share*100, 1) if not np.isnan(urban_share) else np.nan,
        'urban_gdp_2020_bn':            round(urban_gdp_2020/1e9, 1),
        'primary_city':                 p['city_name'] if p is not None else 'N/A',
        'primary_city_gdp_share_pct':   city_share(p),
        'secondary_city':               s['city_name'] if s is not None else 'N/A',
        'secondary_city_gdp_share_pct': city_share(s),
        'tertiary_city':                t['city_name'] if t is not None else 'N/A',
        'tertiary_city_gdp_share_pct':  city_share(t),
    })

shares = pd.DataFrame(rows)

# ── Save outputs ──────────────────────────────────────────
out_dir = Path(r'D:\urban_gdp_project\output')
out_dir.mkdir(exist_ok=True)

city_cols = ['city_name','country_name','iso3','region','subregion',
             'pop_2015','area_km2','ntl_mean','gdp_2015_usd','gdp_final_usd']
df[city_cols].sort_values('gdp_final_usd', ascending=False).to_csv(
    out_dir / 'city_gdp_estimates_v3.csv', index=False)
shares.to_csv(out_dir / 'urban_gdp_shares_v3.csv', index=False)

# ── Print results ─────────────────────────────────────────
anchored = shares.dropna(subset=['urban_gdp_share_pct']).sort_values(
    'urban_gdp_share_pct', ascending=False)

print()
print('='*80)
print('  RESULTS — COUNTRIES WITH WORLD BANK ANCHOR')
print('='*80)
print(anchored[[
    'country_iso3','n_cities','urban_gdp_share_pct',
    'primary_city','primary_city_gdp_share_pct',
    'secondary_city','secondary_city_gdp_share_pct'
]].to_string(index=False))

print()
print('='*70)
print('  REGIONAL SUMMARY')
print('='*70)
reg = anchored.groupby('region').agg(
    countries=('country_iso3','count'),
    avg_urban_share=('urban_gdp_share_pct','mean'),
    avg_n_cities=('n_cities','mean')
).round(1).sort_values('avg_urban_share', ascending=False)
print(reg.to_string())

print()
n_anchored = len(anchored)
mean_share = anchored['urban_gdp_share_pct'].mean()
total_urban = df['gdp_final_usd'].sum() / 1e12
print(f'Countries with WB anchor:    {n_anchored}')
print(f'Mean urban GDP share:        {mean_share:.1f}%')
print(f'Global urban GDP estimated:  {total_urban:.2f} trillion USD')
print(f'City estimates: {out_dir}/city_gdp_estimates_v3.csv')
print(f'Country shares: {out_dir}/urban_gdp_shares_v3.csv')
