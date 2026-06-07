import pandas as pd
import numpy as np
from pathlib import Path

out_dir = Path(r"D:\urban_gdp_project\output")
shares  = pd.read_csv(out_dir / "urban_gdp_shares_v3.csv")
cities  = pd.read_csv(out_dir / "city_gdp_estimates_v3.csv")

EXTRA_GDP = {
    "IRL": 425889000000, "NZL": 212482000000, "CYP":  24688000000,
    "ISL":  21721000000, "LUX":  73429000000, "BWA":  17821000000,
    "MNE":   5542000000, "NAM":  10674000000, "MLT":  14989000000,
    "TGO":   7607000000, "MRT":   8978000000, "KGZ":   7732000000,
    "TJK":   8196000000, "DJI":   3375000000, "PSE":  15556000000,
    "LSO":   2517000000, "SWZ":   4451000000, "BDI":   3112000000,
    "BTN":   2530000000, "COM":   1178000000, "CPV":   1917000000,
    "GNB":   1504000000, "MUS":  11000000000, "TLS":   1984000000,
}

filled = 0
for iso3, nat_gdp in EXTRA_GDP.items():
    mask = shares["country_iso3"] == iso3
    if mask.any():
        city_rows   = cities[cities["iso3"] == iso3]
        urban_2015  = city_rows["gdp_2015_usd"].sum()
        nat_2015    = nat_gdp / 1.15
        urban_share = min(urban_2015 / nat_2015, 0.98)
        shares.loc[mask, "national_gdp_2020_bn"] = round(nat_gdp/1e9, 1)
        shares.loc[mask, "urban_gdp_share_pct"]  = round(urban_share*100, 1)
        shares.loc[mask, "urban_gdp_2020_bn"]    = round(urban_share*nat_gdp/1e9, 1)
        filled += 1

shares.to_csv(out_dir / "urban_gdp_shares_final.csv", index=False)
print("Filled", filled, "additional countries")

n_anchored = shares["national_gdp_2020_bn"].notna().sum()
print("Total countries with WB anchor:", n_anchored)

anchored = shares.dropna(subset=["urban_gdp_share_pct"]).sort_values(
    "urban_gdp_share_pct", ascending=False)

print()
print("=== FINAL REGIONAL SUMMARY ===")
reg = anchored.groupby("region").agg(
    n_countries=("country_iso3", "count"),
    mean_share=("urban_gdp_share_pct", "mean"),
    median_share=("urban_gdp_share_pct", "median"),
    min_share=("urban_gdp_share_pct", "min"),
    max_share=("urban_gdp_share_pct", "max"),
).round(1).sort_values("mean_share", ascending=False)
print(reg.to_string())

print()
print("=== GLOBAL STATISTICS ===")
mean_share  = anchored["urban_gdp_share_pct"].mean()
med_share   = anchored["urban_gdp_share_pct"].median()
n_above_80  = (anchored["urban_gdp_share_pct"] >= 80).sum()
n_below_30  = (anchored["urban_gdp_share_pct"] < 30).sum()
total_urban = anchored["urban_gdp_2020_bn"].sum()
print("Countries analysed:        ", len(anchored))
print("Mean urban GDP share:      ", round(mean_share, 1))
print("Median urban GDP share:    ", round(med_share, 1))
print("Countries above 80% urban: ", n_above_80)
print("Countries below 30% urban: ", n_below_30)
total_str = str(round(total_urban, 0))
print("Total urban GDP (bn USD):  ", total_str)

print()
print("=== TOP 10 MOST URBANISED ===")
cols = ["country_iso3", "country_name", "urban_gdp_share_pct",
        "primary_city", "primary_city_gdp_share_pct"]
print(anchored[cols].head(10).to_string(index=False))

print()
print("=== TOP 10 LEAST URBANISED ===")
print(anchored[cols].tail(10).to_string(index=False))

print()
print("Final output saved to:", out_dir / "urban_gdp_shares_final.csv")