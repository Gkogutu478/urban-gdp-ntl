import pandas as pd
import numpy as np
from pathlib import Path

out_dir = Path(r"D:\urban_gdp_project\output")
shares  = pd.read_csv(out_dir / "urban_gdp_shares_final.csv")
cities  = pd.read_csv(out_dir / "city_gdp_estimates_v3.csv")

anchored = shares.dropna(subset=["urban_gdp_share_pct"])

# ── Table 1: Country-level results ───────────────────────
t1 = anchored[[
    "country_iso3", "country_name", "region",
    "n_cities", "national_gdp_2020_bn", "urban_gdp_share_pct",
    "urban_gdp_2020_bn", "primary_city", "primary_city_gdp_share_pct",
    "secondary_city", "secondary_city_gdp_share_pct",
    "tertiary_city", "tertiary_city_gdp_share_pct"
]].sort_values(["region", "urban_gdp_share_pct"], ascending=[True, False])
t1.to_csv(out_dir / "TABLE1_country_urban_gdp_shares.csv", index=False)
print("Table 1 saved:", len(t1), "countries")

# ── Table 2: Top 50 cities globally ──────────────────────
cities["gdp_bn"]  = (cities["gdp_final_usd"] / 1e9).round(2)
cities["pop_M"]   = (cities["pop_2015"] / 1e6).round(2)
cities["ntl_avg"] = cities["ntl_mean"].round(2)

t2 = cities.nlargest(50, "gdp_final_usd")[[
    "city_name", "iso3", "region", "pop_M", "area_km2", "ntl_avg", "gdp_bn"
]].copy().reset_index(drop=True)
t2.index = t2.index + 1
t2.index.name = "rank"
t2.to_csv(out_dir / "TABLE2_top50_cities_GDP.csv")
print("Table 2 saved: top 50 cities globally")

# ── Table 3: Regional summary ─────────────────────────────
t3 = anchored.groupby("region").agg(
    n_countries=("country_iso3", "count"),
    total_cities=("n_cities", "sum"),
    mean_urban_share=("urban_gdp_share_pct", "mean"),
    median_urban_share=("urban_gdp_share_pct", "median"),
    std_urban_share=("urban_gdp_share_pct", "std"),
    min_urban_share=("urban_gdp_share_pct", "min"),
    max_urban_share=("urban_gdp_share_pct", "max"),
    total_urban_gdp_bn=("urban_gdp_2020_bn", "sum"),
).round(1).sort_values("mean_urban_share", ascending=False).reset_index()
t3.to_csv(out_dir / "TABLE3_regional_summary.csv", index=False)
print("Table 3 saved:", len(t3), "regions")

# ── Table 4: Africa focus ────────────────────────────────
africa_cities = cities[cities["region"] == "Africa"].copy()
top_africa = africa_cities.nlargest(30, "gdp_final_usd")[
    ["city_name", "iso3", "pop_M", "area_km2", "ntl_avg", "gdp_bn"]
].reset_index(drop=True)
top_africa.index = top_africa.index + 1
top_africa.index.name = "rank"
top_africa.to_csv(out_dir / "TABLE4_africa_top30_cities.csv")
print("Table 4 saved: top 30 African cities")

# ── Table 5: East Africa deep-dive (Kenya context) ───────
ea_iso = ["KEN", "TZA", "UGA", "ETH", "RWA", "BDI", "SSD",
          "ERI", "DJI", "SOM", "MOZ", "MDG", "MWI", "ZMB",
          "ZWE", "AGO", "NAM", "BWA", "ZAF"]
ea_cities = cities[cities["iso3"].isin(ea_iso)].copy()
ea_shares = anchored[anchored["country_iso3"].isin(ea_iso)][[
    "country_iso3", "country_name", "n_cities",
    "national_gdp_2020_bn", "urban_gdp_share_pct", "urban_gdp_2020_bn",
    "primary_city", "primary_city_gdp_share_pct",
    "secondary_city", "secondary_city_gdp_share_pct"
]].sort_values("urban_gdp_share_pct", ascending=False)
ea_shares.to_csv(out_dir / "TABLE5_eastern_southern_africa.csv", index=False)
print("Table 5 saved:", len(ea_shares), "Eastern/Southern African countries")

# ── Print previews ────────────────────────────────────────
print()
print("=== TABLE 2 PREVIEW: TOP 20 CITIES GLOBALLY ===")
print(t2.head(20).to_string())

print()
print("=== TABLE 3: REGIONAL SUMMARY ===")
print(t3.to_string(index=False))

print()
print("=== TABLE 5 PREVIEW: EASTERN AND SOUTHERN AFRICA ===")
print(ea_shares[[
    "country_iso3", "country_name", "urban_gdp_share_pct",
    "primary_city", "primary_city_gdp_share_pct"
]].to_string(index=False))

print()
print("=== OUTPUT FILES ===")
for f in sorted(out_dir.glob("TABLE*.csv")):
    size_kb = round(f.stat().st_size / 1024, 1)
    rows    = len(pd.read_csv(f))
    print(f"  {f.name:<45} {rows:>5} rows   {size_kb:>7} KB")