import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import geopandas as gpd
from pathlib import Path

out_dir = Path(r"D:\urban_gdp_project\output")
shares  = pd.read_csv(out_dir / "urban_gdp_shares_final.csv")
cities  = pd.read_csv(out_dir / "city_gdp_estimates_v3.csv")

# Use geopandas built-in world map — no GADM download needed
print("Loading world boundaries from geopandas built-in dataset...")
world = gpd.read_file(r"D:\urban_gdp_project\data\naturalearth\ne_110m_admin_0_countries.shp")
world = world.rename(columns={"ISO_A3": "country_iso3"})
print("World polygons loaded:", len(world))

merged = world.merge(
    shares[["country_iso3", "urban_gdp_share_pct",
            "urban_gdp_2020_bn", "national_gdp_2020_bn",
            "region", "primary_city"]],
    on="country_iso3", how="left"
)
print("Countries matched:", merged["urban_gdp_share_pct"].notna().sum())

# Get city coordinates from UCDB
print("Loading city coordinates from UCDB...")
ucdb = gpd.read_file(
    r"D:\urban_gdp_project\data\ghs_ucdb\GHS_UCDB_2019.gpkg"
)
coords = ucdb[["UC_NM_MN", "CTR_MN_ISO", "GCPNT_LON", "GCPNT_LAT"]].copy()
coords = coords.rename(columns={
    "UC_NM_MN": "city_name",
    "CTR_MN_ISO": "iso3",
    "GCPNT_LON": "lon",
    "GCPNT_LAT": "lat"
})
cities = cities.merge(coords, on=["city_name", "iso3"], how="left")
top30 = cities.nlargest(30, "gdp_final_usd").dropna(subset=["lon", "lat"])
print("Top 30 cities with coordinates:", len(top30))

# ── FIGURE 1: Global choropleth ───────────────────────────
print("Generating Figure 1 — global choropleth...")
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
merged.plot(
    column="urban_gdp_share_pct",
    ax=ax,
    cmap="YlOrRd",
    missing_kwds={"color": "#d4d4d4", "label": "No data"},
    legend=True,
    legend_kwds={
        "label": "Urban GDP share (%)",
        "orientation": "horizontal",
        "fraction": 0.025,
        "pad": 0.04,
        "shrink": 0.5,
    },
    vmin=0,
    vmax=100,
    edgecolor="white",
    linewidth=0.3,
)
max_gdp = top30["gdp_final_usd"].max()
for _, row in top30.iterrows():
    size = (row["gdp_final_usd"] / max_gdp * 150) + 15
    ax.scatter(row["lon"], row["lat"],
               s=size, color="#1a1a2e", alpha=0.8, zorder=5)
dot_handle = mlines.Line2D(
    [], [], color="#1a1a2e", marker="o", linestyle="None",
    markersize=8, alpha=0.8, label="Top 30 cities (sized by GDP)"
)
ax.legend(handles=[dot_handle], fontsize=9,
          loc="lower left", framealpha=0.85)
ax.set_title(
    "Urban Share of National GDP, 2020\n"
    "GHS-UCDB based estimation — 170 countries, 12,818 cities",
    fontsize=15, fontweight="bold", pad=14
)
ax.text(
    0.01, 0.02,
    "Sources: GHS Urban Centre Database 2019, World Bank GDP 2020  |  "
    "Model: Random Forest (CV R2=0.789)",
    transform=ax.transAxes, fontsize=7.5, color="#555555"
)
ax.set_xlim(-180, 180)
ax.set_ylim(-60, 85)
ax.set_axis_off()
plt.tight_layout()
f1 = out_dir / "FIGURE1_global_urban_gdp_share.png"
plt.savefig(f1, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved:", f1)

# ── FIGURE 2: Africa close-up ─────────────────────────────
print("Generating Figure 2 — Africa choropleth...")
africa_iso = shares[shares["region"] == "Africa"]["country_iso3"].tolist()
africa_map = merged[merged["country_iso3"].isin(africa_iso)].copy()
africa_cities = cities[cities["iso3"].isin(africa_iso)].nlargest(
    25, "gdp_final_usd").dropna(subset=["lon", "lat"])

fig, ax = plt.subplots(1, 1, figsize=(11, 13))
africa_map.plot(
    column="urban_gdp_share_pct",
    ax=ax,
    cmap="YlOrRd",
    missing_kwds={"color": "#d4d4d4"},
    legend=True,
    legend_kwds={
        "label": "Urban GDP share (%)",
        "orientation": "vertical",
        "fraction": 0.03,
        "pad": 0.04,
        "shrink": 0.65,
    },
    vmin=0,
    vmax=100,
    edgecolor="white",
    linewidth=0.5,
)
max_gdp_af = africa_cities["gdp_final_usd"].max()
for _, row in africa_cities.iterrows():
    size = (row["gdp_final_usd"] / max_gdp_af * 250) + 20
    ax.scatter(row["lon"], row["lat"],
               s=size, color="#1a1a2e", alpha=0.75, zorder=5)
    ax.annotate(
        row["city_name"],
        (row["lon"], row["lat"]),
        textcoords="offset points",
        xytext=(5, 4),
        fontsize=6.5,
        color="#1a1a2e",
        fontweight="bold",
    )
ax.set_title(
    "Urban Share of National GDP — Africa, 2020",
    fontsize=13, fontweight="bold", pad=10
)
ax.text(
    0.01, 0.01,
    "Sources: GHS-UCDB 2019, World Bank GDP 2020",
    transform=ax.transAxes, fontsize=8, color="#555555"
)
ax.set_xlim(-20, 52)
ax.set_ylim(-36, 38)
ax.set_axis_off()
plt.tight_layout()
f2 = out_dir / "FIGURE2_africa_urban_gdp_share.png"
plt.savefig(f2, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved:", f2)

# ── FIGURE 3: Scatter — urban share vs national GDP ───────
print("Generating Figure 3 — scatter plot...")
plot_df = shares.dropna(
    subset=["urban_gdp_share_pct", "national_gdp_2020_bn"]
).copy()

region_colors = {
    "Asia": "#e63946",
    "Africa": "#2a9d8f",
    "Europe": "#457b9d",
    "Latin America and the Caribbean": "#e9c46a",
    "Northern America": "#264653",
    "Oceania": "#f4a261",
}

fig, ax = plt.subplots(figsize=(13, 8))
for region, grp in plot_df.groupby("region"):
    color = region_colors.get(region, "#aaaaaa")
    ax.scatter(
        grp["national_gdp_2020_bn"],
        grp["urban_gdp_share_pct"],
        c=color, label=region,
        alpha=0.75, s=65,
        edgecolors="white", linewidths=0.5,
    )

label_iso = [
    "USA", "CHN", "JPN", "DEU", "IND", "GBR", "FRA",
    "BRA", "KEN", "NGA", "ZAF", "THA", "ETH", "BGD",
    "IDN", "MEX", "TUR", "KOR", "ARG", "EGY"
]
for _, row in plot_df[plot_df["country_iso3"].isin(label_iso)].iterrows():
    ax.annotate(
        row["country_iso3"],
        (row["national_gdp_2020_bn"], row["urban_gdp_share_pct"]),
        textcoords="offset points",
        xytext=(4, 3),
        fontsize=7.5,
        color="#333333",
    )

ax.set_xscale("log")
ax.axhline(y=56.7, color="#888888", linestyle="--",
           linewidth=1.2, alpha=0.7)
ax.text(
    0.02, 0.955,
    "Global mean: 56.7%  |  Median: 56.4%  |  n=170 countries",
    transform=ax.transAxes, fontsize=9, color="#555555",
    verticalalignment="top"
)
ax.set_xlabel("National GDP 2020 (billion USD, log scale)", fontsize=11)
ax.set_ylabel("Urban GDP share (%)", fontsize=11)
ax.set_title(
    "Urban GDP Concentration vs National Economic Size\n"
    "170 countries, 2020 — GHS-UCDB / World Bank",
    fontsize=13, fontweight="bold"
)
ax.legend(title="Region", fontsize=9, title_fontsize=10,
          loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle="--")
ax.set_ylim(0, 105)
plt.tight_layout()
f3 = out_dir / "FIGURE3_urban_share_vs_gdp_scatter.png"
plt.savefig(f3, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved:", f3)

# ── FIGURE 4: East Africa bar chart ──────────────────────
print("Generating Figure 4 — East Africa bar chart...")
ea_iso = [
    "KEN", "TZA", "UGA", "ETH", "RWA", "BDI",
    "MOZ", "ZMB", "ZAF", "AGO", "MDG", "MWI",
    "ZWE", "NAM", "SOM", "ERI", "SSD", "DJI"
]
ea = shares[shares["country_iso3"].isin(ea_iso)].dropna(
    subset=["urban_gdp_share_pct"]
).sort_values("urban_gdp_share_pct", ascending=True)

bar_colors = [
    "#e63946" if iso == "KEN" else "#2a9d8f"
    for iso in ea["country_iso3"]
]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(
    ea["country_name"],
    ea["urban_gdp_share_pct"],
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5
)
ax.set_xlim(0, 115)
ax.grid(True, axis="x", alpha=0.25, linestyle="--")
plt.tight_layout()
f4 = out_dir / "FIGURE4_east_africa_urban_gdp_bar.png"
plt.savefig(f4, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved:", f4)

print()
print("=== ALL FIGURES COMPLETE ===")
for fname in sorted(out_dir.glob("FIGURE*.png")):
    size_mb = round(fname.stat().st_size / 1e6, 1)
    print(" ", fname.name, size_mb, "MB")
