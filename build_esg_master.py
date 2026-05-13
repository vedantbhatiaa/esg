"""
build_esg_master.py  —  ESG Master Database Builder
=====================================================
Pipeline: CONSOLIDATED Excel  →  data_storage/master/

Run:  python build_esg_master.py
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# ── Folder structure ──────────────────────────────────────────────────────────
BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR  = BASE_DIR / "data_storage" / "master"
MEMBERS_TIP = BASE_DIR / "data_storage" / "members" / "TIP"
MEMBERS_NON = BASE_DIR / "data_storage" / "members" / "non_TIP"
VERSIONS_DIR= BASE_DIR / "data_storage" / "versions"
REPORTS_TIP = BASE_DIR / "data_storage" / "reports" / "TIP"
REPORTS_NON = BASE_DIR / "data_storage" / "reports" / "non_TIP"

for d in [MASTER_DIR, MEMBERS_TIP, MEMBERS_NON, VERSIONS_DIR, REPORTS_TIP, REPORTS_NON]:
    d.mkdir(parents=True, exist_ok=True)

# ── Locate source file ────────────────────────────────────────────────────────
FILENAME    = "CONSOLIDATED_DUMMY_2009_2023.xlsx"
SOURCE_FILE = MASTER_DIR / FILENAME
if not SOURCE_FILE.exists():
    fallback = BASE_DIR / FILENAME
    if fallback.exists():
        shutil.copy(fallback, SOURCE_FILE)
        print(f"[INFO] Copied {FILENAME} → data_storage/master/")
    else:
        print(f"[ERROR] Cannot find '{FILENAME}'. Place it in the project root or data_storage/master/")
        raise SystemExit(1)

# ── Step 1: Load ──────────────────────────────────────────────────────────────
print("[1/6] Loading consolidated data...")
raw = pd.read_excel(SOURCE_FILE, sheet_name="Raw Dummy data", header=0)
raw.columns = raw.columns.str.strip()
raw = raw.dropna(subset=["Row_Label"]).copy()
raw["Year"] = raw["Year"].astype(int)
raw["Data"] = pd.to_numeric(raw["Data"], errors="coerce")

COMPANIES = raw["Company"].unique().tolist()
YEARS     = sorted(raw["Year"].unique().tolist())
print(f"    → {len(COMPANIES)} companies | {YEARS[0]}–{YEARS[-1]} | {raw['Row_Label'].nunique()} KPI fields")

# ── Step 2: Pivot ─────────────────────────────────────────────────────────────
print("[2/6] Pivoting to wide format (Company × Year)...")
wide = raw.pivot_table(index=["Company","Year"], columns="Row_Label", values="Data", aggfunc="first").reset_index()
wide.columns.name = None

ordered = [
    "Company","Year","Total no. of sites","ISO 14001 sites","% certified sites",
    "Production","Water intake","Water intake - KPI","Total Electricity",
    "Renewable Electricity Purchased","Non-Renewable Electricity Purchased",
    "Self-generated AND consumed electricity on-site","Purchased Steam",
    "Sold Electricity","Sold Steam","Natural Gas","Coal","Propane","Fuel Oil",
    "Diesel","Petrol","Biomass","Waste tires","LPG","Other",
    "Total energy","Total energy - KPI","Total CO2 - Scope 1","Total CO2 - Scope 2",
    "Total CO2","Total CO2 - KPI",
]
ordered = [c for c in ordered if c in wide.columns]
wide = wide[ordered]
print(f"    → {wide.shape[0]} rows × {wide.shape[1]} columns")

# ── Step 3: Derived KPIs ──────────────────────────────────────────────────────
print("[3/6] Engineering derived KPIs...")
df = wide.copy()

df["Renewable_Electricity_Share_%"] = (
    df["Renewable Electricity Purchased"] / df["Total Electricity"].replace(0, np.nan) * 100).round(4)
df["Scope1_Share_%"] = (
    df["Total CO2 - Scope 1"] / df["Total CO2"].replace(0, np.nan) * 100).round(4)
df["Scope2_Share_%"] = (
    df["Total CO2 - Scope 2"] / df["Total CO2"].replace(0, np.nan) * 100).round(4)

fuel_cols = ["Natural Gas","Coal","Propane","Fuel Oil","Diesel","Petrol","Biomass","Waste tires","LPG","Other"]
df["Fossil_Energy_Share_%"] = (
    df[[c for c in fuel_cols if c in df.columns]].sum(axis=1, min_count=1)
    / df["Total energy"].replace(0, np.nan) * 100).round(4)

df["Water_per_ton"]   = (df["Water intake"]  / df["Production"].replace(0, np.nan)).round(4)
df["CO2_per_ton"]     = (df["Total CO2"]     / df["Production"].replace(0, np.nan)).round(4)
df["Energy_per_ton"]  = (df["Total energy"]  / df["Production"].replace(0, np.nan)).round(4)
df["ISO_Certification_%"] = (df["% certified sites"] * 100).round(2)

print(f"    → Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 4: Save master outputs ───────────────────────────────────────────────
print("[4/6] Saving master files...")

long_path = MASTER_DIR / "ESG_LONG_ALL_COMPANIES_2009_2023.csv"
raw.to_csv(long_path, index=False)
print(f"    → LONG CSV:         {long_path}")

wide_csv = MASTER_DIR / "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"
df.to_csv(wide_csv, index=False)
print(f"    → WIDE CSV:         {wide_csv}")

wide_xlsx = MASTER_DIR / "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.xlsx"
with pd.ExcelWriter(wide_xlsx, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All_Companies", index=False)
    for company in COMPANIES:
        subset = df[df["Company"] == company].reset_index(drop=True)
        sheet  = company.replace(" ", "_")[:31]
        subset.to_excel(writer, sheet_name=sheet, index=False)
print(f"    → WIDE Excel:       {wide_xlsx}")

sector = df.groupby("Year").agg(
    n_companies=("Company","nunique"), Total_Production=("Production","sum"),
    Total_Energy=("Total energy","sum"), Total_CO2=("Total CO2","sum"),
    Total_Water=("Water intake","sum"), Avg_Energy_KPI=("Total energy - KPI","mean"),
    Avg_CO2_KPI=("Total CO2 - KPI","mean"), Avg_Water_KPI=("Water intake - KPI","mean"),
    Avg_Renewable_Share=("Renewable_Electricity_Share_%","mean"),
    Avg_ISO_Cert=("ISO_Certification_%","mean"),
).reset_index()
sector_path = MASTER_DIR / "ESG_SECTOR_AGGREGATED_2009_2023.csv"
sector.to_csv(sector_path, index=False)
print(f"    → SECTOR CSV:       {sector_path}")

# ── Step 5: Per-company files in members/TIP/ ─────────────────────────────────
print("[5/6] Writing per-company files to members/TIP/...")
for company in COMPANIES:
    co_folder = MEMBERS_TIP / company.replace(" ", "_")
    co_folder.mkdir(parents=True, exist_ok=True)
    co_df = df[df["Company"] == company].reset_index(drop=True)
    co_df.to_csv(co_folder / f"{company.replace(' ','_')}_latest.csv", index=False)
    (REPORTS_TIP / company.replace(" ", "_")).mkdir(parents=True, exist_ok=True)
print(f"    → {len(COMPANIES)} company folders created in members/TIP/")

# ── Step 6: Clean up any legacy paths ────────────────────────────────────────
print("[6/6] Cleaning up legacy paths / duplicate columns...")
MASTER_COLS = [
    "Company","Year","Total no. of sites","ISO 14001 sites","% certified sites",
    "Production","Water intake","Water intake - KPI","Total Electricity",
    "Renewable Electricity Purchased","Non-Renewable Electricity Purchased",
    "Self-generated AND consumed electricity on-site","Purchased Steam",
    "Sold Electricity","Sold Steam","Natural Gas","Coal","Propane","Fuel Oil",
    "Diesel","Petrol","Biomass","Waste tires","LPG","Other",
    "Total energy","Total energy - KPI","Total CO2 - Scope 1","Total CO2 - Scope 2",
    "Total CO2","Total CO2 - KPI","Renewable_Electricity_Share_%","Scope1_Share_%",
    "Scope2_Share_%","Fossil_Energy_Share_%","Water_per_ton","CO2_per_ton",
    "Energy_per_ton","ISO_Certification_%",
]
_dirty = pd.read_csv(wide_csv)
_before = len(_dirty.columns)
_clean  = [c for c in MASTER_COLS if c in _dirty.columns]
_dropped = _before - len(_clean)
if _dropped > 0:
    _dirty[_clean].to_csv(wide_csv, index=False)
    print(f"    Removed {_dropped} legacy duplicate columns from master CSV.")
else:
    print("    Master CSV is clean — no legacy columns found.")

print("\n✅ Master database ready.")
print(f"   Master files:  {MASTER_DIR}")
print(f"   Member files:  {MEMBERS_TIP}")
print(f"   Version files: {VERSIONS_DIR}")
print('\n   Load in Streamlit with:')
print('   df = pd.read_csv("data_storage/master/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv")')

# ── Step 7: TIP members aggregate in members/TIP/ (not in company subfolder) ─
print("[7/7] Writing TIP members aggregate to members/TIP/...")

# Wide master — all TIP companies, all years, all columns including derived KPIs
tip_wide = MEMBERS_TIP / "ESG_MASTER_WIDE_TIP_MEMBERS_2009_2023.csv"
df.to_csv(tip_wide, index=False)
print(f"    → TIP wide CSV:       {tip_wide}")

# Consolidated long format — one row per KPI per company per year
tip_long = MEMBERS_TIP / "ESG_CONSOLIDATED_TIP_MEMBERS_2009_2023.csv"
raw.to_csv(tip_long, index=False)
print(f"    → TIP consolidated:   {tip_long}")

print(f"\n✅ TIP member files ready in {MEMBERS_TIP}")