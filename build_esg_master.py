"""
ESG Master Database Builder
============================
Pipeline: CONSOLIDATED Excel → Wide Master CSV/Excel

Run:  python build_esg_master.py
Output saved to: data/storage/raw/
"""

import os
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
# Script directory — all paths are relative to where this .py file lives
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_OUT     = os.path.join(BASE_DIR, "data_storage", "raw")

os.makedirs(RAW_OUT, exist_ok=True)

# Auto-find the consolidated Excel: check raw folder first, then project root
FILENAME    = "CONSOLIDATED_DUMMY_2009_2023.xlsx"
SOURCE_FILE = os.path.join(RAW_OUT, FILENAME)
if not os.path.exists(SOURCE_FILE):
    fallback = os.path.join(BASE_DIR, FILENAME)
    if os.path.exists(fallback):
        import shutil
        shutil.copy(fallback, SOURCE_FILE)
        print(f"[INFO] Copied {FILENAME} → data_storage/raw/")
    else:
        print(f"[ERROR] Cannot find '{FILENAME}'.")
        print(f"        Place it in either:")
        print(f"          {RAW_OUT}")
        print(f"          {BASE_DIR}")
        raise SystemExit(1)

# ── Step 1: Load raw consolidated data ────────────────────────────────────────
print("[1/5] Loading consolidated data...")
raw = pd.read_excel(SOURCE_FILE, sheet_name="Raw Dummy data", header=0)
raw.columns = raw.columns.str.strip()
raw = raw.dropna(subset=["Row_Label"]).copy()
raw["Year"] = raw["Year"].astype(int)
raw["Data"] = pd.to_numeric(raw["Data"], errors="coerce")

COMPANIES = raw["Company"].unique().tolist()
YEARS     = sorted(raw["Year"].unique().tolist())
print(f"    → {len(COMPANIES)} companies | {YEARS[0]}–{YEARS[-1]} | {raw['Row_Label'].nunique()} KPI fields")

# ── Step 2: Pivot to wide format ──────────────────────────────────────────────
print("[2/5] Pivoting to wide format (Company × Year)...")
wide = raw.pivot_table(
    index=["Company", "Year"],
    columns="Row_Label",
    values="Data",
    aggfunc="first"
).reset_index()
wide.columns.name = None

# Logical column order
ordered = [
    "Company", "Year",
    "Total no. of sites", "ISO 14001 sites", "% certified sites",
    "Production",
    "Water intake", "Water intake - KPI",
    "Total Electricity", "Renewable Electricity Purchased",
    "Non-Renewable Electricity Purchased",
    "Self-generated AND consumed electricity on-site",
    "Purchased Steam", "Sold Electricity", "Sold Steam",
    "Natural Gas", "Coal", "Propane", "Fuel Oil",
    "Diesel", "Petrol", "Biomass", "Waste tires", "LPG", "Other",
    "Total energy", "Total energy - KPI",
    "Total CO2 - Scope 1", "Total CO2 - Scope 2",
    "Total CO2", "Total CO2 - KPI",
]
ordered = [c for c in ordered if c in wide.columns]
wide = wide[ordered]
print(f"    → {wide.shape[0]} rows × {wide.shape[1]} columns")

# ── Step 3: Add derived / formulated KPIs ────────────────────────────────────
print("[3/5] Engineering derived KPIs...")
df = wide.copy()

df["Renewable_Electricity_Share_%"] = (
    df["Renewable Electricity Purchased"] / df["Total Electricity"].replace(0, np.nan) * 100
).round(4)

df["Scope1_Share_%"] = (
    df["Total CO2 - Scope 1"] / df["Total CO2"].replace(0, np.nan) * 100
).round(4)

df["Scope2_Share_%"] = (
    df["Total CO2 - Scope 2"] / df["Total CO2"].replace(0, np.nan) * 100
).round(4)

fuel_cols = ["Natural Gas","Coal","Propane","Fuel Oil","Diesel","Petrol","Biomass","Waste tires","LPG","Other"]
df["Fossil_Energy_Share_%"] = (
    df[[c for c in fuel_cols if c in df.columns]].sum(axis=1, min_count=1)
    / df["Total energy"].replace(0, np.nan) * 100
).round(4)

df["Water_per_ton"]  = (df["Water intake"]  / df["Production"].replace(0, np.nan)).round(4)
df["CO2_per_ton"]    = (df["Total CO2"]      / df["Production"].replace(0, np.nan)).round(4)
df["Energy_per_ton"] = (df["Total energy"]   / df["Production"].replace(0, np.nan)).round(4)
df["ISO_Certification_%"] = (df["% certified sites"] * 100).round(2)

print(f"    → Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 4: Save outputs ──────────────────────────────────────────────────────
print("[4/5] Saving files...")

# Long format (original structure, cleaned)
long_path = os.path.join(RAW_OUT, "ESG_LONG_ALL_COMPANIES_2009_2023.csv")
raw.to_csv(long_path, index=False)
print(f"    → LONG CSV:         {long_path}")

# Wide master CSV
wide_csv = os.path.join(RAW_OUT, "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv")
df.to_csv(wide_csv, index=False)
print(f"    → WIDE CSV:         {wide_csv}")

# Wide master Excel (one sheet per company + all)
wide_xlsx = os.path.join(RAW_OUT, "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.xlsx")
with pd.ExcelWriter(wide_xlsx, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All_Companies", index=False)
    for company in COMPANIES:
        subset = df[df["Company"] == company].reset_index(drop=True)
        sheet  = company.replace(" ", "_")[:31]
        subset.to_excel(writer, sheet_name=sheet, index=False)
print(f"    → WIDE Excel:       {wide_xlsx}")

# Sector-level year aggregation
sector = df.groupby("Year").agg(
    n_companies          = ("Company",               "nunique"),
    Total_Production     = ("Production",            "sum"),
    Total_Energy         = ("Total energy",          "sum"),
    Total_CO2            = ("Total CO2",             "sum"),
    Total_Water          = ("Water intake",          "sum"),
    Avg_Energy_KPI       = ("Total energy - KPI",    "mean"),
    Avg_CO2_KPI          = ("Total CO2 - KPI",       "mean"),
    Avg_Water_KPI        = ("Water intake - KPI",    "mean"),
    Avg_Renewable_Share  = ("Renewable_Electricity_Share_%", "mean"),
    Avg_ISO_Cert         = ("ISO_Certification_%",   "mean"),
).reset_index()
sector_path = os.path.join(RAW_OUT, "ESG_SECTOR_AGGREGATED_2009_2023.csv")
sector.to_csv(sector_path, index=False)
print(f"    → SECTOR CSV:       {sector_path}")

# ── Step 5: Summary ───────────────────────────────────────────────────────────
print("\n[5/5] Done. Files in data_storage/raw/:")
for f in sorted(os.listdir(RAW_OUT)):
    size = os.path.getsize(os.path.join(RAW_OUT, f)) / 1024
    print(f"    {f:<55} {size:>7.1f} KB")

print("\n✅ Master database ready. Load it in Streamlit with:")
print('   df = pd.read_csv("data_storage/raw/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv")')