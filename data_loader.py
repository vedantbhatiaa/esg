"""
data_loader.py  —  TIP ESG Platform
======================================
Loads consolidated data from data_storage/master/.
Handles both wide format (master CSV) and long format (original consolidated Excel).
"""
from pathlib import Path
import pandas as pd

# ── Source file search order ───────────────────────────────────────────────────
_XLSX_CANDIDATES = [
    (Path("data_storage/master/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.xlsx"), "All_Companies"),
    (Path("data_storage/master/CONSOLIDATED_DUMMY_2009_2023.xlsx"),            "Raw Dummy data"),
    # Legacy paths kept for backward compatibility
    (Path("data_storage/raw/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.xlsx"),    "All_Companies"),
    (Path("data_storage/raw/CONSOLIDATED_DUMMY_2009_2023.xlsx"),               "Raw Dummy data"),
    (Path("data_storage/consolidated/CONSOLIDATED_DUMMY_2009_2023.xlsx"),      "Raw Dummy data"),
    (Path("CONSOLIDATED_DUMMY_2009_2023.xlsx"),                                "Raw Dummy data"),
]

_CSV_CANDIDATES = [
    Path("data_storage/master/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"),
    Path("data_storage/raw/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"),
    Path("data_storage/master/ESG_LONG_ALL_COMPANIES_2009_2023.csv"),
    Path("data_storage/consolidated/consolidated_benchmarking.csv"),
]

# ── Field mappings ─────────────────────────────────────────────────────────────
LABEL_TO_FIELD = {
    "Total no. of sites":                              "total_sites",
    "ISO 14001 sites":                                 "iso_sites",
    "Production":                                      "production",
    "Water intake":                                    "water_withdrawals",
    "Renewable Electricity Purchased":                 "renew_elec_purchased",
    "Non-Renewable Electricity Purchased":             "nonrenew_elec_purchased",
    "Self-generated AND consumed electricity on-site": "self_gen_elec",
    "Purchased Steam":                                 "purchased_steam",
    "Sold Electricity":                                "sold_electricity",
    "Sold Steam":                                      "sold_steam",
    "Natural Gas":                                     "nat_gas",
    "Coal":                                            "coal_sub",
    "Propane":                                         "propane",
    "Fuel Oil":                                        "fuel_oil_heavy_a",
    "Diesel":                                          "diesel",
    "Petrol":                                          "petrol",
    "Biomass":                                         "biomass",
    "Waste tires":                                     "waste_tires_mt",
    "LPG":                                             "lpg",
    "Other":                                           "other_fuels",
    "Total amount of waste":                           "waste_total",
    "Total amount of waste ":                          "waste_total",   # trailing-space variant (legacy)
    "Amount of waste sent to recovery":                "waste_recovery",
}

KPI_LABELS = {
    "Water intake - KPI":    "water_kpi",
    "Total energy - KPI":    "energy_kpi",
    "Total CO2 - KPI":       "co2_kpi",
    "% certified sites":     "iso_pct",
}

BENCH_LABELS = {
    "Total energy - KPI": "energy_kpi",
    "Total CO2 - KPI":    "co2_kpi",
    "Water intake - KPI": "water_kpi",
}

WIDE_COL_TO_FIELD = {
    "Total no. of sites":                              "total_sites",
    "ISO 14001 sites":                                 "iso_sites",
    "Production":                                      "production",
    "Water intake":                                    "water_withdrawals",
    "Renewable Electricity Purchased":                 "renew_elec_purchased",
    "Non-Renewable Electricity Purchased":             "nonrenew_elec_purchased",
    "Self-generated AND consumed electricity on-site": "self_gen_elec",
    "Purchased Steam":                                 "purchased_steam",
    "Sold Electricity":                                "sold_electricity",
    "Sold Steam":                                      "sold_steam",
    "Natural Gas":                                     "nat_gas",
    "Coal":                                            "coal_sub",
    "Propane":                                         "propane",
    "Fuel Oil":                                        "fuel_oil_heavy_a",
    "Diesel":                                          "diesel",
    "Petrol":                                          "petrol",
    "Biomass":                                         "biomass",
    "Waste tires":                                     "waste_tires_mt",
    "LPG":                                             "lpg",
    "Other":                                           "other_fuels",
    "Total amount of waste":                           "waste_total",
    "Amount of waste sent to recovery":                "waste_recovery",
    # ── Master wide CSV column names (used after build_esg_master.py) ─────────
    "Total Waste":                                     "waste_total",
    "Waste Recovered":                                 "waste_recovery",
}

TEMPLATE_INPUT_FIELDS = {
    "total_sites","iso_sites","production","water_withdrawals",
    "renew_elec_purchased","nonrenew_elec_purchased","self_gen_elec",
    "purchased_steam","sold_electricity","sold_steam",
    "nat_gas","coal_sub","propane","fuel_oil_heavy_a",
    "diesel","petrol","biomass","waste_tires_mt","lpg","other_fuels",
    "co2_scope2_steam","waste_total","waste_recovery",
}


def _is_wide_format(df: pd.DataFrame) -> bool:
    return "Company" in df.columns and "Year" in df.columns and "Row_Label" not in df.columns


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleaning: parse Year, coerce numerics."""
    df.columns = [str(c).strip() for c in df.columns]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"])
    if _is_wide_format(df):
        for col in df.columns:
            if col not in ("Company", "Year"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df["Data"] = pd.to_numeric(df["Data"], errors="coerce")
        df = df.dropna(subset=["Row_Label"])
        df["Row_Label"] = df["Row_Label"].astype(str).str.strip()
    return df


def load_consolidated() -> pd.DataFrame:
    """
    Load source data. Returns a cleaned DataFrame (wide or long).

    Load order — CSV is checked FIRST because the app save function updates
    the master CSV in real time. If CSV is loaded first, every Streamlit rerun
    automatically picks up the latest saved data (including updates to any year).
    XLSX is only used as a fallback when no CSV is available.

    Priority:
      1. data_storage/master/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv  ← app saves here
      2. data_storage/raw/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv
      3. XLSX files (stale after app saves — fallback only)
    """
    # ── 1. CSV first (always up-to-date after app saves) ─────────────────────
    for path in _CSV_CANDIDATES:
        if path.exists():
            try:
                df = _clean_df(pd.read_csv(path))
                if not df.empty:
                    print(f"[data_loader] Loaded CSV: {path.name} — {len(df)} rows")
                    return df
            except Exception as e:
                print(f"[data_loader] CSV error ({path.name}): {e}")

    # ── 2. XLSX fallback (only when no CSV exists, e.g. fresh install) ────────
    for path, sheet in _XLSX_CANDIDATES:
        if path.exists():
            try:
                df = _clean_df(pd.read_excel(path, sheet_name=sheet, header=0))
                if not df.empty:
                    print(f"[data_loader] Loaded XLSX fallback: {path.name} — {len(df)} rows")
                    return df
            except Exception as e:
                print(f"[data_loader] XLSX error ({path.name}): {e}")

    print("[data_loader] WARNING: No data file found. Run python build_esg_master.py first.")
    return pd.DataFrame()


def get_companies(df: pd.DataFrame) -> list:
    return sorted(df["Company"].dropna().unique().tolist()) if not df.empty else []


def get_years(df: pd.DataFrame, company: str = None) -> list:
    if df.empty: return []
    sub = df[df["Company"] == company] if company else df
    return sorted(sub["Year"].dropna().unique().astype(int).tolist())


def get_company_hist(df: pd.DataFrame, company: str) -> dict:
    """Returns {field: {year: value}} for all INPUT fields. Keys are always valid TemplateInputs field names."""
    if df.empty: return {}
    comp_df = df[df["Company"] == company]

    if _is_wide_format(df):
        result = {}
        for col in comp_df.columns:
            if col in ("Company","Year"): continue
            field = WIDE_COL_TO_FIELD.get(col, col)
            if field not in TEMPLATE_INPUT_FIELDS: continue
            rows = comp_df[["Year", col]].dropna(subset=[col])
            if not rows.empty:
                result[field] = {int(yr): float(val) for yr, val in zip(rows["Year"], rows[col]) if pd.notna(val)}
        return result

    result = {}
    for label, field in LABEL_TO_FIELD.items():
        if field not in TEMPLATE_INPUT_FIELDS: continue
        rows = comp_df[comp_df["Row_Label"] == label.strip()][["Year","Data"]].dropna()
        if not rows.empty:
            result[field] = {int(yr): float(val) for yr, val in zip(rows["Year"], rows["Data"]) if pd.notna(val)}
    return result


def get_step_data(company_hist: dict, year: int) -> dict:
    return {f: float(ym[year]) for f, ym in company_hist.items() if year in ym and pd.notna(ym[year])}


def get_hist_raw(company_hist: dict, years: list) -> dict:
    hist = {}
    for field, ym in company_hist.items():
        vals = [float(ym.get(yr, 0.0)) for yr in years]
        if any(v != 0.0 for v in vals):
            hist[field] = vals
    return hist


def get_kpi_hints(df: pd.DataFrame, company: str, year: int) -> dict:
    if df.empty: return {}
    if _is_wide_format(df):
        row = df[(df["Company"] == company) & (df["Year"] == year)]
        if row.empty: return {}
        hints = {}
        for col, kpi in {"Total energy - KPI":"energy_kpi","Total CO2 - KPI":"co2_kpi",
                          "Water intake - KPI":"water_kpi","% certified sites":"iso_pct"}.items():
            if col in row.columns and pd.notna(row[col].values[0]):
                hints[kpi] = float(row[col].values[0])
        return hints
    comp_yr = df[(df["Company"] == company) & (df["Year"] == year)]
    hints = {}
    for label, kpi in KPI_LABELS.items():
        rows = comp_yr[comp_yr["Row_Label"] == label.strip()]["Data"].dropna()
        if not rows.empty:
            hints[kpi] = float(rows.iloc[0])
    return hints


def get_benchmark_kpis(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    year_df = df[df["Year"] == year]
    if _is_wide_format(df):
        kpi_map = {"Total energy - KPI":"energy_kpi","Total CO2 - KPI":"co2_kpi","Water intake - KPI":"water_kpi"}
        available = {k: v for k, v in kpi_map.items() if k in year_df.columns}
        if not available: return pd.DataFrame()
        result = year_df[["Company"] + list(available.keys())].copy()
        return result.rename(columns=available).reset_index(drop=True)
    frames = []
    for label, col in BENCH_LABELS.items():
        rows = year_df[year_df["Row_Label"] == label.strip()][["Company","Data"]].copy()
        if not rows.empty:
            frames.append(rows.rename(columns={"Data":col}).set_index("Company")[col])
    return pd.concat(frames, axis=1).reset_index() if frames else pd.DataFrame()


def improvement_since(company_hist: dict, field: str, base_year: int, end_year: int):
    ym   = company_hist.get(field, {})
    base = ym.get(base_year)
    end  = ym.get(end_year)
    if base and end and base != 0:
        return (end - base) / abs(base) * 100
    return None


def company_trend(df: pd.DataFrame, company: str, row_label: str) -> "pd.Series":
    if df.empty: return pd.Series(dtype=float)
    if _is_wide_format(df):
        col = WIDE_COL_TO_FIELD.get(row_label, row_label)
        if col not in df.columns: return pd.Series(dtype=float)
        return df[df["Company"] == company].set_index("Year")[col].sort_index()
    mask = (df["Company"] == company) & (df["Row_Label"] == row_label.strip())
    return df[mask].set_index("Year")["Data"].sort_index()


def load_sector_aggregated() -> "pd.DataFrame":
    """Load pre-aggregated sector CSV from master/."""
    candidates = [
        Path("data_storage/master/ESG_SECTOR_AGGREGATED_2009_2023.csv"),
        Path("data_storage/raw/ESG_SECTOR_AGGREGATED_2009_2023.csv"),
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["Year"] = df["Year"].astype(int)
                print(f"[data_loader] Loaded sector aggregation: {path}")
                return df
            except Exception as e:
                print(f"[data_loader] Sector CSV error: {e}")
    return pd.DataFrame()