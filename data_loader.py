"""
data_loader.py — TIP ESG Platform
Loads 2024_Consolidation.xlsx (Raw data sheet) and maps every field
to TemplateInputs field names. Falls back to CSV if XLSX not found.
"""
from pathlib import Path
import pandas as pd

_XLSX_CANDIDATES = [
    # Dummy dataset with fictional company names (primary)
    (Path("data_storage/consolidated/CONSOLIDATED_DUMMY_2009_2023.xlsx"), "Raw Dummy data"),
    # Real consolidation fallback
    (Path("data_storage/consolidated/2024_Consolidation.xlsx"), "Raw data"),
    (Path("2024_Consolidation.xlsx"), "Raw data"),
]
_CSV_CANDIDATES = [
    Path("data_storage/consolidated/consolidated_benchmarking.csv"),
    Path("consolidated_benchmarking.csv"),
]

# ── Input field mapping: Row_Label → TemplateInputs field ────────────────────
LABEL_TO_FIELD: dict = {
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
    "Total amount of waste ":                          "waste_total",
    "Amount of waste sent to recovery":                "waste_recovery",
}

KPI_LABELS: dict = {
    "Water intake - KPI":    "water_kpi",
    "Total energy - KPI":    "energy_kpi",
    "Total CO2 - KPI":       "co2_kpi",
    "% certified sites":     "iso_pct",
    "Waste intensity - KPI ": "waste_kpi",
}

BENCH_LABELS: dict = {
    "Total energy - KPI": "energy_kpi",
    "Total CO2 - KPI":    "co2_kpi",
    "Water intake - KPI": "water_kpi",
}


def load_consolidated() -> pd.DataFrame:
    """Load XLSX first, CSV as fallback. Returns cleaned DataFrame."""
    for path, sheet in _XLSX_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=0)
                df.columns = [str(c).strip() for c in df.columns]
                df["Data"] = pd.to_numeric(df["Data"], errors="coerce")
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["Row_Label", "Year"])
                df["Row_Label"] = df["Row_Label"].astype(str).str.strip()
                print(f"[data_loader] Loaded {path.name} (sheet: {sheet}) — {len(df)} rows")
                return df
            except Exception as e:
                print(f"[data_loader] XLSX error ({path.name}): {e}")
    for path in _CSV_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                df["Data"] = pd.to_numeric(df["Data"], errors="coerce")
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["Row_Label", "Year"])
                df["Row_Label"] = df["Row_Label"].astype(str).str.strip()
                print(f"[data_loader] Loaded CSV: {path} — {len(df)} rows")
                return df
            except Exception as e:
                print(f"[data_loader] CSV error: {e}")
    print("[data_loader] WARNING: No data file found.")
    return pd.DataFrame()


def get_companies(df: pd.DataFrame) -> list:
    if df.empty: return []
    return sorted(df["Company"].dropna().unique().tolist())


def get_years(df: pd.DataFrame, company: str = None) -> list:
    if df.empty: return []
    sub = df[df["Company"] == company] if company else df
    return sorted(sub["Year"].dropna().unique().astype(int).tolist())


def get_company_hist(df: pd.DataFrame, company: str) -> dict:
    """Returns {field: {year: value}} for all mapped input fields."""
    if df.empty: return {}
    comp_df = df[df["Company"] == company]
    result = {}
    for label, field in LABEL_TO_FIELD.items():
        rows = comp_df[comp_df["Row_Label"] == label.strip()][["Year", "Data"]].dropna()
        if not rows.empty:
            result[field] = {int(yr): float(val) for yr, val in zip(rows["Year"], rows["Data"]) if not pd.isna(val)}
    return result


def get_step_data(company_hist: dict, year: int) -> dict:
    """Returns {field: value} for a given year — pre-fills stepper inputs."""
    return {f: float(ym[year]) for f, ym in company_hist.items() if year in ym and not pd.isna(ym[year])}


def get_hist_raw(company_hist: dict, years: list) -> dict:
    """Returns {field: [value_per_year]} — populates historical columns."""
    hist = {}
    for field, ym in company_hist.items():
        vals = [float(ym.get(yr, 0.0)) for yr in years]
        if any(v != 0.0 for v in vals):
            hist[field] = vals
    return hist


def get_kpi_hints(df: pd.DataFrame, company: str, year: int) -> dict:
    """Returns prior-year KPIs as {kpi_name: value} for reference hints."""
    if df.empty: return {}
    comp_yr = df[(df["Company"] == company) & (df["Year"] == year)]
    hints = {}
    for label, kpi in KPI_LABELS.items():
        rows = comp_yr[comp_yr["Row_Label"] == label.strip()]["Data"].dropna()
        if not rows.empty:
            hints[kpi] = float(rows.iloc[0])
    return hints


def get_benchmark_kpis(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Returns [Company, energy_kpi, co2_kpi, water_kpi] for quartile calcs."""
    if df.empty: return pd.DataFrame()
    year_df = df[df["Year"] == year]
    frames = []
    for label, col in BENCH_LABELS.items():
        rows = year_df[year_df["Row_Label"] == label.strip()][["Company", "Data"]].copy()
        rows = rows.rename(columns={"Data": col})
        if not rows.empty:
            frames.append(rows.set_index("Company")[col])
    return pd.concat(frames, axis=1).reset_index() if frames else pd.DataFrame()


def improvement_since(company_hist: dict, field: str, base_year: int, end_year: int):
    """% change from base_year to end_year."""
    ym = company_hist.get(field, {})
    base, end = ym.get(base_year), ym.get(end_year)
    if base and end and base != 0:
        return (end - base) / abs(base) * 100
    return None


def company_trend(df: pd.DataFrame, company: str, row_label: str) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    mask = (df["Company"] == company) & (df["Row_Label"] == row_label.strip())
    return df[mask].set_index("Year")["Data"].sort_index()