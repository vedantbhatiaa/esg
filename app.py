"""
TIP ESG Platform -- Streamlit Frontend
========================================
Run: streamlit run app.py

Changes from original:
  - formula_engine now imported from formula_engine.py (was .ipynb -- crashed on startup)
  - Removed dead load_consolidated_data() function (used local_storage but was never called)
  - analysis page now reads LONG_DATA from real sector CSV when available
  - data_loader now auto-finds master CSV in data_storage/raw/ (no manual path fix needed)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import html as _html          # H3: HTML-escape user-derived values before unsafe_allow_html
from filelock import FileLock  # H1: advisory lock for concurrent CSV writes

from formula_engine import (
    TemplateInputs, calculate, validate_submission,
    get_benchmarks, build_template_dataframe, fmt_num,
    yoy_change, ValidationFlag, BenchmarkResult
)

import data_loader as dl

# Chatbot is embedded in page_readiness only — no global import needed

# Load fresh from disk on every Streamlit rerun (Streamlit reruns the full
# script on every user interaction, so this is always up-to-date after a save).
# data_loader checks data_storage/master/ first, then falls back to raw/ etc.
_CONSOLIDATED_DF = dl.load_consolidated()
_COMPANIES       = dl.get_companies(_CONSOLIDATED_DF)
_SECTOR_DF       = dl.load_sector_aggregated()

# ─────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TIP ESG Platform · dss+",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* -- Sidebar -- */
[data-testid="stSidebar"] { background: #0A2240 !important; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong
{ color: #ffffff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stSidebarNav"] { display: none; }

/* -- Sidebar nav buttons -- */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: rgba(255,255,255,0.75) !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 9px 12px !important;
    width: 100% !important;
    transition: background .15s, color .15s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important; outline: none !important;
    border: none !important; color: #ffffff !important;
}

/* -- Main buttons -- */
.stButton > button {
    border-radius: 7px; font-weight: 500; font-size: 13px;
    border: 1.5px solid #D1D5DB; transition: all .15s;
}
.stButton > button:hover { border-color: #6B7280; }

/* -- Form inputs -- */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    border-radius: 7px; border: 1.5px solid #D1D5DB; font-size: 14px !important;
}

/* -- KPI cards -- */
.kpi-card {
    background: #fff; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 16px 18px; text-align: left;
}
.kpi-card .label { font-size: 11px; color: #6B7280;
    text-transform: uppercase; letter-spacing: .5px; font-weight: 500; }
.kpi-card .value { font-size: 26px; font-weight: 700; color: #111827; margin: 5px 0 2px; }
.kpi-card .unit  { font-size: 12px; color: #9CA3AF; }
.kpi-card .delta { font-size: 12px; font-weight: 600; margin-top: 4px; }
.delta-pos { color: #059669; }
.delta-neg { color: #DC2626; }

/* -- Stepper -- */
.step-bar { display:flex; align-items:center; gap:0;
    background:#fff; border:1px solid #E5E7EB; border-radius:10px;
    padding:16px 20px; margin-bottom:20px; }
.step-item { display:flex; align-items:center; flex:1; min-width:0; }
.step-circle { width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:12px; font-weight:700; flex-shrink:0; }
.sc-done   { background:#00916E; color:#fff; }
.sc-active { background:#1D4ED8; color:#fff; }
.sc-todo   { background:#F3F4F6; color:#9CA3AF; border:2px solid #E5E7EB; }
.step-label { font-size:11.5px; font-weight:500; margin-left:7px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sl-done   { color:#00916E; }
.sl-active { color:#1D4ED8; }
.sl-todo   { color:#9CA3AF; }
.step-line { flex:1; height:2px; background:#E5E7EB; margin:0 6px; min-width:8px; }
.sl-done-line { background:#00916E; }

/* -- Table legend -- */
.tbl-legend { display:flex; gap:14px; padding:10px 16px;
    background:#F9FAFB; border-top:1px solid #E5E7EB;
    border-radius:0 0 8px 8px; flex-wrap:wrap; }
.tl { display:flex; align-items:center; gap:5px; font-size:11px; color:#6B7280; }
.tl-sw { width:14px; height:14px; border-radius:3px;
    border:1px solid #D1D5DB; display:inline-block; }

/* -- Band chart -- */
.band-container { margin: 6px 0 12px; }
.band-row-wrap { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.band-lbl { font-size:12px; font-weight:500; color:#374151; width:170px; flex-shrink:0; }
.band-track { flex:1; height:18px; border-radius:4px; background:#F3F4F6; position:relative; }
.band-seg { position:absolute; top:0; height:100%; }
.band-pin { position:absolute; width:4px; height:28px;
    background:#0A2240; border-radius:2px; top:-5px; transform:translateX(-50%); }
.band-pin-val { position:absolute; font-size:10px; font-weight:700;
    color:#0A2240; top:-18px; transform:translateX(-50%);
    white-space:nowrap; background:#fff; padding:0 2px; }
.band-chip { font-size:11px; font-weight:600; padding:3px 9px;
    border-radius:10px; flex-shrink:0; }
.chip-top  { background:#D1FAE5; color:#065F46; }
.chip-mid  { background:#FEF3C7; color:#92400E; }
.chip-bot  { background:#FEE2E2; color:#991B1B; }

/* -- Flag cards -- */
.flag-card { display:flex; align-items:flex-start; gap:10px;
    padding:12px 14px; border-radius:8px; border:1px solid; margin-bottom:10px; }
.fc-warn  { background:#FFF7ED; border-color:#FCD34D; }
.fc-error { background:#FEF2F2; border-color:#FECACA; }
.fc-ok    { background:#ECFDF5; border-color:#6EE7B7; }
.fc-icon  { width:20px; height:20px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:10px; font-weight:700; flex-shrink:0; }
.fi-warn  { background:#D97706; color:#fff; }
.fi-error { background:#DC2626; color:#fff; }
.fi-ok    { background:#00916E; color:#fff; }
.fc-title { font-size:13px; font-weight:600; color:#111827; }
.fc-detail{ font-size:12px; color:#6B7280; margin-top:3px; line-height:1.6; }

/* -- AI card -- */
.ai-card { background:#fff; border:1px solid #E5E7EB;
    border-radius:10px; overflow:hidden; margin-bottom:12px; }
.ai-head  { display:flex; align-items:center; gap:8px;
    padding:10px 14px; background:#F9FAFB; border-bottom:1px solid #E5E7EB; }
.ai-pulse { width:8px; height:8px; border-radius:50%;
    background:#00916E; flex-shrink:0; }
.ai-title { font-size:13px; font-weight:600; color:#111827; }
.ai-badge { margin-left:auto; background:#E6F5F1; color:#007A5C;
    font-size:11px; padding:2px 9px; border-radius:10px;
    border:1px solid #6EE7B7; font-weight:500; }
.ai-body  { padding:12px 14px; font-size:13px; color:#374151; line-height:1.8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
HIST_YEARS = list(range(2009, 2023))
CURR_YEAR  = 2023
LONG_YEARS = list(range(2009, 2024))

# ── All 31 electricity-by-country names (matches UI editor row order) ─────────
ELEC_ALL_COUNTRIES = [
    "Canada", "Chile", "Mexico", "United States",
    "Australia", "Japan", "Korea", "New Zealand",
    "Austria", "Belgium", "Czech Republic", "Denmark", "Finland", "France",
    "Germany", "Hungary", "Iceland", "Ireland", "Italy", "Luxembourg",
    "Netherlands", "Norway", "Poland", "Portugal", "Spain", "Sweden",
    "Switzerland", "Turkey", "United Kingdom",
    "China", "India",
]

def _elec_col(country: str) -> str:
    """Canonical master CSV column name for a country's electricity (GJ)."""
    return "Elec_" + country.replace(" ", "_") + "_GJ"

# Dict: country name → master CSV column name
ELEC_COUNTRY_COLS = {c: _elec_col(c) for c in ELEC_ALL_COUNTRIES}

COMPANIES = _COMPANIES if _COMPANIES else [
    "VerdaTyres Corp", "AlphaTread Ltd", "BetaRubber Inc", "GammaTire SA",
    "DeltaGrip GmbH", "EpsilonWheel Co", "ZetaTrac LLC", "EtaRoad AG",
    "ThetaDrive NV", "IotaTire PLC",
]

# -- Static fallback HIST_RAW (used only if no company selected yet) -----------
HIST_RAW: dict[str, list] = {
    "total_sites":   [38,39,40,40,41,42,43,44,46,48,51,51,52,52],
    "iso_sites":     [36,38,39,39,40,41,42,43,45,47,51,51,52,52],
    "production":    [2_840_000,3_510_000,3_770_000,3_520_000,3_640_000,3_620_000,
                      3_540_000,3_630_000,3_700_000,3_910_000,3_860_000,3_050_000,3_320_000,3_580_000],
    "water_withdrawals": [18_000_000,19_200_000,20_100_000,19_700_000,20_500_000,21_000_000,
                          21_500_000,22_000_000,22_800_000,23_100_000,23_100_000,20_300_000,21_100_000,20_900_000],
    "renew_elec_purchased": [0,0,0,0,0,0,0,300_000,250_000,1_100_000,706_562,1_528_836,2_557_561,4_082_923],
    "nonrenew_elec_purchased": [9_500_000,9_400_000,9_600_000,9_300_000,9_400_000,9_500_000,
                                9_600_000,9_400_000,9_300_000,9_100_000,12_271_131,9_667_437,10_297_758,9_037_549],
    "nat_gas": [13_000_000,14_000_000,15_000_000,14_500_000,15_000_000,15_200_000,
                15_500_000,15_700_000,15_900_000,16_100_000,16_210_969,14_040_397,15_939_109,15_927_554],
    "coal_sub": [500_000,490_000,480_000,470_000,460_000,450_000,
                 440_000,430_000,420_000,410_000,456_997,337_992,360_848,395_006],
    "lpg": [1_000_000,1_050_000,1_100_000,1_100_000,1_100_000,1_150_000,
            1_150_000,1_180_000,1_200_000,1_220_000,1_237_839,1_124_479,1_271_422,1_329_571],
    "waste_total":    [330_000,330_000,335_000,335_000,340_000,342_000,
                       344_000,346_000,348_000,350_000,352_000,295_000,320_000,335_000],
    "waste_recovery": [280_000,281_000,283_000,284_000,286_000,287_000,
                       289_000,292_000,295_000,298_000,299_200,253_700,275_200,284_750],
}

# -- LONG_DATA: built from real sector CSV if available, else static fallback --
def _build_long_data() -> tuple[dict, dict]:
    """
    Build LONG_DATA and FUEL_MIX from the real consolidated wide DataFrame.
    Falls back to static values only for missing years/fields.
    """
    static_long = {
        "energy":     [28.1,32.3,33.6,32.4,33.0,32.4,31.8,32.1,33.0,34.2,33.2,28.5,32.3,32.5,32.4],
        "co2":        [2.41,2.69,2.88,2.80,2.87,2.86,2.73,2.72,2.80,2.85,2.76,2.22,2.27,2.06,2.05],
        "water":      [22.4,23.8,24.9,24.4,23.9,22.9,22.9,23.5,23.5,23.2,23.1,20.3,21.1,20.9,21.5],
        "scope1":     [1.08,1.19,1.21,1.17,1.20,1.15,1.09,1.08,1.12,1.15,1.11,0.94,1.06,1.05,1.03],
        "scope2":     [1.33,1.50,1.67,1.63,1.67,1.71,1.64,1.64,1.68,1.70,1.65,1.27,1.21,1.01,1.02],
        "energy_kpi": [9.9,9.2,8.9,9.2,9.1,8.9,8.9,8.8,8.9,8.8,8.6,9.3,9.7,9.1,8.7],
        "co2_kpi":    [0.850,0.765,0.764,0.795,0.789,0.791,0.771,0.748,0.758,0.729,0.715,0.729,0.684,0.576,0.551],
        "renew_pct":  [0,0,0,0,0,0,0,2.3,2.2,9.7,10.6,21.8,31.4,40.6,48.3],
        "waste_recov":[83,83,84,84,84,84,84,85,85,85,85,86,86,85,86],
        "prod":       [2.84,3.51,3.77,3.52,3.64,3.62,3.54,3.63,3.70,3.91,3.86,3.05,3.32,3.58,3.72],
    }
    static_fuel = {
        "Natural Gas": [46,46,47,47,47,46,47,47,48,49,49,49,49,49,50],
        "Electricity": [34.7,34.2,34.9,34.6,35.3,36.7,37.1,37.9,38.2,38.8,39.1,39.3,39.8,40.4,40.7],
        "Fuel Oil":    [8.5,6.7,6,5.8,5.1,4.8,3.7,3.2,3,2.6,2.4,1.8,1.4,0.5,0.5],
        "LPG":         [2.4,2.4,2.3,3.6,3.5,3.5,3.5,3.6,3.5,3.6,3.7,3.9,3.9,4.1,4.2],
        "Coal":        [3.2,3.1,2.8,2.8,3.7,3.6,2.3,2,2.1,1.6,1.4,1.2,1.2,1.2,1.2],
        "Other":       [5.2,7.6,7,5.4,5.4,5.4,6.4,5.3,4.7,4.4,4,4.8,4.3,4.8,3.4],
    }

    def _safe_list(series, fallback):
        result = []
        for i, v in enumerate(series):
            try:
                f = float(v)
                result.append(f if not np.isnan(f) else fallback[i])
            except Exception:
                result.append(fallback[i])
        return result

    df = _CONSOLIDATED_DF
    if df.empty or "Row_Label" in df.columns:
        # Long format or no data -- use static
        if not _SECTOR_DF.empty:
            try:
                s = _SECTOR_DF.set_index("Year").reindex(LONG_YEARS)
                live = {
                    "energy":     _safe_list((s["Total_Energy"]/1e6), static_long["energy"]),
                    "co2":        _safe_list((s["Total_CO2"]/1e6),    static_long["co2"]),
                    "water":      _safe_list((s["Total_Water"]/1e6),  static_long["water"]),
                    "energy_kpi": _safe_list(s["Avg_Energy_KPI"],     static_long["energy_kpi"]),
                    "co2_kpi":    _safe_list(s["Avg_CO2_KPI"],        static_long["co2_kpi"]),
                    "renew_pct":  _safe_list(s["Avg_Renewable_Share"],static_long["renew_pct"]),
                    "prod":       _safe_list((s["Total_Production"]/1e6), static_long["prod"]),
                    "scope1":     static_long["scope1"],
                    "scope2":     static_long["scope2"],
                    "waste_recov":static_long["waste_recov"],
                }
                return live, static_fuel
            except Exception as e:
                print(f"[app] Sector DF error: {e}")
        return static_long, static_fuel

    # Wide format -- compute directly from master DataFrame
    try:
        grp = df.groupby("Year")

        def _col_sum(col, divisor=1):
            if col in df.columns:
                return grp[col].sum() / divisor
            return None

        def _col_mean(col):
            if col in df.columns:
                return grp[col].mean()
            return None

        energy_s  = _col_sum("Total energy", 1e6)
        co2_s     = _col_sum("Total CO2", 1e6)
        scope1_s  = _col_sum("Total CO2 - Scope 1", 1e6)
        scope2_s  = _col_sum("Total CO2 - Scope 2", 1e6)
        water_s   = _col_sum("Water intake", 1e6)
        prod_s    = _col_sum("Production", 1e6)
        ekpi_m    = _col_mean("Total energy - KPI")
        co2kpi_m  = _col_mean("Total CO2 - KPI")
        renew_m   = _col_mean("Renewable_Electricity_Share_%")

        def _to_list(series, fallback):
            if series is None:
                return fallback
            s = series.reindex(LONG_YEARS)
            return _safe_list(s.values, fallback)

        live = {
            "energy":     _to_list(energy_s,  static_long["energy"]),
            "co2":        _to_list(co2_s,     static_long["co2"]),
            "scope1":     _to_list(scope1_s,  static_long["scope1"]),
            "scope2":     _to_list(scope2_s,  static_long["scope2"]),
            "water":      _to_list(water_s,   static_long["water"]),
            "prod":       _to_list(prod_s,    static_long["prod"]),
            "energy_kpi": _to_list(ekpi_m,    static_long["energy_kpi"]),
            "co2_kpi":    _to_list(co2kpi_m,  static_long["co2_kpi"]),
            "renew_pct":  _to_list(renew_m,   static_long["renew_pct"]),
            "waste_recov":static_long["waste_recov"],
        }

        # Fuel mix as % of total energy per year
        fuel_cols = {
            "Natural Gas": "Natural Gas",
            "Electricity": "Total Electricity",
            "Fuel Oil":    "Fuel Oil",
            "LPG":         "LPG",
            "Coal":        "Coal",
            "Other":       "Other",
        }
        total_e_by_yr = grp["Total energy"].sum() if "Total energy" in df.columns else None
        live_fuel = {}
        for label, col in fuel_cols.items():
            if col in df.columns and total_e_by_yr is not None:
                fuel_sum = grp[col].sum().reindex(LONG_YEARS)
                total_e  = total_e_by_yr.reindex(LONG_YEARS)
                pct = (fuel_sum / total_e.replace(0, np.nan) * 100).fillna(0)
                live_fuel[label] = _safe_list(pct.values, static_fuel.get(label, [0]*15))
            else:
                live_fuel[label] = static_fuel.get(label, [0]*15)

        return live, live_fuel if any(sum(v) > 0 for v in live_fuel.values()) else static_fuel

    except Exception as e:
        print(f"[app] Wide DF live computation error: {e}")
        return static_long, static_fuel


LONG_DATA, FUEL_MIX = _build_long_data()

# L2 FIX: track whether analysis charts are showing real or fallback data.
# Surfaced as a warning banner in page_analysis() so analysts never mistake
# synthetic demo numbers for real client submissions.
_USING_FALLBACK_DATA = _CONSOLIDATED_DF.empty

CLIENTS = {
    "verdatyres@tip-reporting.com":   "VerdaTyres Corp",
    "alphatread@tip-reporting.com":   "AlphaTread Ltd",
    "betarubber@tip-reporting.com":   "BetaRubber Inc",
    "gammatire@tip-reporting.com":    "GammaTire SA",
    "deltagrip@tip-reporting.com":    "DeltaGrip GmbH",
    "epsilonwheel@tip-reporting.com": "EpsilonWheel Co",
    "zetatrac@tip-reporting.com":     "ZetaTrac LLC",
    "etaroad@tip-reporting.com":      "EtaRoad AG",
    "thetadrive@tip-reporting.com":   "ThetaDrive NV",
    "iotatire@tip-reporting.com":     "IotaTire PLC",
}

STEP_META = [
    ("ISO 14001",  "Certified sites and facility coverage"),
    ("Production", "Annual production volume"),
    ("Water",      "Water withdrawals by source"),
    ("Energy",     "Electricity and fuel consumption"),
    ("CO2",        "Emission inputs and auto-calculated totals"),
    ("Waste",      "Waste generated, recovered and eliminated"),
]

# ─────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "authenticated":      False,
        "user_name":          "",
        "user_company":       "",
        "user_email":         "",
        "is_dss":             False,
        "page":               "login",
        "step":               0,
        "template_done":      False,
        "company_setup_done": False,
        "reporting_company":  "",
        "reporting_year":     2023,
        "employee_name":      "",
        "company_hist":       {},
        "live_hist_raw":      {},
        "kpi_hints":          {},
        "step_data": {
            "total_sites": 54, "iso_sites": 54,
            "production": 3_720_000,
            "water_withdrawals": 21_500_000,
            "renew_elec_purchased": 5_200_000,
            "nonrenew_elec_purchased": 8_500_000,
            "self_gen_elec": 45_000,
            "purchased_steam": 1_050_000,
            "sold_electricity": 8_000,
            "sold_steam": 0,
            "nat_gas": 16_100_000, "coal_sub": 380_000,
            "propane": 340_000, "fuel_oil_heavy_a": 150_000,
            "diesel": 190_000, "petrol": 0, "biomass": 0,
            "waste_tires_mt": 0, "lpg": 1_350_000, "other_fuels": 0,
            "co2_scope2_steam": 60_000,
            "waste_total": 338_000, "waste_recovery": 290_000,
        },
        "flags_resolved": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def get_current_outputs():
    sd = st.session_state.step_data
    inp = TemplateInputs(
        company=st.session_state.get("reporting_company") or st.session_state.user_company,
        year=st.session_state.get("reporting_year", CURR_YEAR),
        **{k: float(sd.get(k, 0)) for k in [
            "total_sites", "iso_sites", "production", "water_withdrawals",
            "renew_elec_purchased", "nonrenew_elec_purchased", "self_gen_elec",
            "purchased_steam", "sold_electricity", "sold_steam",
            "nat_gas", "coal_sub", "propane", "fuel_oil_heavy_a",
            "diesel", "petrol", "biomass", "waste_tires_mt", "lpg", "other_fuels",
            "co2_scope2_steam", "waste_total", "waste_recovery",
        ]}
    )
    return inp, calculate(inp)


# Valid TemplateInputs fields -- used to guard against unexpected keys from consolidated data
_VALID_TEMPLATE_FIELDS = {
    "total_sites", "iso_sites", "production", "water_withdrawals",
    "renew_elec_purchased", "nonrenew_elec_purchased", "self_gen_elec",
    "purchased_steam", "sold_electricity", "sold_steam",
    "nat_gas", "coal_sub", "propane", "fuel_oil_heavy_a",
    "diesel", "petrol", "biomass", "waste_tires_mt", "lpg", "other_fuels",
    "co2_scope2_steam", "waste_total", "waste_recovery",
}

def _get_fresh_hist(company: str = None) -> dict:
    """
    Always load company historical data fresh from _CONSOLIDATED_DF.
    Because _CONSOLIDATED_DF is reloaded from disk on every Streamlit rerun,
    this automatically reflects any saves (including updates to previous years)
    without requiring a manual reload or re-running the stepper.
    Falls back to session state cache, then HIST_RAW if company is unknown.
    """
    co = company or st.session_state.get("reporting_company") or st.session_state.get("user_company") or ""
    if co and not _CONSOLIDATED_DF.empty:
        hist = dl.get_company_hist(_CONSOLIDATED_DF, co)
        if hist:
            return dl.get_hist_raw(hist, HIST_YEARS)
    # Fallback: session state (set during company setup), then static defaults
    return st.session_state.get("live_hist_raw") or HIST_RAW


def get_hist_outputs():
    # Always use fresh data from _CONSOLIDATED_DF so any saved updates
    # (including changes to previous years) are immediately visible.
    company = (st.session_state.get("reporting_company") or
               st.session_state.get("user_company") or "")
    _hist = _get_fresh_hist(company)
    outs = []
    for i, yr in enumerate(HIST_YEARS):
        # Only pass keys that are valid TemplateInputs fields
        fields = {
            k: _hist[k][i]
            for k in _hist
            if i < len(_hist[k]) and k in _VALID_TEMPLATE_FIELDS
        }
        inp = TemplateInputs(company=company, year=yr, **fields)
        outs.append((yr, inp, calculate(inp)))
    return outs


def kpi_card_html(label, value, unit, delta, delta_positive=True):
    # H3 FIX: escape every caller-supplied string before injecting into HTML
    e = _html.escape
    delta_cls = "delta-pos" if delta_positive else "delta-neg"
    arrow = "▼" if delta_positive else "▲"
    return f"""<div class="kpi-card">
        <div class="label">{e(str(label))}</div>
        <div class="value">{e(str(value))}</div>
        <div class="unit">{e(str(unit))}</div>
        <div class="delta {delta_cls}">{arrow} {e(str(delta))}</div>
    </div>"""


def band_html(label, val, q25, median, q75, unit, lower_better=True):
    span    = q75 - q25
    bmin    = q25 - span * 0.4
    bmax    = q75 + span * 0.4
    rng     = bmax - bmin if (bmax - bmin) != 0 else 1
    pos_pct = max(2, min(98, (val - bmin) / rng * 100))
    if lower_better:
        top_cls = "chip-top" if val <= q25 else "chip-mid" if val <= median else "chip-bot"
        top_lbl = "Top 25%"  if val <= q25 else "Average"  if val <= median else "Below avg"
    else:
        top_cls = "chip-top" if val >= q75 else "chip-mid" if val >= median else "chip-bot"
        top_lbl = "Top 25%"  if val >= q75 else "Average"  if val >= median else "Below avg"
    val_str = f"{val:.3f}" if isinstance(val, float) and val < 10 else fmt_num(val)
    # H3 FIX: escape label before injecting into HTML
    e = _html.escape
    return f"""
    <div class="band-row-wrap">
      <div class="band-lbl">{e(str(label))}</div>
      <div class="band-track">
        <div class="band-seg" style="left:0%;width:25%;background:#D1FAE5;border-radius:4px 0 0 4px"></div>
        <div class="band-seg" style="left:25%;width:50%;background:#FEF3C7"></div>
        <div class="band-seg" style="left:75%;width:25%;background:#FEE2E2;border-radius:0 4px 4px 0"></div>
        <div class="band-pin" style="left:{pos_pct}%">
          <div class="band-pin-val" style="left:0">{val_str} {unit}</div>
        </div>
      </div>
      <span class="band-chip {top_cls}">{top_lbl}</span>
    </div>"""


# ─────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────
def show_login():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px">
          <div style="display:inline-flex;align-items:center;gap:8px;margin-bottom:4px">
            <div style="width:10px;height:10px;border-radius:50%;background:#00916E;display:inline-block"></div>
            <span style="font-size:22px;font-weight:700;color:#0A2240">TIP ESG Platform</span>
          </div>
          <div style="font-size:13px;color:#6B7280">Tire Industry Project · Powered by dss+</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Sign in to your workspace")
            st.caption("dss+ employees use @consultdss.com · TIP member companies use their company email")

            email    = st.text_input("Email address", value="employee@consultdss.com",
                                     placeholder="you@consultdss.com or you@company.com")
            password = st.text_input("Password", type="password", value="demo1234")

            if st.button("Sign in", type="primary", use_container_width=True):
                email_l   = email.strip().lower()
                is_dss    = "@consultdss.com" in email_l
                is_client = email_l in CLIENTS
                if not is_dss and not is_client:
                    st.error("Email not recognised. Try employee@consultdss.com or verdatyres@tip-reporting.com")
                else:
                    name_parts = email.split("@")[0].replace(".", " ").split()
                    name       = " ".join(p.capitalize() for p in name_parts)
                    st.session_state.authenticated = True
                    st.session_state.user_email    = email_l
                    st.session_state.user_name     = name
                    st.session_state.is_dss        = is_dss
                    st.session_state.user_company  = "All Companies" if is_dss else CLIENTS[email_l]
                    st.session_state.page          = "entry"
                    st.rerun()

            st.caption("Demo: employee@consultdss.com (dss+ analyst) · verdatyres@tip-reporting.com (client, any password)")


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:18px 16px 14px;border-bottom:1px solid rgba(255,255,255,.1)">
          <div style="color:#fff;font-size:15px;font-weight:700">TIP ESG Platform</div>
          <div style="color:rgba(255,255,255,.4);font-size:11px;margin-top:3px">dss+ · Tire Industry Project</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.is_dss:
            # H3 FIX: escape company name before HTML injection
            _safe_co = _html.escape(st.session_state.user_company)
            st.markdown(f"""
            <div style="margin:10px 12px 0;padding:10px 12px;background:rgba(255,255,255,.06);border-radius:8px">
              <div style="color:rgba(255,255,255,.4);font-size:10px;text-transform:uppercase;letter-spacing:.6px">Company</div>
              <div style="color:#fff;font-size:13px;font-weight:500;margin-top:3px">{_safe_co}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="padding:12px 8px 4px;color:rgba(255,255,255,.3);font-size:10px;text-transform:uppercase;letter-spacing:.7px">Data & Reports</div>', unsafe_allow_html=True)

        for page_id, label in [("entry","KPI Data Entry"),("analysis","Analysis"),("benchmarking","Benchmarking")]:
            active = st.session_state.page == page_id
            if active:
                st.markdown(f"""<div style="background:rgba(0,145,110,0.85);border-radius:8px;padding:9px 12px;
                    margin-bottom:2px;color:#fff;font-size:13.5px;font-weight:600;">{label}</div>""", unsafe_allow_html=True)
            else:
                if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
                    st.session_state.page = page_id
                    st.rerun()

        if st.session_state.is_dss:
            st.markdown('<div style="padding:12px 8px 4px;color:rgba(255,255,255,.3);font-size:10px;text-transform:uppercase;letter-spacing:.7px;margin-top:8px">dss+ Internal</div>', unsafe_allow_html=True)
            for page_id, label in [("verification","Verification"),("readiness","AI Readiness")]:
                active = st.session_state.page == page_id
                if active:
                    st.markdown(f"""<div style="background:rgba(0,145,110,0.85);border-radius:8px;padding:9px 12px;
                        margin-bottom:2px;color:#fff;font-size:13.5px;font-weight:600;">{label}</div>""", unsafe_allow_html=True)
                else:
                    if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
                        st.session_state.page = page_id
                        st.rerun()

        st.markdown("---")
        # H3 FIX: escape all user-derived values before HTML injection
        name_init = _html.escape(
            "".join(p[0].upper() for p in st.session_state.user_name.split()[:2])
        )
        _safe_name = _html.escape(st.session_state.user_name)
        role_lbl   = "dss+ Employee" if st.session_state.is_dss else f"Client · {st.session_state.user_company}"
        _safe_role = _html.escape(role_lbl)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:9px">
          <div style="width:32px;height:32px;border-radius:50%;background:#00916E;color:#fff;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0">{name_init}</div>
          <div>
            <div style="color:#fff;font-size:13px;font-weight:500">{_safe_name}</div>
            <div style="color:rgba(255,255,255,.4);font-size:11px">{_safe_role}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ─────────────────────────────────────────────────────────
# STEPPER BAR
# ─────────────────────────────────────────────────────────
def render_stepper_bar():
    items = ""
    for i, (name, _) in enumerate(STEP_META):
        if i < st.session_state.step:
            sc, sl, icon = "sc-done",   "sl-done",   "v"
        elif i == st.session_state.step:
            sc, sl, icon = "sc-active", "sl-active", str(i+1)
        else:
            sc, sl, icon = "sc-todo",   "sl-todo",   str(i+1)
        line = f'<div class="step-line {"sl-done-line" if i>0 and i<=st.session_state.step else ""}"></div>' if i > 0 else ""
        items += f'{line}<div class="step-item"><div class="step-circle {sc}">{icon}</div><span class="step-label {sl}">{name}</span></div>'
    st.markdown(f'<div class="step-bar">{items}</div>', unsafe_allow_html=True)


STEP_FIELDS = [
    # Step 0: ISO 14001
    [("total_sites", "Total no. of sites",           "All facilities globally (no.)",              None),
     ("iso_sites",   "ISO 14001 certified sites",    "Sites with active ISO 14001 certification",  None)],
    # Step 1: Production
    [("production",  "Annual production",            "Total weight of finished products (metric T)", None)],
    # Step 2: Water
    [("water_withdrawals", "Total water withdrawals","All sources: surface, ground, municipal (m³)",None)],
    # Step 3: Energy
    [("renew_elec_purchased",    "Renewable electricity purchased",              "Grid-purchased certified renewable electricity (GJ)", None),
     ("nonrenew_elec_purchased", "Non-renewable electricity purchased",          "Standard grid electricity (GJ)",                      None),
     ("self_gen_elec",           "Self-generated renewable electricity on-site", "On-site solar, wind, hydro (GJ)",                     None),
     ("purchased_steam",         "Purchased steam",                              "GJ",                                                  None),
     ("sold_electricity",        "Sold electricity",                             "GJ (enter 0 if none)",                                None),
     ("sold_steam",              "Sold steam",                                   "GJ (enter 0 if none)",                                None),
     ("nat_gas",                 "Natural gas",                                  "GJ LHV",                                              None),
     ("coal_sub",                "Coal (all types)",                             "GJ LHV",                                              None),
     ("propane",                 "Propane",                                      "GJ LHV",                                              None),
     ("fuel_oil_heavy_a",        "Fuel oil",                                     "GJ LHV",                                              None),
     ("diesel",                  "Diesel",                                       "GJ LHV",                                              None),
     ("petrol",                  "Petrol",                                       "GJ LHV",                                              None),
     ("biomass",                 "Biomass",                                      "GJ LHV (biogenic CO2 excluded)",                      None),
     ("waste_tires_mt",          "Waste tires",                                  "metric T (converted to GJ internally)",               None),
     ("lpg",                     "LPG",                                          "GJ LHV",                                              None),
     ("other_fuels",             "Other fuels",                                  "GJ LHV",                                              None)],
    # Step 4: CO2
    [("co2_scope2_steam", "Scope 2 CO2 from purchased steam", "T.CO2 -- company-provided figure from steam supplier", None)],
    # Step 5: Waste
    [("waste_total",    "Total waste generated",  "metric T -- all waste streams",                  None),
     ("waste_recovery", "Waste sent to recovery", "metric T -- recycling, composting, energy rec.", None)],
]


# ─────────────────────────────────────────────────────────
# PAGE 1 -- KPI DATA ENTRY
# ─────────────────────────────────────────────────────────
def _build_master_row(inp, out) -> dict:
    """
    Build a dict whose keys exactly match the master wide CSV column names.
    Ensures no duplicate columns when appended to the master CSV.
    Includes waste KPIs and electricity-by-country columns.
    """
    renew_share  = (inp.renew_elec_purchased + inp.self_gen_elec) / max(out.total_electricity, 1) * 100
    scope1_share = out.total_co2_scope1 / max(out.total_co2, 1) * 100
    scope2_share = out.total_co2_scope2 / max(out.total_co2, 1) * 100
    fuel_total   = (inp.nat_gas + inp.coal_sub + inp.propane + inp.fuel_oil_heavy_a + inp.diesel + inp.petrol)
    fossil_share = fuel_total / max(out.total_energy, 1) * 100
    prod         = max(inp.production, 1)

    # Waste derived
    waste_total    = float(getattr(inp, "waste_total", 0) or 0)
    waste_recovery = float(getattr(inp, "waste_recovery", 0) or 0)
    recovery_rate  = round(waste_recovery / waste_total * 100, 4) if waste_total else 0.0
    waste_elim     = round(waste_total - waste_recovery, 4)

    row = {
        "Company": inp.company, "Year": inp.year,
        "Total no. of sites": int(round(inp.total_sites)),
        "ISO 14001 sites":    int(round(inp.iso_sites)),
        "% certified sites":  round(out.pct_certified, 6),
        "Production":         round(inp.production, 4),
        "Water intake":       round(inp.water_withdrawals, 4),
        "Water intake - KPI": round(out.water_kpi, 6),
        "Total Electricity":                               round(out.total_electricity, 4),
        "Renewable Electricity Purchased":                 round(inp.renew_elec_purchased, 4),
        "Non-Renewable Electricity Purchased":             round(inp.nonrenew_elec_purchased, 4),
        "Self-generated AND consumed electricity on-site": round(inp.self_gen_elec, 4),
        "Purchased Steam":   round(inp.purchased_steam, 4),
        "Sold Electricity":  round(inp.sold_electricity, 4),
        "Sold Steam":        round(inp.sold_steam, 4),
        "Natural Gas":       round(inp.nat_gas, 4),
        "Coal":              round(inp.coal_sub, 4),
        "Propane":           round(inp.propane, 4),
        "Fuel Oil":          round(inp.fuel_oil_heavy_a, 4),
        "Diesel":            round(inp.diesel, 4),
        "Petrol":            round(inp.petrol, 4),
        "Biomass":           round(inp.biomass, 4),
        "Waste tires":       round(inp.waste_tires_mt, 4),
        "LPG":               round(inp.lpg, 4),
        "Other":             round(inp.other_fuels, 4),
        "Total energy":          round(out.total_energy, 4),
        "Total energy - KPI":    round(out.energy_kpi, 6),
        "Total CO2 - Scope 1":   round(out.total_co2_scope1, 4),
        "Total CO2 - Scope 2":   round(out.total_co2_scope2, 4),
        "Total CO2":             round(out.total_co2, 4),
        "Total CO2 - KPI":       round(out.co2_kpi, 6),
        # ── Waste fields ──────────────────────────────────────────────────────
        "Total Waste":           round(waste_total, 4),
        "Waste Recovered":       round(waste_recovery, 4),
        "Recovery Rate":         recovery_rate,
        # ── Country electricity placeholders (filled by _save_electricity_to_master) ─
        **{_elec_col(c): None for c in ELEC_ALL_COUNTRIES},
        # ── Derived KPIs ──────────────────────────────────────────────────────
        "Renewable_Electricity_Share_%": round(renew_share, 4),
        "Scope1_Share_%":                round(scope1_share, 4),
        "Scope2_Share_%":                round(scope2_share, 4),
        "Fossil_Energy_Share_%":         round(fossil_share, 4),
        "Water_per_ton":                 round(inp.water_withdrawals / prod, 4),
        "CO2_per_ton":                   round(out.total_co2 / prod, 4),
        "Energy_per_ton":                round(out.total_energy / prod, 4),
        "ISO_Certification_%":           round(out.pct_certified * 100, 2),
        "Waste_Recovery_Rate_%":         recovery_rate,
        "Total_Electricity_by_Country_GJ": None,  # filled after country save
    }
    return row


def _save_version_parquet(inp, combined_df: pd.DataFrame) -> str:
    """
    Save the ENTIRE company template (all years) as a Parquet snapshot.
    Stored in data_storage/versions/{CompanyName}/ — subfolder only, never flat.
    Filename: CompanyName_Year_YYYYMMDD_HHMMSS.parquet (year = the year just edited).
    NEVER overwritten — each save event creates a new file (full audit trail).
    Reading this file shows the complete state of all years for that company
    at the exact moment the save was made.
    """
    from pathlib import Path
    from datetime import datetime
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    co_safe = inp.company.replace(" ", "_").replace("/", "_")
    # Extract ALL rows for this company from the combined master DataFrame
    company_all_years = combined_df[combined_df["Company"] == inp.company].copy()
    filename = f"{co_safe}_{inp.year}_{ts}.parquet"
    # Subfolder only — no flat file
    ver_dir  = Path("data_storage") / "versions" / co_safe
    ver_dir.mkdir(parents=True, exist_ok=True)
    try:
        company_all_years.to_parquet(ver_dir / filename, index=False)
        return f"{co_safe}/{filename}"
    except Exception as e:
        return f"[version save failed: {e}]"



def _drop_zero_elec_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Return df with Elec_*_GJ country columns removed if every value in that
    column is zero or null across all rows.  Non-electricity columns are
    never touched.  Used so files only carry countries with actual consumption.
    """
    elec_cols = [c for c in df.columns if c.startswith("Elec_") and c.endswith("_GJ")
                 and c != "Total_Electricity_by_Country_GJ"]
    zero_cols = [c for c in elec_cols
                 if df[c].fillna(0).eq(0).all()]
    return df.drop(columns=zero_cols) if zero_cols else df

def _sync_consolidate_excel(master_df: "pd.DataFrame") -> None:
    """
    Sync the CONSOLIDATED_DUMMY Excel (Raw Dummy data sheet) from the master wide CSV.

    The Raw Dummy data sheet stores data in long format: one row per
    (Company, Year, Row_Label). This function overwrites it completely from
    the current master wide DataFrame so that the consolidate stays in sync
    after any save from the platform.
    """
    from pathlib import Path
    from openpyxl import load_workbook

    xl_path = Path("data_storage/master/CONSOLIDATED_DUMMY_2009_2023.xlsx")
    if not xl_path.exists():
        return  # nothing to sync yet

    # Mapping: wide-CSV column  →  (Section, Row_Label)
    COL_MAP = {
        "Total no. of sites":                              ("ISO 14001",    "Total no. of sites"),
        "ISO 14001 sites":                                 ("ISO 14001",    "ISO 14001 sites"),
        "% certified sites":                               ("ISO 14001",    "% certified sites"),
        "Production":                                      ("Production",   "Production"),
        "Water intake":                                    ("Water",        "Water intake"),
        "Water intake - KPI":                              ("Water",        "Water intake - KPI"),
        "Total Electricity":                               ("Energy",       "Total Electricity"),
        "Renewable Electricity Purchased":                 ("Energy",       "Renewable Electricity Purchased"),
        "Non-Renewable Electricity Purchased":             ("Energy",       "Non-Renewable Electricity Purchased"),
        "Self-generated AND consumed electricity on-site": ("Energy",       "Self-generated AND consumed electricity on-site"),
        "Purchased Steam":                                 ("Energy",       "Purchased Steam"),
        "Sold Electricity":                                ("Energy",       "Sold Electricity"),
        "Sold Steam":                                      ("Energy",       "Sold Steam"),
        "Natural Gas":                                     ("Energy",       "Natural Gas"),
        "Coal":                                            ("Energy",       "Coal"),
        "Propane":                                         ("Energy",       "Propane"),
        "Fuel Oil":                                        ("Energy",       "Fuel Oil"),
        "Diesel":                                          ("Energy",       "Diesel"),
        "Petrol":                                          ("Energy",       "Petrol"),
        "Biomass":                                         ("Energy",       "Biomass"),
        "Waste tires":                                     ("Energy",       "Waste tires"),
        "LPG":                                             ("Energy",       "LPG"),
        "Other":                                           ("Energy",       "Other"),
        "Total energy":                                    ("Energy",       "Total energy"),
        "Total energy - KPI":                              ("Energy",       "Total energy - KPI"),
        "Total CO2 - Scope 1":                             ("CO2 emissions","Total CO2 - Scope 1"),
        "Total CO2 - Scope 2":                             ("CO2 emissions","Total CO2 - Scope 2"),
        "Total CO2":                                       ("CO2 emissions","Total CO2"),
        "Total CO2 - KPI":                                 ("CO2 emissions","Total CO2 - KPI"),
        "Total Waste":                                     ("Waste",        "Total Waste"),
        "Waste Recovered":                                 ("Waste",        "Waste Recovered"),
        "Recovery Rate":                                   ("Waste",        "Recovery Rate"),
        **{_elec_col(c): ("Energy", f"Electricity - {c}") for c in ELEC_ALL_COUNTRIES},
    }

    # Build long rows from master_df
    long_rows = []  # list of dicts: Company, Row, Year, Data, Section, Row_Label, Notes, Consistency test
    row_order = list(COL_MAP.keys())
    # Assign fixed row numbers to match what build_esg_master.py uses
    ROW_NUM = {col: i + 1 for i, col in enumerate(row_order)}

    # Pre-compute which electricity country columns have any non-zero value
    # across the whole master — only those countries get rows in the consolidate.
    active_elec_cols = {
        col for col in COL_MAP
        if col.startswith("Elec_") and col.endswith("_GJ")
        and col in master_df.columns
        and master_df[col].fillna(0).ne(0).any()
    }

    for _, wrow in master_df.sort_values(["Company", "Year"]).iterrows():
        company = wrow["Company"]
        year    = int(wrow["Year"]) if pd.notna(wrow.get("Year")) else None
        if not company or not year:
            continue
        for col, (section, label) in COL_MAP.items():
            # Skip electricity country columns that are all-zero across the dataset
            is_elec_country = col.startswith("Elec_") and col.endswith("_GJ")
            if is_elec_country and col not in active_elec_cols:
                continue
            val = wrow.get(col)
            # For an active electricity country, skip rows where this company-year is zero
            if is_elec_country and (pd.isna(val) or float(val) == 0):
                continue
            long_rows.append({
                "Company": company,
                "Row":     ROW_NUM[col],
                "Year":    year,
                "Data":    float(val) if pd.notna(val) else None,
                "Section": section,
                "Row_Label": label,
                "Notes":   None,
                "Consistency test": None,
            })

    if not long_rows:
        return

    try:
        wb = load_workbook(xl_path)
        ws = wb["Raw Dummy data"]
        # Clear existing data rows (keep header row 1)
        for r in range(2, ws.max_row + 1):
            for c in range(1, 9):
                ws.cell(r, c).value = None
        # Write new rows
        cols = ["Company", "Row", "Year", "Data", "Section", "Row_Label", "Notes", "Consistency test"]
        for i, row in enumerate(long_rows):
            for j, col in enumerate(cols):
                ws.cell(i + 2, j + 1).value = row[col]
        wb.save(xl_path)
    except (PermissionError, OSError):
        pass  # file locked — skip, master CSV is the source of truth
    except Exception:
        pass  # any other error is also non-fatal


def _sync_company_member_files(master_df: "pd.DataFrame") -> list:
    """
    Write per-company CSVs in data_storage/members/TIP/<CompanyName>/<CompanyName>_latest.csv
    from the current master wide DataFrame.
    Skips any file that is locked (e.g. open in Excel) instead of crashing.
    Returns list of company names that were skipped.
    """
    from pathlib import Path
    members_tip = Path("data_storage/members/TIP")
    skipped = []
    for company, grp in master_df.groupby("Company"):
        co_safe   = str(company).replace(" ", "_")
        co_folder = members_tip / co_safe
        co_folder.mkdir(parents=True, exist_ok=True)
        try:
            # Drop electricity country columns that are all zero for this company
            grp_clean = _drop_zero_elec_cols(grp.reset_index(drop=True))
            grp_clean.to_csv(co_folder / f"{co_safe}_latest.csv", index=False)
        except (PermissionError, OSError):
            skipped.append(str(company))
    return skipped


def _save_submission_to_csv(inp, out) -> str:
    """
    Three independent operations:

    1. MASTER CSV (data_storage/master/) — overwrite the row for this company+year.
       The master always holds the LATEST values. Second save for same company+year
       replaces the first row.

    2. VERSION Parquet (data_storage/versions/) — always ADD a new timestamped file.
       Never overwritten. Full audit trail of every save event.

    3. SYNC (after master is written):
       - CONSOLIDATED_DUMMY Excel Raw Dummy data sheet (long format)
       - Per-company CSVs in data_storage/members/TIP/<Company>/
       - TIP members aggregate CSV
    """
    import os, tempfile
    from pathlib import Path
    from datetime import datetime

    csv_path = Path("data_storage/master/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row     = pd.DataFrame([_build_master_row(inp, out)])
    master_cols = list(new_row.columns)

    def _align(df):
        """Align DataFrame to master column order: strip extras, fill missing."""
        if df.empty:
            return pd.DataFrame(columns=master_cols)
        extra = [c for c in df.columns if c not in master_cols]
        if extra:
            df = df.drop(columns=extra)
        for col in master_cols:
            if col not in df.columns:
                df[col] = None
        return df[master_cols]

    def _load_best_existing():
        """
        Load the most complete existing master DataFrame.
        Checks all candidate paths and picks the one with the most rows.
        """
        candidates = [
            csv_path,
            Path("data_storage/raw/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"),
        ]
        best = pd.DataFrame(columns=master_cols)
        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if "Company" in df.columns and "Year" in df.columns and len(df) > len(best):
                        best = df
                except PermissionError:
                    pass
                except Exception:
                    pass
        return _align(best)

    # ── 1. Build combined DataFrame ──────────────────────────────────────────
    # H1 FIX: advisory file lock held for the full read-modify-write cycle.
    # Prevents data corruption if two analysts save simultaneously.
    # Timeout=10s: if another process holds the lock and crashes, we don't
    # block forever.  The PermissionError branch below still handles Excel locks.
    lock_path = csv_path.with_suffix(".lock")
    with FileLock(str(lock_path), timeout=10):
        existing = _load_best_existing()
        mask     = ~((existing["Company"] == inp.company) & (existing["Year"] == inp.year))
        existing = existing[mask]
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined = combined.sort_values(["Company", "Year"]).reset_index(drop=True)

        n_records   = len(combined)
        n_companies = combined["Company"].nunique()

        # ── 2. Save version Parquet BEFORE touching master (audit trail first) ───
        version_filename = _save_version_parquet(inp, combined)

        # ── 3. Write master CSV, then sync all dependent files ───────────────────
        try:
            # Master CSV keeps all country columns (even all-zero) as the full schema.
            # Derived outputs (member files, TIP aggregate) strip all-zero country cols.
            combined.to_csv(csv_path, index=False)
            # Sync TIP members aggregate
            tip_master_path = Path("data_storage/members/TIP/ESG_MASTER_WIDE_TIP_MEMBERS_2009_2023.csv")
            _update_tip_members_file(csv_path, tip_master_path)
            # Sync per-company member files
            _sync_company_member_files(combined)
            # Sync CONSOLIDATED_DUMMY Excel
            _sync_consolidate_excel(combined)
            return (f"✅ Saved {inp.company} — {inp.year}. "
                    f"Master: {n_records} records across {n_companies} companies. "
                    f"Version: {version_filename}")
        except PermissionError:
            ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"ESG_MASTER_{inp.company.replace(' ','_')}_{inp.year}_{ts}.csv"
            backup_path = csv_path.parent / backup_name
            try:
                combined.to_csv(backup_path, index=False)
                return (
                    f"⚠️ Master file open in Excel — saved backup: **{backup_name}**\n"
                    f"Version snapshot: {version_filename}\n"
                    f"Close Excel and click Save again."
                )
            except Exception as e2:
                return f"❌ Save failed (file locked AND backup failed): {e2}"
        except Exception as e:
            return f"❌ Save failed: {e}"


def page_entry():
    rep_year = st.session_state.get("reporting_year", CURR_YEAR)
    st.markdown(f"## {'KPI Data Entry' if not st.session_state.template_done else f'ESG KPI Template — {rep_year}'}")

    if st.session_state.template_done:
        col_a, col_b = st.columns([5, 1])
        with col_b:
            if st.button("Edit Inputs"):
                st.session_state.template_done      = False
                st.session_state.step               = 0
                st.session_state.company_setup_done = False
                st.rerun()

        # -- Save to database panel ------------------------------------------
        _save_company = st.session_state.get("reporting_company") or st.session_state.get("user_company", "")
        _save_year    = st.session_state.get("reporting_year", CURR_YEAR)
        with st.expander(f"💾  Save {_save_company} {_save_year} to master database", expanded=False):
            st.markdown(
                f"Saves **KPI data** (Main Data Input) **and Electricity by Country** for "
                f"**{_save_company} — {_save_year}** into the master CSV and TIP members file.  "
                f"Existing record for this company and year will be overwritten."
            )
            col_sv1, col_sv2 = st.columns([3, 1])
            with col_sv2:
                if st.button("Save to database", type="primary", use_container_width=True, key="save_db_btn"):
                    _save_inp, _save_out = get_current_outputs()
                    # Step 1: Save KPI data
                    _msg_kpi = _save_submission_to_csv(_save_inp, _save_out)
                    # Step 2: Save electricity-by-country data (if any entered)
                    _msg_elec = ""
                    if st.session_state.get("elec_data") is not None:
                        _msg_elec = _save_electricity_to_master(_save_company, _save_year)
                    # Show combined result
                    if _msg_kpi.startswith("✅"):
                        st.success("✅ Saved successfully — added to your database.")
                    else:
                        st.error(_msg_kpi)
            with col_sv1:
                pass

        tab_main, tab_elec, tab_waste, tab_qual, tab_conv = st.tabs([
            "Main Data Input", "Electricity by Country", "Waste", "Qualitative Data", "Conversion Tables",
        ])
        with tab_main:  render_template_table()
        with tab_elec:  render_electricity_tab()
        with tab_waste: render_waste_tab()
        with tab_qual:  render_qualitative_tab()
        with tab_conv:  render_conversion_tab()
        return

    # -- Company Setup Pre-Step -------------------------------------------------
    if not st.session_state.company_setup_done:
        with st.container(border=True):
            st.markdown("### Step 0 of 6 — Company Setup")
            st.caption("Select your company and reporting year. Historical data will be pre-loaded from the consolidated dataset.")
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                company_options = COMPANIES if COMPANIES else ["(No data loaded)"]
                login_company   = st.session_state.user_company
                default_idx     = 0
                if login_company and login_company in company_options:
                    default_idx = company_options.index(login_company)

                sel_company = st.selectbox("Company name", options=company_options,
                                           index=default_idx,
                                           help="Select the TIP member company for this submission.")
                emp_name = st.text_input("Employee name (for records)",
                                         value=st.session_state.user_name,
                                         placeholder="Full name of person completing this submission")

            with col2:
                avail_years = sorted(
                    [int(y) for y in _CONSOLIDATED_DF["Year"].dropna().unique()]
                    if not _CONSOLIDATED_DF.empty else [2023], reverse=True
                )
                rep_year_sel = st.selectbox(
                    "Reporting year (data you are submitting)",
                    options=avail_years + ([max(avail_years)+1] if avail_years else [2024]),
                    index=0,
                    help="The calendar year the KPI data covers.",
                )
                n_hist = len([y for y in avail_years if y < rep_year_sel])
                st.info(f"**{sel_company}** has **{n_hist} years** of historical data available (through {max(avail_years) if avail_years else '—'}).")

            st.divider()
            if st.button("Load historical data and start entry", type="primary"):
                hist      = dl.get_company_hist(_CONSOLIDATED_DF, sel_company)
                prev_data = dl.get_step_data(hist, rep_year_sel - 1)
                hist_raw  = dl.get_hist_raw(hist, HIST_YEARS)
                kpi_hints = dl.get_kpi_hints(_CONSOLIDATED_DF, sel_company, rep_year_sel - 1)

                st.session_state.reporting_company  = sel_company
                st.session_state.reporting_year     = rep_year_sel
                st.session_state.employee_name      = emp_name
                st.session_state.company_hist       = hist
                st.session_state.live_hist_raw      = hist_raw
                st.session_state.kpi_hints          = kpi_hints

                if prev_data:
                    for field, val in prev_data.items():
                        if field in st.session_state.step_data:
                            st.session_state.step_data[field] = val

                st.session_state.company_setup_done = True
                st.rerun()
        return

    # -- Stepper ----------------------------------------------------------------
    render_stepper_bar()
    _hist    = _get_fresh_hist()   # always fresh — reflects any saved updates
    _prev_yr = st.session_state.get("reporting_year", CURR_YEAR) - 1
    step     = st.session_state.step
    name, desc = STEP_META[step]
    fields   = STEP_FIELDS[step]

    with st.container(border=True):
        st.markdown(f"### Step {step+1} of {len(STEP_META)} — {name}")
        st.caption(desc)

        # Prior-year KPI reference hints
        kpi_hints = st.session_state.get("kpi_hints", {})
        if kpi_hints:
            sel_co    = st.session_state.get("reporting_company", "")
            hint_parts = []
            if step == 2 and "water_kpi"  in kpi_hints: hint_parts.append(f"Water KPI {_prev_yr}: **{kpi_hints['water_kpi']:.2f} m³/T**")
            if step in (3,4) and "energy_kpi" in kpi_hints: hint_parts.append(f"Energy KPI {_prev_yr}: **{kpi_hints['energy_kpi']:.2f} GJ/T**")
            if step == 4 and "co2_kpi"    in kpi_hints: hint_parts.append(f"CO2 KPI {_prev_yr}: **{kpi_hints['co2_kpi']:.3f} T/T**")
            if step == 0 and "iso_pct"    in kpi_hints: hint_parts.append(f"ISO certified {_prev_yr}: **{kpi_hints['iso_pct']*100:.1f}%**")
            if hint_parts:
                st.info(f"{sel_co} prior-year reference — " + " · ".join(hint_parts))

        st.divider()
        inp, out = get_current_outputs()
        n_cols   = 2 if len(fields) > 3 else 1
        cols_list = st.columns(n_cols)

        for idx, fdef in enumerate(fields):
            key, label, sublabel = fdef[0], fdef[1], fdef[2] or ""
            hist_vals = _hist.get(key, [])
            hist_val  = hist_vals[-1] if hist_vals else None
            with cols_list[idx % n_cols]:
                if sublabel:
                    st.caption(sublabel)
                val = st.number_input(label,
                                      value=float(st.session_state.step_data.get(key, 0)),
                                      step=1.0, format="%.0f", key=f"input_{key}")
                st.session_state.step_data[key] = val
                if hist_val:
                    st.caption(f"{_prev_yr} reference: {fmt_num(hist_val)}")

        # Live preview
        st.divider()
        inp2, out2 = get_current_outputs()
        if step == 0:
            st.metric("% ISO Certified", f"{out2.pct_certified*100:.1f}%")
        elif step == 1:
            st.metric("Production entered", fmt_num(inp2.production) + " metric T")
        elif step == 2:
            c1, c2 = st.columns(2)
            c1.metric("Water withdrawals",  fmt_num(inp2.water_withdrawals) + " m³")
            c2.metric("Water intensity KPI", f"{out2.water_kpi:.2f} m³/T")
        elif step == 3:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total electricity",   fmt_num(out2.total_electricity) + " GJ")
            c2.metric("Total energy",         fmt_num(out2.total_energy) + " GJ")
            c3.metric("Energy intensity KPI", f"{out2.energy_kpi:.2f} GJ/T")
        elif step == 4:
            c1, c2, c3 = st.columns(3)
            c1.metric("Scope 1 CO₂", fmt_num(out2.total_co2_scope1) + " T")
            c2.metric("Scope 2 CO₂", fmt_num(out2.total_co2_scope2) + " T")
            c3.metric("CO₂ intensity KPI", f"{out2.co2_kpi:.3f} T/T")
        elif step == 5:
            c1, c2, c3 = st.columns(3)
            c1.metric("Waste elimination", fmt_num(out2.waste_elimination) + " T")
            c2.metric("Recovery rate",     f"{out2.waste_recovery_pct*100:.1f}%")
            c3.metric("Waste check",       "Consistent" if out2.check_waste else "Inconsistent")

        st.divider()
        nav_l, nav_r = st.columns(2)
        with nav_l:
            if step > 0 and st.button("Previous step"):
                st.session_state.step -= 1; st.rerun()
        with nav_r:
            if step < len(STEP_META) - 1:
                if st.button("Save & Continue", type="primary"):
                    st.session_state.step += 1; st.rerun()
            else:
                if st.button("Generate Template", type="primary"):
                    st.session_state.template_done = True; st.rerun()


# ─────────────────────────────────────────────────────────
# TEMPLATE TABLE
# ─────────────────────────────────────────────────────────
def render_template_table():
    company  = st.session_state.get("reporting_company") or st.session_state.get("user_company") or "TIP Member Company"
    if company == "All Companies": company = "TIP Member Company"
    rep_year = st.session_state.get("reporting_year", CURR_YEAR)
    # Always reload from _CONSOLIDATED_DF so updates to any year are visible
    _hist    = _get_fresh_hist(company)

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
        background:#fff;border:1px solid #E5E7EB;border-radius:10px;
        padding:18px 24px;margin-bottom:14px">
      <div>
        <div style="font-size:17px;font-weight:700;color:#0A2240;letter-spacing:-.2px">
          Tire Industry Project — Key Performance Indicators
        </div>
        <div style="font-size:26px;font-weight:800;color:#00916E;margin-top:5px;letter-spacing:-.4px">
          {_html.escape(company)}
        </div>
        <div style="font-size:12px;color:#9CA3AF;margin-top:4px">Corporate units · ESG KPI Template — {rep_year}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:.5px">Reporting year</div>
        <div style="font-size:36px;font-weight:800;color:#0A2240;line-height:1">{rep_year}</div>
        <div style="font-size:11px;color:#9CA3AF;margin-top:3px">Data range: 2009–{rep_year}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("Template generated from your inputs. Blue cells = company input, grey italic = auto-calculated formula.")

    inp, out = get_current_outputs()
    hist     = get_hist_outputs()

    ROWS = [
        ("section","ISO 14001",None,None,None),
        ("input","Total no. of sites","no.","total_sites",None),
        ("input","ISO 14001 certified sites","no.","iso_sites",None),
        ("calc","% certified sites","%",None,lambda i,o:f"{o.pct_certified*100:.1f}%"),
        ("section","Production",None,None,None),
        ("input","Production","metric T","production",None),
        ("section","Water",None,None,None),
        ("input","Water withdrawals","m³","water_withdrawals",None),
        ("calc","Water intensity KPI","m³/T",None,lambda i,o:f"{o.water_kpi:.2f}"),
        ("section","Energy",None,None,None),
        ("calc","Total Electricity","GJ",None,lambda i,o:f"{o.total_electricity:,.0f}"),
        ("input","— Renewable electricity purchased","GJ","renew_elec_purchased",None),
        ("input","— Non-renewable electricity purchased","GJ","nonrenew_elec_purchased",None),
        ("input","— Self-generated renewable on-site","GJ","self_gen_elec",None),
        ("input","Purchased Steam","GJ","purchased_steam",None),
        ("input","Sold Electricity","GJ","sold_electricity",None),
        ("input","Sold Steam","GJ","sold_steam",None),
        ("input","Natural Gas","GJ LHV","nat_gas",None),
        ("input","Coal (all types)","GJ LHV","coal_sub",None),
        ("input","Propane","GJ LHV","propane",None),
        ("input","Fuel Oil","GJ LHV","fuel_oil_heavy_a",None),
        ("input","Diesel","GJ LHV","diesel",None),
        ("input","Petrol","GJ LHV","petrol",None),
        ("input","Biomass","GJ LHV","biomass",None),
        ("input","Waste tires","metric T","waste_tires_mt",None),
        ("input","LPG","GJ LHV","lpg",None),
        ("input","Other fuels","GJ LHV","other_fuels",None),
        ("calc","TOTAL ENERGY","GJ",None,lambda i,o:f"{o.total_energy:,.0f}"),
        ("calc","Energy intensity KPI","GJ/T",None,lambda i,o:f"{o.energy_kpi:.2f}"),
        ("section","CO2 Emissions",None,None,None),
        ("input","Scope 2 — Steam","T.CO2","co2_scope2_steam",None),
        ("calc","CO2 — Natural Gas","T.CO2",None,lambda i,o:f"{o.co2_nat_gas:,.0f}"),
        ("calc","CO2 — Coal","T.CO2",None,lambda i,o:f"{o.co2_coal:,.0f}"),
        ("calc","CO2 — Propane","T.CO2",None,lambda i,o:f"{o.co2_propane:,.0f}"),
        ("calc","CO2 — Fuel Oil","T.CO2",None,lambda i,o:f"{o.co2_fuel_oil:,.0f}"),
        ("calc","CO2 — Diesel","T.CO2",None,lambda i,o:f"{o.co2_diesel:,.0f}"),
        ("calc","CO2 — Petrol","T.CO2",None,lambda i,o:f"{o.co2_petrol:,.0f}"),
        ("calc","CO2 — LPG","T.CO2",None,lambda i,o:f"{o.co2_lpg:,.0f}"),
        ("calc","TOTAL CO2 Scope 1","T.CO2",None,lambda i,o:f"{o.total_co2_scope1:,.0f}"),
        ("calc","TOTAL CO2 Scope 2","T.CO2",None,lambda i,o:f"{o.total_co2_scope2:,.0f}"),
        ("calc","TOTAL CO2 (S1+S2)","T.CO2",None,lambda i,o:f"{o.total_co2:,.0f}"),
        ("calc","CO2 intensity KPI","T.CO2/T",None,lambda i,o:f"{o.co2_kpi:.3f}"),
        ("section","Waste",None,None,None),
        ("input","Total waste generated","metric T","waste_total",None),
        ("input","Waste sent to recovery","metric T","waste_recovery",None),
        ("calc","Waste sent to elimination","metric T",None,lambda i,o:f"{o.waste_elimination:,.0f}"),
        ("calc","Recovery rate","%",None,lambda i,o:f"{o.waste_recovery_pct*100:.1f}%"),
        ("calc","Waste intensity KPI","kg/T",None,lambda i,o:f"{i.waste_total/i.production*1000:.1f}" if i.production else "—"),
    ]

    data = []
    for rdef in ROWS:
        rtype, label, unit, key, fn = rdef
        if rtype == "section":
            row = {"Indicator": f"▸ {label}", "Unit": ""}
            for yr, hi, ho in hist: row[str(yr)] = ""
            row[str(rep_year)] = ""; row["YoY %"] = ""
            data.append({"_type":"section","_row":row}); continue

        row = {"Indicator": label, "Unit": unit or ""}
        hist_nums = []
        for yr, hi, ho in hist:
            v = getattr(hi, key, None) if key else None
            if v is None and fn: v = fn(hi, ho)
            try:
                row[str(yr)] = f"{int(round(float(v))):,}"
            except (TypeError, ValueError):
                row[str(yr)] = str(v) if v else "—"
            try: hist_nums.append(float(str(v).replace(",","").replace("%","").replace("—","0")))
            except: hist_nums.append(0)

        cv = getattr(inp, key, None) if key else None
        if cv is None and fn: cv = fn(inp, out)
        row[str(rep_year)] = (f"{float(cv):,.0f}" if isinstance(cv,(int,float)) and not isinstance(cv,str) else (cv if cv else "—"))

        try:
            cn  = float(str(cv).replace(",","").replace("%",""))
            pn  = hist_nums[-1] if hist_nums else 0
            yoy = (cn - pn) / abs(pn) * 100 if pn else None
            row["YoY %"] = f"{yoy:+.1f}%" if yoy is not None else "—"
        except: row["YoY %"] = "—"
        data.append({"_type":rtype,"_row":row,"_key":key,"_label":label})

    all_rows  = [d["_row"]  for d in data]
    all_types = [d["_type"] for d in data]
    df_tbl    = pd.DataFrame(all_rows)
    curr_col  = str(rep_year)

    def style_row(row, idx):
        rt = all_types[idx]
        return [
            ("background-color:#E8F5F0;color:#065F46;font-weight:800;font-size:13px;"
             "border-top:2px solid #6EE7B7;padding-top:8px;padding-bottom:8px;"
             "letter-spacing:.3px;text-transform:uppercase") if rt == "section"
            else "background-color:#DBEAFE;color:#1E40AF;font-weight:700" if (col == curr_col and rt == "input")
            else "background-color:#EFF6FF;color:#1D4ED8;font-style:italic" if (col == curr_col and rt == "calc")
            else "background-color:#F8FAFC;color:#6B7280;font-style:italic" if rt == "calc"
            else "background-color:#F0F9FF;"
            for col in df_tbl.columns
        ]

    styled     = df_tbl.style.apply(lambda row: style_row(row, row.name), axis=1)
    tbl_height = min(900, max(400, len(all_rows)*36+60))
    st.dataframe(styled, hide_index=True, height=tbl_height, use_container_width=True)
    st.markdown(f"""<div class="tbl-legend">
      <div class="tl"><div class="tl-sw" style="background:#F0F9FF;border-color:#BAE6FD"></div>Company input (historical)</div>
      <div class="tl"><div class="tl-sw" style="background:#DBEAFE;border-color:#93C5FD"></div>Company input ({rep_year})</div>
      <div class="tl"><div class="tl-sw" style="background:#EFF6FF;border-color:#A5B4FC"></div>Auto-calculated ({rep_year})</div>
      <div class="tl"><div class="tl-sw" style="background:#F8FAFC;border-color:#E5E7EB"></div>Auto-calculated (historical)</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# ELECTRICITY TAB
# ─────────────────────────────────────────────────────────
def _update_tip_members_file(master_path: "Path", tip_master_path: "Path") -> None:
    """Rebuild the TIP members aggregate strictly from the latest master on disk.

    This prevents mismatches where the in-memory combined_df used during save
    (bootstrap/reconstruction) differs from the finally-written master CSV.
    """
    import pandas as pd
    tip_master_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        master_df = pd.read_csv(master_path)
    except Exception as e:
        print(f"[tip_members] Could not read master to rebuild tip members: {e}")
        return

    # In the current dataset all companies are TIP members, so TIP aggregate
    # is identical to master — but strip all-zero electricity country columns.
    try:
        _drop_zero_elec_cols(master_df).to_csv(tip_master_path, index=False)
    except Exception as e:
        print(f"[tip_members] Could not write: {e}")



def _save_electricity_to_master(company: str, year: int) -> str:
    """
    Save electricity-by-country data (from the Electricity tab editor) into:
      1. Master wide CSV  — columns Elec_<Country>_GJ  (GJ = MWh x 3.6)
      2. TIP members aggregate CSV
      3. Per-company member CSVs in data_storage/members/TIP/<Company>/
      4. CONSOLIDATED_DUMMY Excel (Raw Dummy data sheet, long format)
      5. Parquet snapshot of the complete company+year row

    Updates ALL years that have non-zero values in the electricity editor.
    Only the 7 countries already in the master schema are written:
        Canada, Mexico, United States, Japan, France, Hungary, Italy
    Any other country rows in the editor UI are displayed but not persisted.
    """
    from pathlib import Path
    from datetime import datetime

    COUNTRY_COL = ELEC_COUNTRY_COLS  # all 31 countries
    MWH_TO_GJ = 3.6

    elec_df = st.session_state.get("elec_data", pd.DataFrame())
    if elec_df.empty:
        return "No electricity data entered yet."

    csv_path = Path("data_storage/master/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv")
    if not csv_path.exists():
        return "Master CSV not found. Save KPI data first."

    try:
        master = pd.read_csv(csv_path)
    except PermissionError:
        return "Master CSV is open in Excel — close it and try again."

    # Ensure all country columns exist in master (add if missing)
    for col in COUNTRY_COL.values():
        if col not in master.columns:
            master[col] = None
    if "Total_Electricity_by_Country_GJ" not in master.columns:
        master["Total_Electricity_by_Country_GJ"] = None

    yr_cols = [c for c in elec_df.columns if str(c).isdigit() and 2000 < int(c) < 2030]

    updated_years = []
    for yr_str in yr_cols:
        yr   = int(yr_str)
        mask = (master["Company"] == company) & (master["Year"] == yr)
        if not mask.any():
            continue

        year_series = elec_df.set_index("Country")[yr_str]
        for country, mwh_val in year_series.items():
            col_name = COUNTRY_COL.get(str(country))
            if col_name is None:
                continue  # country not in master schema
            gj_val = float(mwh_val) * MWH_TO_GJ if pd.notna(mwh_val) else 0.0
            master.loc[mask, col_name] = round(gj_val, 4)

        # Recompute total-by-country for this row
        country_vals = [master.loc[mask, c].values[0]
                        for c in COUNTRY_COL.values() if c in master.columns]
        master.loc[mask, "Total_Electricity_by_Country_GJ"] = round(
            sum(v for v in country_vals if pd.notna(v)), 4)

        updated_years.append(yr)

    if not updated_years:
        return f"No KPI rows found for {company}. Save KPI data first."

    try:
        # H1 FIX: use the same advisory lock as the KPI save path
        lock_path = csv_path.with_suffix(".lock")
        with FileLock(str(lock_path), timeout=10):
            master.to_csv(csv_path, index=False)
    except PermissionError:
        return "Master CSV is open in Excel — close it and try again."

    # Sync all dependent files
    tip_master_path = Path("data_storage/members/TIP/ESG_MASTER_WIDE_TIP_MEMBERS_2009_2023.csv")
    _update_tip_members_file(csv_path, tip_master_path)
    _sync_company_member_files(master)
    _sync_consolidate_excel(master)

    # Parquet snapshot
    co_safe  = company.replace(" ", "_").replace("/", "_")
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    ver_dir  = Path("data_storage") / "versions" / co_safe
    ver_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{co_safe}_{year}_elec_{ts}.parquet"
    single_row = master[(master["Company"] == company) & (master["Year"] == year)].copy()
    try:
        single_row.to_parquet(ver_dir / filename, index=False)
    except Exception:
        filename = "[parquet skipped]"

    return (f"Electricity saved — {len(updated_years)} year(s) updated "
            f"({min(updated_years)}-{max(updated_years)}) converted MWh to GJ. "
            f"Consolidate + member files synced. "
            f"Snapshot: versions/{co_safe}/{filename}")


def render_electricity_tab():
    """
    Electricity-by-country editor.

    Fix for two bugs:
    1. VALUES RESET BUG — st.data_editor with a static key causes Streamlit to
       discard edits on the first rerun. Fix: never use a static key on the
       data_editor when its underlying data comes from session_state. Instead
       read the widget result back via `on_change` / direct assignment and
       give the editor a key that is stable only within one company+year session,
       so it re-initialises exactly when the company or year changes.

    2. PRE-LOAD BUG — elec_data was always initialised to zeros even when the
       master CSV already had non-zero Elec_*_GJ values for this company.
       Fix: on first load (or when company/year changes) read Elec_*_GJ cols
       from _CONSOLIDATED_DF, convert GJ→MWh, and populate the editor.
    """
    # Countries shown in the UI (all 31 — display-only for countries not in master schema)
    ELEC_COUNTRIES = ELEC_ALL_COUNTRIES  # module-level list of all 31
    COUNTRY_COL_GJ = ELEC_COUNTRY_COLS  # all 31 countries stored in master
    GJ_TO_MWH = 1.0 / 3.6
    YEARS = list(range(2009, 2024))

    company  = st.session_state.get("reporting_company") or st.session_state.get("user_company", "")
    rep_year = st.session_state.get("reporting_year", CURR_YEAR)

    # ── Key that tracks which company+year the editor was last initialised for ──
    # When this changes we rebuild elec_data from the master so the editor
    # always shows what is actually stored in the DB.
    load_key = f"{company}|{rep_year}"
    needs_reload = st.session_state.get("_elec_load_key") != load_key

    if needs_reload:
        # Build base DataFrame of zeros
        rows = [{"Country": c, "Unit": "MWh", **{str(yr): 0.0 for yr in YEARS}}
                for c in ELEC_COUNTRIES]
        df = pd.DataFrame(rows)

        # Pre-populate from master CSV for countries that are stored
        if not _CONSOLIDATED_DF.empty and company:
            for country, col_gj in COUNTRY_COL_GJ.items():
                if col_gj not in _CONSOLIDATED_DF.columns:
                    continue
                co_df = _CONSOLIDATED_DF[_CONSOLIDATED_DF["Company"] == company]
                for _, mrow in co_df.iterrows():
                    yr = int(mrow["Year"]) if pd.notna(mrow.get("Year")) else None
                    if yr is None or yr not in YEARS:
                        continue
                    gj_val = mrow.get(col_gj)
                    if pd.notna(gj_val) and float(gj_val) != 0:
                        mwh_val = round(float(gj_val) * GJ_TO_MWH, 2)
                        idx = df.index[df["Country"] == country]
                        if len(idx):
                            df.loc[idx[0], str(yr)] = mwh_val

        # Ensure all year columns are numeric (avoid object dtype after assignment)
        for yr in YEARS:
            df[str(yr)] = pd.to_numeric(df[str(yr)], errors="coerce").fillna(0.0)

        st.session_state.elec_data     = df
        st.session_state._elec_load_key = load_key
        # Drop the old widget key so Streamlit re-renders a fresh editor
        if "_elec_editor_key_idx" not in st.session_state:
            st.session_state._elec_editor_key_idx = 0
        st.session_state._elec_editor_key_idx += 1

    # ── Editor key: unique per company+year so Streamlit does not reuse ───────
    # the old internal widget state (which is what causes edits to be lost).
    editor_key = f"elec_editor_{st.session_state.get('_elec_editor_key_idx', 0)}"

    st.markdown("#### Non-Renewable Electricity Purchased by Country")


    col_cfg = {
        "Country": st.column_config.TextColumn("Country", disabled=True, width="medium"),
        "Unit":    st.column_config.TextColumn("Unit",    disabled=True, width="small"),
    }
    for yr in YEARS:
        col_cfg[str(yr)] = st.column_config.NumberColumn(
            str(yr), min_value=0, format="%.2f", width="small"
        )

    # Render the editor — DO NOT write its return value back to session_state
    # here; instead use the on-change callback approach via a separate Save button.
    # The data_editor return value IS the live edited state on every rerun.
    edited = st.data_editor(
        st.session_state.elec_data,
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
        height=900,
        key=editor_key,
        # num_rows="fixed" so no row add/delete accidentally resets things
        num_rows="fixed",
    )
    # Always keep session_state in sync with what the editor returns this frame
    st.session_state.elec_data = edited

    # ── Save button ───────────────────────────────────────────────────────────
    col_a, col_b = st.columns([4, 1])
    with col_b:
        if st.button("💾 Save electricity data", type="primary", key="elec_save_btn"):
            msg = _save_electricity_to_master(company, rep_year)
            if "saved" in msg.lower() or "synced" in msg.lower():
                st.success("✅ Saved successfully — added to your database.")
            else:
                st.warning(msg)

    # ── Summary metrics ───────────────────────────────────────────────────────
    rep_yr_str = str(rep_year)
    total_rep  = edited[rep_yr_str].sum() if rep_yr_str in edited.columns else 0
    total_all  = sum(edited[str(yr)].sum() for yr in YEARS if str(yr) in edited.columns)
    c1, c2 = st.columns(2)
    c1.metric(f"Total — {rep_yr_str} (all countries)", f"{total_rep:,.0f} MWh")
    c2.metric("Grand total all years", f"{total_all:,.0f} MWh")




# ─────────────────────────────────────────────────────────
# WASTE TAB
# ─────────────────────────────────────────────────────────
def render_waste_tab():
    inp, out = get_current_outputs()
    hist     = get_hist_outputs()
    rep_year = st.session_state.get("reporting_year", CURR_YEAR)
    st.markdown("#### Waste KPIs — Corporate Units")
    st.caption("Total waste must equal Recovery + Elimination. The consistency check validates this.")

    WASTE_ROWS = [
        ("section","Global Information",None,None,None),
        ("input","Total no. of sites","no.","total_sites",None),
        ("input","Production","metric T","production",None),
        ("section","Waste",None,None,None),
        ("input","Total amount of waste","metric T","waste_total",None),
        ("input","Amount of waste sent to recovery","metric T","waste_recovery",None),
        ("calc","Amount of waste sent to elimination","metric T",None,lambda i,o:f"{o.waste_elimination:,.0f}"),
        ("calc","Consistency check","—",None,lambda i,o:"OK" if o.check_waste else "Error"),
        ("calc","Recovery rate","%",None,lambda i,o:f"{o.waste_recovery_pct*100:.1f}%"),
        ("calc","Waste intensity","kg/T prod",None,lambda i,o:f"{i.waste_total/i.production*1000:.2f}" if i.production else "—"),
    ]
    data = []
    for rtype, label, unit, key, fn in WASTE_ROWS:
        if rtype == "section":
            row = {"Indicator": f"▸ {label}", "Unit": ""}
            for yr, hi, ho in hist: row[str(yr)] = ""
            row[str(rep_year)] = ""; row["YoY %"] = ""
            data.append({"_type":"section","_row":row}); continue
        row = {"Indicator": label, "Unit": unit or ""}
        hist_nums = []
        for yr, hi, ho in hist:
            v = getattr(hi, key, None) if key else None
            if v is None and fn: v = fn(hi, ho)
            try:
                row[str(yr)] = f"{int(round(float(v))):,}" if isinstance(v,(int,float)) else (str(v) if v else "—")
            except (TypeError, ValueError):
                row[str(yr)] = str(v) if v is not None else "—"
            try: hist_nums.append(float(str(v).replace(",","").replace("%","").replace("—","0")))
            except: hist_nums.append(0)
        cv = getattr(inp, key, None) if key else None
        if cv is None and fn: cv = fn(inp, out)
        row[str(rep_year)] = str(cv) if cv is not None else "—"
        try:
            cn = float(str(cv).replace(",","").replace("%",""))
            pn = hist_nums[-1] if hist_nums else 0
            row["YoY %"] = f"{(cn-pn)/abs(pn)*100:+.1f}%" if pn else "—"
        except: row["YoY %"] = "—"
        data.append({"_type":rtype,"_row":row})

    all_rows  = [d["_row"]  for d in data]
    all_types = [d["_type"] for d in data]
    df_w = pd.DataFrame(all_rows)
    curr_col = str(rep_year)

    def _style_waste(row, idx):
        rt = all_types[idx]
        return [
            "background:#F0FDF8;font-weight:700;color:#065F46" if rt == "section"
            else "background:#DBEAFE;font-weight:600" if (rt == "input" and col == curr_col)
            else "background:#F0F9FF" if rt == "input"
            else "background:#F8FAFC;font-style:italic;color:#6B7280"
            for col in df_w.columns
        ]

    st.dataframe(df_w.style.apply(lambda row: _style_waste(row, row.name), axis=1),
                 hide_index=True, use_container_width=True, height=400)
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Waste", f"{inp.waste_total:,.0f} T")
    c2.metric("Recovery Rate", f"{out.waste_recovery_pct*100:.1f}%")
    c3.metric("Consistency", "OK" if out.check_waste else "Error")


# ─────────────────────────────────────────────────────────
# QUALITATIVE TAB
# ─────────────────────────────────────────────────────────
def render_qualitative_tab():
    st.markdown("""
    <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:14px 18px;margin-bottom:16px;font-size:13px;color:#374151;line-height:1.7">
    This section gathers qualitative data to help gain additional insights for better interpretation of your
    quantitative data. Please report your company's main programs, trends, or actions that are already
    implemented, under implementation or planned.<br>
    <span style="color:#9CA3AF;font-size:12px">Non-public information will be kept confidential and only used at an aggregated level.</span>
    </div>
    """, unsafe_allow_html=True)

    def qual_section(icon, title, questions):
        st.markdown(f"""
        <div style="background:#0A2240;color:#fff;font-size:13px;font-weight:700;
            padding:9px 16px;border-radius:8px 8px 0 0;margin-top:18px;margin-bottom:0">
          {icon} {title}
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            for q_label, q_hint, q_key in questions:
                st.markdown(f"**{q_label}**")
                if q_hint: st.caption(q_hint)
                c1, c2, c3 = st.columns([2,2,1])
                with c1: st.text_area("Public information",   key=f"pub_{title}_{q_key}",   height=90, placeholder="Information for the Global KPIs Report...")
                with c2: st.text_area("Non-public (confidential)", key=f"nonpub_{title}_{q_key}", height=90, placeholder="Used only at aggregated level...")
                with c3: st.text_area("Other comments",       key=f"cmt_{title}_{q_key}",   height=90, placeholder="Any additional remarks...")
                st.divider()

    qual_section("", "Energy", [
        ("Program — Management approach", "Explain how your organization manages the energy topic: policies, commitments, ISO 50001 certifications, goals & targets.", "program"),
        ("Impacts", "Include the expected impacts related to the program initiatives. Do you expect efforts to impact the Energy KPI?", "impacts"),
        ("Specific projects completed / underway", "Report specific projects related to energy that you are currently running, implementing or planning.", "projects"),
    ])
    qual_section("", "CO2 Emissions", [
        ("Program — Management approach", "Explain how your organization manages CO2: policies, commitments, goals & targets.", "program"),
        ("Impacts", "Do you expect the efforts to positively or negatively impact the CO2 KPI?", "impacts"),
        ("Specific projects completed / underway", "Report specific projects related to CO2 reduction.", "projects"),
    ])
    qual_section("", "Water", [
        ("Program — Management approach", "Explain how your organization manages water: policies, commitments, goals & targets.", "program"),
        ("Specific projects completed / underway", "Report specific projects related to water management.", "projects"),
    ])
    qual_section("", "Waste", [
        ("Program — Management approach", "Explain how your organization manages waste: policies, commitments, goals & targets.", "program"),
        ("Specific projects completed / underway", "Report the specific projects related to waste that you are currently running.", "projects"),
    ])

    st.markdown("""<div style="background:#0A2240;color:#fff;font-size:13px;font-weight:700;
        padding:9px 16px;border-radius:8px 8px 0 0;margin-top:18px;margin-bottom:0">
      Additional Information</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Other information that may affect the five environmental KPIs**")
        st.text_area("Additional comments", key="qual_additional", height=120,
                     placeholder="e.g. major plant closures, acquisitions, production restructuring...")


# ─────────────────────────────────────────────────────────
# CONVERSION TABLES TAB
# ─────────────────────────────────────────────────────────
def render_conversion_tab():
    st.markdown("#### Unit Conversion Tables")
    st.caption("Reference factors used to normalise data to corporate units. Do not edit.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Energy conversion factors**")
        st.dataframe(pd.DataFrame({
            "Energy Type": ["Natural Gas","Propane","LPG","Diesel","Petrol","Fuel Oil","Coal","Biomass","Waste Tires"],
            "Unit":        ["GJ LHV"]*9,
            "CO2 EF (T.CO2/GJ)": [0.0561,0.0631,0.0561,0.0741,0.0693,0.0774,0.0950,0.0,0.0475],
        }), hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Unit conversion factors**")
        st.dataframe(pd.DataFrame({
            "Indicator": ["Production","Production","Water","Energy (electric)","Energy (electric)","Waste","Waste"],
            "From unit": ["kg","lb","m³","MWh","TJ","kg","lb"],
            "To unit":   ["metric T","metric T","m³","GJ","GJ","metric T","metric T"],
            "Factor":    [0.001,0.000454,1.0,3.6,1000.0,0.001,0.000454],
        }), hide_index=True, use_container_width=True)
    st.divider()
    st.markdown("**Source:** WBCSD TIP methodology · IEA country factors (Scope 2) · IPCC 2006 Guidelines")


# ─────────────────────────────────────────────────────────
# PAGE 2 -- ANALYSIS (wired to real sector data)
# ─────────────────────────────────────────────────────────
def page_analysis():
    import plotly.graph_objects as go

    data_src = "live consolidated data" if not _CONSOLIDATED_DF.empty else "built-in demo data"
    st.markdown("## Analysis & Trends")
    st.caption(f"Sector aggregated across all TIP member companies · Source: {data_src}")

    # L2 FIX: warn clearly when no real data is loaded — prevents analysts or
    # auditors from mistaking illustrative fallback numbers for real submissions.
    if _USING_FALLBACK_DATA:
        st.warning(
            "⚠️ No consolidated master CSV found — charts are showing illustrative "
            "fallback data only. Run build_esg_master.py and save at least one "
            "company submission to see real data here.",
            icon=None,
        )

    yrs     = [str(y) for y in LONG_YEARS]
    yrs_int = LONG_YEARS

    C = {
        "navy":"#0A2240","red":"#C8102E","green":"#00916E","blue":"#1D4ED8",
        "teal":"#0891B2","amber":"#D97706","purple":"#7C3AED","coral":"#EA580C",
        "gray":"#6B7280","grid":"#F3F4F6","bg":"#FFFFFF",
    }
    PALETTE_10 = ["#C8102E","#0A2240","#00916E","#1D4ED8","#D97706",
                  "#7C3AED","#0891B2","#EA580C","#059669","#DB2777"]

    def _layout(title="", height=300, legend_h=True, **kw):
        base = dict(
            title=dict(text=title, font=dict(size=13, color=C["navy"])),
            height=height, margin=dict(l=10,r=10,t=40,b=30),
            plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
            xaxis=dict(gridcolor=C["grid"], tickfont=dict(size=10)),
            yaxis=dict(gridcolor=C["grid"], tickfont=dict(size=10)),
            legend=dict(orientation="h" if legend_h else "v",
                        y=1.12 if legend_h else 1, font=dict(size=10)),
            hovermode="x unified",
        )
        base.update(kw)
        return base

    def _line(x, y, name, color, dash="solid", width=2, fill=None, fill_color=None, marker_size=4):
        kw = dict(x=x, y=y, name=name, mode="lines+markers",
                  line=dict(color=color, width=width, dash=dash),
                  marker=dict(size=marker_size, color=color),
                  hovertemplate="%{y:.2f}<extra>" + name + "</extra>")
        if fill:
            kw["fill"] = fill
            kw["fillcolor"] = fill_color or "rgba(128,128,128,.08)"
        return go.Scatter(**kw)

    df = _CONSOLIDATED_DF
    has_wide = (not df.empty and "Row_Label" not in df.columns)

    def _sector(col, divisor=1):
        if has_wide and col in df.columns:
            return (df.groupby("Year")[col].sum() / divisor).reindex(yrs_int)
        return None

    def _sector_mean(col):
        if has_wide and col in df.columns:
            return df.groupby("Year")[col].mean().reindex(yrs_int)
        return None

    def _co_series(company, col, divisor=1):
        if has_wide and col in df.columns:
            s = df[df["Company"]==company].set_index("Year")[col] / divisor
            return s.reindex(yrs_int)
        return None

    def _safe(series, fallback):
        if series is None:
            return fallback
        return [(float(v) if (v == v and v is not None) else fallback[i])
                for i, v in enumerate(series.values)]

    companies = sorted(df["Company"].unique().tolist()) if has_wide else []

    energy_total  = _safe(_sector("Total energy", 1e6),            LONG_DATA["energy"])
    co2_total     = _safe(_sector("Total CO2", 1e6),               LONG_DATA["co2"])
    scope1_total  = _safe(_sector("Total CO2 - Scope 1", 1e6),     LONG_DATA["scope1"])
    scope2_total  = _safe(_sector("Total CO2 - Scope 2", 1e6),     LONG_DATA["scope2"])
    water_total   = _safe(_sector("Water intake", 1e6),            LONG_DATA["water"])
    energy_kpi    = _safe(_sector_mean("Total energy - KPI"),      LONG_DATA["energy_kpi"])
    co2_kpi       = _safe(_sector_mean("Total CO2 - KPI"),         LONG_DATA["co2_kpi"])
    water_kpi_v   = _safe(_sector_mean("Water intake - KPI"),      [7.0]*15)
    renew_pct     = _safe(_sector_mean("Renewable_Electricity_Share_%"), LONG_DATA["renew_pct"])
    waste_recov   = _safe(_sector_mean("Waste_Recovery_Rate_%"),   LONG_DATA["waste_recov"])
    iso_cert      = _safe(_sector_mean("ISO_Certification_%"),     [93.0]*15)
    waste_total_v = _safe(_sector("Total Waste"),                  [v*330000 for v in LONG_DATA["prod"]])
    waste_recov_a = _safe(_sector("Waste Recovered"),              [v*280000 for v in LONG_DATA["prod"]])

    # ── Headline KPI strip ─────────────────────────────────────────────────────
    def _delta(cur, prv, good_if_down=True):
        if prv and prv != 0:
            pct = (cur - prv) / abs(prv) * 100
            good = (pct < 0) == good_if_down
            arrow = "▼" if pct < 0 else "▲"
            col = "#00916E" if good else "#C8102E"
            return f'<span style="color:{col};font-size:11px">{arrow} {abs(pct):.1f}%</span>'
        return '<span style="font-size:11px;color:#9CA3AF">—</span>'

    kpi_items = [
        ("Total Energy 2023",   f"{energy_total[-1]:.1f}M",  "GJ",        _delta(energy_total[-1], energy_total[-2], True)),
        ("Total CO₂ 2023",      f"{co2_total[-1]:.2f}M",    "T.CO₂",     _delta(co2_total[-1], co2_total[-2], True)),
        ("CO₂ Intensity",       f"{co2_kpi[-1]:.3f}",       "T.CO₂/T",   _delta(co2_kpi[-1], co2_kpi[-2], True)),
        ("Renewable Electricity",f"{renew_pct[-1]:.1f}%",   "of elec",   _delta(renew_pct[-1], renew_pct[-2], False)),
        ("Waste Recovery",      f"{waste_recov[-1]:.1f}%",  "of waste",  _delta(waste_recov[-1], waste_recov[-2], False)),
    ]
    kpi_cols = st.columns(5)
    for i, (label, val, unit, delta_html) in enumerate(kpi_items):
        kpi_cols[i].markdown(
            f'''<div style="border:0.5px solid #E5E7EB;border-radius:8px;
                padding:12px 14px;background:#fff">
            <div style="font-size:11px;color:#6B7280;margin-bottom:3px">{label}</div>
            <div style="font-size:21px;font-weight:600;color:#0A2240;line-height:1.1">{val}</div>
            <div style="font-size:11px;color:#9CA3AF">{unit}</div>
            <div style="margin-top:3px">{delta_html}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs — clients see General only; DSS sees all 4 ──────────────────────
    is_dss_user = st.session_state.get("is_dss", False)

    if is_dss_user:
        tab_gen, tab_p12, tab_p3, tab_p4 = st.tabs([
            "General",
            "Pathway 1 & 2  —  Energy & CO₂",
            "Pathway 3  —  Water",
            "Pathway 4  —  Waste & Environment",
        ])
    else:
        # Client employees: sector aggregates only, no competitor data
        st.info(
            "📊 Showing TIP sector aggregates. "
            "Pathway-level analysis with individual company breakdowns "
            "is available to dss+ analysts only.",
            icon=None,
        )
        tab_gen = st.tabs(["General"])[0]
        tab_p12 = None
        tab_p3  = None
        tab_p4  = None

    # ── TAB 1: GENERAL ──────────────────────────────────────────────────────────
    with tab_gen:
        st.caption("Core environmental KPIs for CSR / sustainability reporting — sector totals 2009–2023")

        c1, c2 = st.columns(2)
        with c1:
            f = go.Figure([_line(yrs, energy_total, "Total energy (M GJ)", C["green"],
                fill="tozeroy", fill_color="rgba(0,145,110,.08)")])
            f.update_layout(**_layout("Total energy consumption (M GJ)", 260))
            st.plotly_chart(f, use_container_width=True)
        with c2:
            f = go.Figure([
                _line(yrs, scope1_total, "Scope 1", C["red"],
                    fill="tozeroy", fill_color="rgba(200,16,46,.10)"),
                _line(yrs, scope2_total, "Scope 2", C["blue"],
                    fill="tozeroy", fill_color="rgba(29,78,216,.10)"),
            ])
            f.update_layout(**_layout("CO₂ emissions — Scope 1 vs Scope 2 (M T.CO₂)", 260))
            st.plotly_chart(f, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            f = go.Figure()
            fuel_colors = {"Natural Gas":"#F59E0B","Electricity":"#3B82F6",
                           "Fuel Oil":"#EF4444","LPG":"#8B5CF6","Coal":"#6B7280","Other":"#D1D5DB"}
            for fuel, vals in FUEL_MIX.items():
                f.add_trace(go.Bar(x=yrs, y=vals, name=fuel,
                    marker_color=fuel_colors.get(fuel,"#ccc"), marker_line_width=0,
                    hovertemplate=f"{fuel}: %{{y:.1f}}%<extra></extra>"))
            f.update_layout(**_layout("Fuel mix evolution (%)", 260,
                barmode="stack",
                yaxis=dict(range=[0,100],ticksuffix="%",gridcolor=C["grid"])))
            st.plotly_chart(f, use_container_width=True)
        with c4:
            f = go.Figure([_line(yrs, water_total, "Water withdrawals (M m³)", C["teal"],
                fill="tozeroy", fill_color="rgba(8,145,178,.08)")])
            f.update_layout(**_layout("Water withdrawals (M m³)", 260))
            st.plotly_chart(f, use_container_width=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            f = go.Figure(go.Bar(x=yrs, y=renew_pct,
                marker_color=["rgba(0,145,110,.9)" if i>=12 else "rgba(0,145,110,.4)" for i in range(len(yrs))],
                hovertemplate="%{y:.1f}%<extra>Renewable</extra>"))
            f.update_layout(**_layout("Renewable electricity share (%)", 230,
                yaxis=dict(range=[0,100],ticksuffix="%",gridcolor=C["grid"])))
            st.plotly_chart(f, use_container_width=True)
        with c6:
            f = go.Figure()
            f.add_trace(go.Bar(x=yrs, y=waste_recov, name="Recovery %",
                marker_color="rgba(0,145,110,.75)", marker_line_width=0))
            f.add_trace(go.Bar(x=yrs, y=[100-v for v in waste_recov],
                name="Elimination %", marker_color="rgba(200,16,46,.35)", marker_line_width=0))
            f.update_layout(**_layout("Waste recovery rate (%)", 230,
                barmode="stack", yaxis=dict(range=[0,100],ticksuffix="%",gridcolor=C["grid"])))
            st.plotly_chart(f, use_container_width=True)
        with c7:
            f = go.Figure(go.Bar(x=yrs, y=iso_cert,
                marker_color="rgba(10,34,64,.7)", marker_line_width=0,
                hovertemplate="%{y:.1f}%<extra>ISO 14001</extra>"))
            f.update_layout(**_layout("ISO 14001 certification rate (%)", 230,
                yaxis=dict(range=[0,100],ticksuffix="%",gridcolor=C["grid"])))
            st.plotly_chart(f, use_container_width=True)

        # ── Client-visible: own company vs sector average ─────────────────────
        if not is_dss_user:
            client_co = st.session_state.get("user_company", "")
            if has_wide and client_co and client_co in df["Company"].values:
                st.markdown("---")
                st.markdown(f"##### {client_co} — your performance vs TIP sector average")
                st.caption("Showing your company data alongside anonymous sector benchmarks. "
                           "No individual competitor data is disclosed.")

                _c1, _c2 = st.columns(2)
                with _c1:
                    _s_co  = _co_series(client_co, "Total energy - KPI")
                    if _s_co is not None:
                        _co_vals = [float(v) if not np.isnan(float(v)) else None for v in _s_co.values]
                        _f = go.Figure()
                        _f.add_trace(go.Scatter(x=yrs, y=energy_kpi, name="TIP sector avg",
                            mode="lines", line=dict(color="#9CA3AF", width=2, dash="dot"),
                            hovertemplate="Sector avg: %{y:.2f}<extra></extra>"))
                        _f.add_trace(go.Scatter(x=yrs, y=_co_vals,
                            name=client_co.split()[0], mode="lines+markers",
                            line=dict(color=C["navy"], width=2),
                            marker=dict(size=4),
                            hovertemplate=f"{client_co.split()[0]}: %{{y:.2f}} GJ/T<extra></extra>"))
                        _f.update_layout(**_layout("Energy intensity — your company vs sector (GJ/T)", 240))
                        st.plotly_chart(_f, use_container_width=True)

                with _c2:
                    _s_co2 = _co_series(client_co, "Total CO2 - KPI")
                    if _s_co2 is not None:
                        _co2_vals = [float(v) if not np.isnan(float(v)) else None for v in _s_co2.values]
                        _f2 = go.Figure()
                        _f2.add_trace(go.Scatter(x=yrs, y=co2_kpi, name="TIP sector avg",
                            mode="lines", line=dict(color="#9CA3AF", width=2, dash="dot"),
                            hovertemplate="Sector avg: %{y:.3f}<extra></extra>"))
                        _f2.add_trace(go.Scatter(x=yrs, y=_co2_vals,
                            name=client_co.split()[0], mode="lines+markers",
                            line=dict(color=C["red"], width=2),
                            marker=dict(size=4),
                            hovertemplate=f"{client_co.split()[0]}: %{{y:.3f}} T.CO₂/T<extra></extra>"))
                        _f2.update_layout(**_layout("CO₂ intensity — your company vs sector (T.CO₂/T)", 240))
                        st.plotly_chart(_f2, use_container_width=True)

                _c3, _c4 = st.columns(2)
                with _c3:
                    _sw = _co_series(client_co, "Water intake - KPI")
                    if _sw is not None:
                        _w_vals = [float(v) if not np.isnan(float(v)) else None for v in _sw.values]
                        _fw = go.Figure()
                        _fw.add_trace(go.Scatter(x=yrs, y=water_kpi_v, name="TIP sector avg",
                            mode="lines", line=dict(color="#9CA3AF", width=2, dash="dot")))
                        _fw.add_trace(go.Scatter(x=yrs, y=_w_vals,
                            name=client_co.split()[0], mode="lines+markers",
                            line=dict(color=C["teal"], width=2), marker=dict(size=4)))
                        _fw.update_layout(**_layout("Water intensity — your company vs sector (m³/T)", 240))
                        st.plotly_chart(_fw, use_container_width=True)

                with _c4:
                    _sr = _co_series(client_co, "Waste_Recovery_Rate_%")
                    if _sr is None: _sr = _co_series(client_co, "Recovery Rate")
                    if _sr is not None:
                        _r_vals = [float(v) if not np.isnan(float(v)) else None for v in _sr.values]
                        _fr = go.Figure()
                        _fr.add_trace(go.Scatter(x=yrs, y=waste_recov, name="TIP sector avg",
                            mode="lines", line=dict(color="#9CA3AF", width=2, dash="dot")))
                        _fr.add_trace(go.Scatter(x=yrs, y=_r_vals,
                            name=client_co.split()[0], mode="lines+markers",
                            line=dict(color=C["green"], width=2), marker=dict(size=4)))
                        _fr.update_layout(**_layout("Waste recovery rate — your company vs sector (%)", 240,
                            yaxis=dict(ticksuffix="%", gridcolor=C["grid"])))
                        st.plotly_chart(_fr, use_container_width=True)

    # ── TAB 2: PATHWAY 1 & 2 — DSS EMPLOYEES ONLY ────────────────────────────
    if is_dss_user and tab_p12 is not None:
     with tab_p12:
        st.markdown("##### Pathway 1 — Energy consumption & intensity")
        st.caption("KPI 1: Total energy (GJ) · Energy intensity (GJ / metric T of production)")

        c1, c2 = st.columns(2)
        with c1:
            f = go.Figure([_line(yrs, energy_total, "Total energy (M GJ)", C["green"],
                fill="tozeroy", fill_color="rgba(0,145,110,.08)")])
            f.update_layout(**_layout("Sector total energy (M GJ)", 270))
            st.plotly_chart(f, use_container_width=True)
        with c2:
            f = go.Figure([_line(yrs, energy_kpi, "Energy intensity (GJ/T)", C["purple"])])
            f.add_hrect(y0=8.0, y1=9.5, fillcolor="rgba(0,145,110,.07)", line_width=0,
                annotation_text="Target range", annotation_font_size=10)
            f.update_layout(**_layout("Energy intensity — GJ per metric ton", 270))
            st.plotly_chart(f, use_container_width=True)

        if has_wide and companies:
            rows = []
            for co in companies:
                s = _co_series(co, "Total energy - KPI")
                if s is not None:
                    v09 = float(s.iloc[0])  if not np.isnan(float(s.iloc[0]))  else None
                    v23 = float(s.iloc[-1]) if not np.isnan(float(s.iloc[-1])) else None
                    if v09 and v23:
                        rows.append({"Company": co.split()[0], "2009": v09, "2023": v23})
            if rows:
                f = go.Figure()
                names = [r["Company"] for r in rows]
                f.add_trace(go.Bar(name="2009", x=names, y=[r["2009"] for r in rows],
                    marker_color="rgba(10,34,64,.4)", marker_line_width=0))
                f.add_trace(go.Bar(name="2023", x=names, y=[r["2023"] for r in rows],
                    marker_color=C["navy"], marker_line_width=0))
                f.update_layout(**_layout("Energy intensity (GJ/T) — 2009 vs 2023 by company",
                    260, barmode="group",
                    yaxis=dict(title="GJ/metric T", gridcolor=C["grid"])))
                st.plotly_chart(f, use_container_width=True)

        st.divider()
        st.markdown("##### Pathway 2 — CO₂ emissions & intensity")
        st.caption("KPI 2: Total CO₂ Scope 1+2 (T.CO₂) · CO₂ intensity (T.CO₂ / metric T)")

        c3, c4 = st.columns(2)
        with c3:
            f = go.Figure()
            f.add_trace(go.Scatter(x=yrs, y=scope1_total, name="Scope 1",
                fill="tozeroy", fillcolor="rgba(200,16,46,.15)",
                line=dict(color=C["red"], width=2)))
            combined = [s1+s2 for s1,s2 in zip(scope1_total, scope2_total)]
            f.add_trace(go.Scatter(x=yrs, y=combined, name="Scope 1+2",
                fill="tonexty", fillcolor="rgba(29,78,216,.12)",
                line=dict(color=C["blue"], width=2)))
            f.update_layout(**_layout("CO₂ — Scope 1 & 2 stacked (M T.CO₂)", 270))
            st.plotly_chart(f, use_container_width=True)
        with c4:
            f = go.Figure([_line(yrs, co2_kpi, "CO₂ intensity (T.CO₂/T)", C["red"])])
            f.add_hrect(y0=0.55, y1=0.70, fillcolor="rgba(0,145,110,.07)", line_width=0,
                annotation_text="Target range", annotation_font_size=10)
            f.update_layout(**_layout("CO₂ intensity — T.CO₂ per metric ton", 270))
            st.plotly_chart(f, use_container_width=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            s1l, s2l = scope1_total[-1], scope2_total[-1]
            f = go.Figure(go.Pie(
                labels=["Scope 1","Scope 2"], values=[s1l, s2l], hole=0.55,
                marker_colors=[C["red"], C["blue"]], textfont_size=12,
                hovertemplate="%{label}: %{value:.3f}M<extra></extra>"))
            f.update_layout(**_layout("Scope 1 vs Scope 2 — 2023", 240, legend_h=False))
            st.plotly_chart(f, use_container_width=True)
        with c6:
            if companies:
                co_names, co_vals = [], []
                for co in companies:
                    s = _co_series(co, "Total CO2 - KPI")
                    if s is not None:
                        v = float(s.iloc[-1])
                        if not np.isnan(v):
                            co_names.append(co.split()[0]); co_vals.append(v)
                if co_names:
                    pairs = sorted(zip(co_vals, co_names))
                    sv, sn = zip(*pairs)
                    f = go.Figure(go.Bar(x=list(sv), y=list(sn), orientation="h",
                        marker_color=[C["green"] if v < 0.70 else C["red"] for v in sv],
                        marker_line_width=0,
                        hovertemplate="%{x:.3f} T.CO₂/T<extra>%{y}</extra>"))
                    f.update_layout(**_layout("CO₂ intensity by company — 2023", 240,
                        legend_h=False,
                        xaxis=dict(title="T.CO₂/metric T", gridcolor=C["grid"])))
                    st.plotly_chart(f, use_container_width=True)
        with c7:
            if companies:
                import pandas as _pd
                yoy_rows = []
                for co in companies:
                    s = _co_series(co, "Total CO2 - KPI")
                    if s is not None:
                        v09 = float(s.iloc[0]);  v23 = float(s.iloc[-1])
                        if not (np.isnan(v09) or np.isnan(v23)):
                            yoy_rows.append({"Company": co.split()[0],
                                "2009": f"{v09:.3f}", "2023": f"{v23:.3f}",
                                "Change": f"{(v23-v09)/v09*100:+.1f}%"})
                if yoy_rows:
                    st.markdown("**CO₂ intensity 2009 → 2023**")
                    st.dataframe(_pd.DataFrame(yoy_rows).set_index("Company"),
                        use_container_width=True, height=220)

    # ── TAB 3: PATHWAY 3 — DSS EMPLOYEES ONLY ────────────────────────────────
    if is_dss_user and tab_p3 is not None:
     with tab_p3:
        st.markdown("##### Pathway 3 — Water withdrawals & intensity")
        st.caption("KPI 3: Total water intake (m³) · Water intensity (m³ / metric T of production)")

        c1, c2 = st.columns(2)
        with c1:
            f = go.Figure([_line(yrs, water_total, "Water withdrawals (M m³)", C["teal"],
                fill="tozeroy", fill_color="rgba(8,145,178,.08)")])
            f.update_layout(**_layout("Sector water withdrawals (M m³)", 280))
            st.plotly_chart(f, use_container_width=True)
        with c2:
            f = go.Figure([_line(yrs, water_kpi_v, "Water intensity (m³/T)", C["teal"])])
            f.add_hrect(y0=5.5, y1=7.5, fillcolor="rgba(0,145,110,.07)", line_width=0,
                annotation_text="Target range", annotation_font_size=10)
            f.update_layout(**_layout("Water intensity — m³ per metric ton", 280))
            st.plotly_chart(f, use_container_width=True)

        if has_wide and companies:
            f = go.Figure()
            for i, co in enumerate(companies):
                s = _co_series(co, "Water intake - KPI")
                if s is not None:
                    vals = [float(v) if not np.isnan(float(v)) else None for v in s.values]
                    f.add_trace(go.Scatter(x=yrs, y=vals, name=co.split()[0],
                        mode="lines+markers",
                        line=dict(color=PALETTE_10[i % 10], width=1.5),
                        marker=dict(size=3),
                        hovertemplate=f"{co.split()[0]}: %{{y:.2f}} m³/T<extra></extra>"))
            f.add_trace(go.Scatter(x=yrs, y=water_kpi_v, name="Sector avg",
                mode="lines", line=dict(color="#000", width=2, dash="dot")))
            f.update_layout(**_layout(
                "Water intensity (m³/T) — per company vs sector average", 310, legend_h=False))
            st.plotly_chart(f, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                co_wkpi = {}
                for co in companies:
                    s = _co_series(co, "Water intake - KPI")
                    if s is not None:
                        v = float(s.iloc[-1])
                        if not np.isnan(v): co_wkpi[co.split()[0]] = v
                if co_wkpi:
                    pairs = sorted(co_wkpi.items(), key=lambda x: x[1])
                    names, vals = zip(*pairs)
                    f = go.Figure(go.Bar(x=list(vals), y=list(names), orientation="h",
                        marker_color=[C["green"] if v<7.0 else C["amber"] for v in vals],
                        marker_line_width=0,
                        hovertemplate="%{x:.2f} m³/T<extra>%{y}</extra>"))
                    f.update_layout(**_layout("Water intensity ranking — 2023 (m³/T)",
                        260, legend_h=False))
                    st.plotly_chart(f, use_container_width=True)
            with c4:
                if "Water intake" in df.columns and "Production" in df.columns:
                    yr_df = df[df["Year"]==2023].dropna(subset=["Water intake","Production"])
                    if not yr_df.empty:
                        f = go.Figure()
                        for idx, (_, row) in enumerate(yr_df.iterrows()):
                            f.add_trace(go.Scatter(
                                x=[row["Production"]/1e6], y=[row["Water intake"]/1e6],
                                name=str(row["Company"]).split()[0],
                                mode="markers+text",
                                text=[str(row["Company"]).split()[0]],
                                textposition="top center",
                                marker=dict(size=12, color=PALETTE_10[idx % 10])))
                        f.update_layout(**_layout("Water vs production — 2023", 260,
                            legend_h=False,
                            xaxis=dict(title="Production (M T)", gridcolor=C["grid"]),
                            yaxis=dict(title="Water (M m³)", gridcolor=C["grid"])))
                        st.plotly_chart(f, use_container_width=True)

    # ── TAB 4: PATHWAY 4 — DSS EMPLOYEES ONLY ────────────────────────────────
    if is_dss_user and tab_p4 is not None:
     with tab_p4:
        st.markdown("##### Pathway 4 — Waste management & environmental certification")
        st.caption("KPI 4a: Total waste & recovery rate · KPI 4b: ISO 14001 site certification")

        c1, c2 = st.columns(2)
        with c1:
            f = go.Figure()
            f.add_trace(go.Bar(x=yrs, y=waste_recov_a, name="Recovered (T)",
                marker_color="rgba(0,145,110,.75)", marker_line_width=0,
                hovertemplate="Recovered: %{y:,.0f} T<extra></extra>"))
            f.add_trace(go.Bar(x=yrs,
                y=[t-r for t,r in zip(waste_total_v, waste_recov_a)],
                name="Eliminated (T)", marker_color="rgba(200,16,46,.4)", marker_line_width=0,
                hovertemplate="Eliminated: %{y:,.0f} T<extra></extra>"))
            f.update_layout(**_layout("Waste — recovery vs elimination (metric T)", 280,
                barmode="stack", yaxis=dict(gridcolor=C["grid"])))
            st.plotly_chart(f, use_container_width=True)
        with c2:
            f = go.Figure(go.Scatter(x=yrs, y=waste_recov, name="Recovery rate",
                fill="tozeroy", fillcolor="rgba(0,145,110,.10)",
                line=dict(color=C["green"], width=2),
                hovertemplate="Recovery: %{y:.1f}%<extra></extra>"))
            f.update_layout(**_layout("Waste recovery rate — sector average (%)", 280,
                yaxis=dict(range=[50,100], ticksuffix="%", gridcolor=C["grid"])))
            st.plotly_chart(f, use_container_width=True)

        if has_wide and companies:
            c3, c4 = st.columns(2)
            with c3:
                co_wr = {}
                for co in companies:
                    s = _co_series(co, "Waste_Recovery_Rate_%")
                    if s is None: s = _co_series(co, "Recovery Rate")
                    if s is not None:
                        v = float(s.iloc[-1])
                        if not np.isnan(v): co_wr[co.split()[0]] = v
                if co_wr:
                    pairs = sorted(co_wr.items(), key=lambda x: x[1], reverse=True)
                    names, vals = zip(*pairs)
                    f = go.Figure(go.Bar(x=list(names), y=list(vals),
                        marker_color=[C["green"] if v>=80 else C["amber"] for v in vals],
                        marker_line_width=0,
                        hovertemplate="%{y:.1f}%<extra>%{x}</extra>"))
                    f.add_hline(y=80, line_dash="dot", line_color=C["navy"],
                        annotation_text="80% target", annotation_font_size=10)
                    f.update_layout(**_layout("Waste recovery by company — 2023 (%)", 260,
                        yaxis=dict(range=[0,100],ticksuffix="%",gridcolor=C["grid"])))
                    st.plotly_chart(f, use_container_width=True)
            with c4:
                f = go.Figure(go.Bar(x=yrs, y=iso_cert,
                    marker_color="rgba(10,34,64,.75)", marker_line_width=0,
                    hovertemplate="%{y:.1f}%<extra>ISO 14001</extra>"))
                f.add_hline(y=100, line_dash="dot", line_color=C["green"],
                    annotation_text="100% target", annotation_font_size=10)
                f.update_layout(**_layout("ISO 14001 certification rate — sector (%)", 260,
                    yaxis=dict(range=[0,100],ticksuffix="%",gridcolor=C["grid"])))
                st.plotly_chart(f, use_container_width=True)

            import pandas as _pd2
            iso_rows = {}
            for co in companies:
                s = _co_series(co, "ISO_Certification_%")
                if s is not None:
                    iso_rows[co.split()[0]] = [
                        round(float(v),1) if not np.isnan(float(v)) else 0.0
                        for v in s.values]
            if iso_rows:
                heat_df = _pd2.DataFrame(iso_rows, index=yrs).T
                f = go.Figure(go.Heatmap(
                    z=heat_df.values.tolist(),
                    x=heat_df.columns.tolist(),
                    y=heat_df.index.tolist(),
                    colorscale=[[0,"#FEF2F2"],[0.5,"#FDE68A"],[1,"#ECFDF5"]],
                    text=[[f"{v:.0f}%" for v in row] for row in heat_df.values],
                    texttemplate="%{text}", textfont=dict(size=9),
                    zmin=0, zmax=100,
                    hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
                    colorbar=dict(title="% certified", ticksuffix="%")))
                f.update_layout(**_layout(
                    "ISO 14001 certification (% of sites) per company — all years",
                    220 + len(iso_rows)*18, legend_h=False,
                    margin=dict(l=100,r=30,t=40,b=30),
                    xaxis=dict(tickangle=-45, gridcolor=C["grid"]),
                    yaxis=dict(gridcolor=C["grid"])))
                st.plotly_chart(f, use_container_width=True)


# ─────────────────────────────────────────────────────────
# PAGE 3 -- BENCHMARKING
# ─────────────────────────────────────────────────────────
def _compute_industry_scores(df, year):
    """Compute median scores across all companies for a given year (for radar)."""
    if df.empty or "Row_Label" in df.columns:
        return [65, 70, 65, 74, 52, 74]
    yr_df = df[df["Year"] == year]
    if yr_df.empty:
        return [65, 70, 65, 74, 52, 74]
    scores = []
    kpi_map = [
        ("Total CO2 - KPI",   True,  0.55, 0.82),
        ("Total energy - KPI",True,  8.0,  10.5),
        ("Water intake - KPI",True,  5.5,  9.0),
        ("Total CO2 - KPI",   True,  0.55, 0.82),   # proxy for waste (reuse co2)
        ("Renewable_Electricity_Share_%", False, 0.0, 100.0),
    ]
    for col, lower_better, best, worst in kpi_map:
        if col in yr_df.columns:
            med = float(yr_df[col].median())
            span = abs(worst - best) if worst != best else 1
            if lower_better:
                s = max(0, min(100, (worst - med) / span * 100))
            else:
                s = max(0, min(100, (med - best) / span * 100))
            scores.append(round(s, 1))
        else:
            scores.append(65)
    scores += [74]   # H&S -- not in KPI set
    return scores


def _load_company_year_outputs(company: str, year: int):
    """
    Load inputs and compute outputs for any company+year from the consolidated DB.
    Returns (TemplateInputs, TemplateOutputs) or falls back to session state.
    """
    hist = dl.get_company_hist(_CONSOLIDATED_DF, company)
    if hist:
        sd = dl.get_step_data(hist, year)
        sd_clean = {k: v for k, v in sd.items() if k in _VALID_TEMPLATE_FIELDS}
        if sd_clean:
            inp = TemplateInputs(company=company, year=year, **sd_clean)
            return inp, calculate(inp)
    # fallback to session state
    return get_current_outputs()


def _compute_kpi_improvement(company: str, base_year: int, end_year: int) -> dict:
    """
    Compute improvement % for each KPI between base_year and end_year.
    Returns {kpi_attr: pct_change_string} for CO2, energy, water KPIs + raw fields.
    """
    base_inp, base_out = _load_company_year_outputs(company, base_year)
    end_inp,  end_out  = _load_company_year_outputs(company, end_year)

    def _pct(b, e):
        if b and b != 0 and e:
            return f"{(e - b) / abs(b) * 100:+.1f}%"
        return "N/A"

    renew_base = (base_inp.renew_elec_purchased + base_inp.self_gen_elec) / max(base_out.total_electricity, 1) * 100
    renew_end  = (end_inp.renew_elec_purchased  + end_inp.self_gen_elec)  / max(end_out.total_electricity,  1) * 100
    wrec_base  = base_out.waste_recovery_pct * 100
    wrec_end   = end_out.waste_recovery_pct  * 100

    return {
        "CO₂ intensity":        _pct(base_out.co2_kpi,    end_out.co2_kpi),
        "Energy intensity":     _pct(base_out.energy_kpi, end_out.energy_kpi),
        "Water intensity":      _pct(base_out.water_kpi,  end_out.water_kpi),
        "Renewable electricity":_pct(renew_base,           renew_end),
        "Waste recovery rate":  _pct(wrec_base,            wrec_end),
    }



def page_benchmarking():
    st.markdown("## Peer Benchmarking")

    # ── Company + year selector ──────────────────────────────────────────────
    companies_in_db = dl.get_companies(_CONSOLIDATED_DF) or COMPANIES
    default_co = (st.session_state.get("reporting_company") or
                  st.session_state.get("user_company") or companies_in_db[0])
    if default_co not in companies_in_db:
        default_co = companies_in_db[0]

    sel_col, yr_col, _ = st.columns([2, 1, 3])
    with sel_col:
        company = st.selectbox("Company", options=companies_in_db,
                               index=companies_in_db.index(default_co),
                               key="bench_company_sel")
    with yr_col:
        avail_yrs = dl.get_years(_CONSOLIDATED_DF, company) or [CURR_YEAR]
        rep_year  = st.selectbox("Year", options=sorted(avail_yrs, reverse=True),
                                 key="bench_year_sel")

    # ── Load real data for selected company+year from consolidated DB ─────────
    inp, out = _load_company_year_outputs(company, rep_year)
    renew_val = (inp.renew_elec_purchased + inp.self_gen_elec) / max(out.total_electricity, 1) * 100

    bench_source = "live consolidated dataset" if not _CONSOLIDATED_DF.empty else "built-in demo data"
    st.caption(f"Benchmarking uses the {bench_source} · {len(companies_in_db)} companies · quartiles from all TIP members.")
    st.divider()

    # ── Live quartiles from ALL companies for the reporting year ──────────────
    bench_kpis = dl.get_benchmark_kpis(_CONSOLIDATED_DF, rep_year)

    def live_bench(kpi_col, company_value, unit, lower_better):
        vals = bench_kpis[kpi_col].dropna().values if (not bench_kpis.empty and kpi_col in bench_kpis.columns) else []
        if len(vals) >= 3:
            q25 = float(np.percentile(vals, 25))
            med = float(np.percentile(vals, 50))
            q75 = float(np.percentile(vals, 75))
        else:
            q25 = company_value * 0.85; med = company_value; q75 = company_value * 1.15
        return BenchmarkResult(kpi_col, company_value, q25, med, q75, unit, lower_better)

    benchmarks = [
        live_bench("Total CO2 - KPI",             out.co2_kpi,                "T.CO2/T", True),
        live_bench("Total energy - KPI",           out.energy_kpi,             "GJ/T",    True),
        live_bench("Water intake - KPI",           out.water_kpi,              "m³/T",    True),
        live_bench("Renewable_Electricity_Share_%",renew_val,                  "%",       False),
        live_bench("waste_recovery_pct",           out.waste_recovery_pct*100, "%",       False),
    ]
    kpi_labels = ["CO₂ intensity","Energy intensity","Water intensity","Renewable electricity","Waste recovery rate"]
    for b, lbl in zip(benchmarks, kpi_labels): b.kpi_name = lbl

    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            st.markdown("#### Industry band positioning")
            st.caption(f"{company} {rep_year} KPI vs all TIP member quartile ranges.")
            st.markdown("""
            <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap">
              <span style="font-size:11px;display:flex;align-items:center;gap:4px;color:#6B7280">
                <span style="width:12px;height:10px;background:#D1FAE5;border-radius:2px;display:inline-block"></span>Top 25%</span>
              <span style="font-size:11px;display:flex;align-items:center;gap:4px;color:#6B7280">
                <span style="width:12px;height:10px;background:#FEF3C7;border-radius:2px;display:inline-block"></span>Mid 50%</span>
              <span style="font-size:11px;display:flex;align-items:center;gap:4px;color:#6B7280">
                <span style="width:12px;height:10px;background:#FEE2E2;border-radius:2px;display:inline-block"></span>Bottom 25%</span>
            </div>""", unsafe_allow_html=True)
            bands_html = '<div class="band-container">'
            for b in benchmarks:
                bands_html += band_html(b.kpi_name, b.company_value, b.q25, b.median, b.q75, b.unit, b.lower_is_better)
            bands_html += "</div>"
            st.markdown(bands_html, unsafe_allow_html=True)

    with col_r:
        with st.container(border=True):
            st.markdown("#### ESG profile — vs TIP industry median")
            st.caption(f"Normalised 0–100 from actual quartile positions. {company} {rep_year} vs sector median.")
            dims = ["CO₂ intensity","Energy efficiency","Water management","Waste recovery","Renewable energy","H&S performance"]

            company_scores = []
            for b in benchmarks[:5]:
                rng = max(b.q75 - b.q25, 0.001)
                raw = (b.company_value - b.q25) / rng
                company_scores.append(max(0, min(100, (1 - raw) * 100 if b.lower_is_better else raw * 100)))
            company_scores += [75]   # H&S: not in KPI set, shown as neutral

            industry_scores = _compute_industry_scores(_CONSOLIDATED_DF, rep_year)

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=company_scores + [company_scores[0]], theta=dims + [dims[0]],
                fill="toself", name=f"{company} {rep_year}",
                line=dict(color="#00916E", width=2), fillcolor="rgba(0,145,110,.15)"))
            fig.add_trace(go.Scatterpolar(
                r=industry_scores + [industry_scores[0]], theta=dims + [dims[0]],
                fill="toself", name=f"TIP median ({rep_year})",
                line=dict(color="#9CA3AF", width=1.5, dash="dot"), fillcolor="rgba(156,163,175,.08)"))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=9))),
                showlegend=True, height=340, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Improvement table: computed from real data for each year ──────────────
    with st.container(border=True):
        base_yr = min(dl.get_years(_CONSOLIDATED_DF, company) or [2009])
        st.markdown(f"#### Improvement rate — {company} vs TIP sector average ({base_yr}→{rep_year})")

        impr = _compute_kpi_improvement(company, base_yr, rep_year)

        def _sector_impr(col_name):
            if _CONSOLIDATED_DF.empty or "Row_Label" in _CONSOLIDATED_DF.columns:
                return "N/A"
            try:
                df_col = _CONSOLIDATED_DF[[c for c in [col_name] if c in _CONSOLIDATED_DF.columns]]
                if df_col.empty: return "N/A"
                base = _CONSOLIDATED_DF[_CONSOLIDATED_DF["Year"] == base_yr][col_name].mean()
                end  = _CONSOLIDATED_DF[_CONSOLIDATED_DF["Year"] == rep_year][col_name].mean()
                if pd.notna(base) and pd.notna(end) and base != 0:
                    return f"{(end - base) / abs(base) * 100:+.1f}%"
            except Exception:
                pass
            return "N/A"

        impr_df = pd.DataFrame({
            "KPI":                          list(impr.keys()),
            f"Your improvement ({base_yr}→{rep_year})": list(impr.values()),
            "TIP sector average": [
                _sector_impr("Total CO2 - KPI"),
                _sector_impr("Total energy - KPI"),
                _sector_impr("Water intake - KPI"),
                _sector_impr("Renewable_Electricity_Share_%"),
                "N/A",
            ],
        })

        def _style_impr(val):
            if val == "N/A": return "color:#9CA3AF"
            try:
                num = float(str(val).replace("+","").replace("%",""))
                return "color:#059669;font-weight:600" if num < 0 else (
                       "color:#DC2626;font-weight:600" if num > 10 else "color:#D97706")
            except: return ""

        styled = impr_df.style.map(_style_impr, subset=[f"Your improvement ({base_yr}→{rep_year})", "TIP sector average"])
        st.dataframe(styled, hide_index=True, use_container_width=True)
        st.caption("For intensity KPIs (CO₂, Energy, Water): negative = improvement. "
                   "For Renewable electricity: positive = improvement.")

    # ── KPI trend charts: company vs sector for ALL 5 KPIs ───────────────────
    if not _CONSOLIDATED_DF.empty and "Row_Label" not in _CONSOLIDATED_DF.columns:
        trend_kpis = [
            ("Total CO2 - KPI",             "CO₂ intensity (T.CO₂/T)", "#EF4444"),
            ("Total energy - KPI",          "Energy intensity (GJ/T)", "#F59E0B"),
            ("Water intake - KPI",          "Water intensity (m³/T)",  "#3B82F6"),
            ("Renewable_Electricity_Share_%","Renewable electricity (%)", "#00916E"),
        ]
        available = [(col, lbl, clr) for col, lbl, clr in trend_kpis if col in _CONSOLIDATED_DF.columns]

        if available:
            yrs_all = sorted(_CONSOLIDATED_DF["Year"].dropna().unique().astype(int).tolist())
            comp_df = _CONSOLIDATED_DF[_CONSOLIDATED_DF["Company"] == company]

            n_charts = len(available)
            chart_cols = st.columns(2)
            for idx, (col, lbl, clr) in enumerate(available):
                with chart_cols[idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{lbl} — {company} vs peers**")
                        s_mean = _CONSOLIDATED_DF.groupby("Year")[col].mean().reindex(yrs_all)
                        s_min  = _CONSOLIDATED_DF.groupby("Year")[col].min().reindex(yrs_all)
                        s_max  = _CONSOLIDATED_DF.groupby("Year")[col].max().reindex(yrs_all)
                        c_vals = comp_df.set_index("Year")[col].reindex(yrs_all) if not comp_df.empty else None

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=yrs_all, y=s_max.tolist(), fill=None, mode="lines",
                            line=dict(width=0, color="rgba(200,200,200,0)"), showlegend=False))
                        fig.add_trace(go.Scatter(x=yrs_all, y=s_min.tolist(), fill="tonexty", mode="lines",
                            line=dict(width=0), fillcolor="rgba(156,163,175,0.15)", name="Sector range"))
                        fig.add_trace(go.Scatter(x=yrs_all, y=s_mean.tolist(), mode="lines",
                            name="Sector avg", line=dict(color="#9CA3AF", width=1.5, dash="dot")))
                        if c_vals is not None and c_vals.notna().any():
                            fig.add_trace(go.Scatter(x=yrs_all, y=c_vals.tolist(), mode="lines+markers",
                                name=company, line=dict(color=clr, width=2.5), marker=dict(size=5)))
                        fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10),
                            plot_bgcolor="#fff", paper_bgcolor="#fff",
                            xaxis=dict(gridcolor="#F3F4F6"),
                            yaxis=dict(gridcolor="#F3F4F6"),
                            legend=dict(font=dict(size=10), orientation="h", y=-0.25))
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Trend charts require the wide master CSV. Run build_esg_master.py first.")

    # ── Strengths / gaps ──────────────────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        with st.container(border=True):
            st.markdown("#### Key strengths")
            strengths = [b for b in benchmarks if
                (b.lower_is_better and b.company_value <= b.q25) or
                (not b.lower_is_better and b.company_value >= b.q75)]
            if strengths:
                for b in strengths:
                    st.success(f"**{b.kpi_name}** — top quartile  {b.company_value:.3f} {b.unit} ≤ Q25 {b.q25:.3f}")
            else:
                st.info("No top-quartile metrics for the selected company and year.")
    with col_r2:
        with st.container(border=True):
            st.markdown("#### Improvement areas")
            gaps = [b for b in benchmarks if
                (b.lower_is_better and b.company_value > b.median) or
                (not b.lower_is_better and b.company_value < b.median)]
            if gaps:
                for b in gaps:
                    st.warning(f"**{b.kpi_name}** — {b.company_value:.3f} vs median {b.median:.3f} {b.unit}")
            else:
                st.success("All KPIs at or above sector median.")


# ─────────────────────────────────────────────────────────
# PAGE 4 -- VERIFICATION (dss+ only)
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# SHARED: dss+ company/year selector
# ─────────────────────────────────────────────────────────
def _dss_company_selector(page_key: str):
    """
    Lets a dss+ analyst pick any company + year from the consolidated DB.
    Stores selection in session_state under dss_{page_key}_company / _year.
    Returns (company, year, inp, out, prev_inp, prev_out, company_hist).
    """
    co_key = f"dss_{page_key}_company"
    yr_key = f"dss_{page_key}_year"

    companies_in_db = dl.get_companies(_CONSOLIDATED_DF) or COMPANIES
    default_co = (st.session_state.get("reporting_company") or
                  st.session_state.get("user_company") or companies_in_db[0])
    if default_co not in companies_in_db:
        default_co = companies_in_db[0]

    col1, col2, _ = st.columns([2, 1, 3])
    with col1:
        sel_co = st.selectbox(
            "Company to review", options=companies_in_db,
            index=companies_in_db.index(st.session_state.get(co_key, default_co)),
            key=f"sel_{page_key}_co"
        )
    with col2:
        avail_years = dl.get_years(_CONSOLIDATED_DF, sel_co) or [CURR_YEAR]
        sel_yr = st.selectbox(
            "Year", options=sorted(avail_years, reverse=True),
            key=f"sel_{page_key}_yr"
        )

    st.session_state[co_key] = sel_co
    st.session_state[yr_key] = sel_yr

    # Load data for selected and previous year
    hist = dl.get_company_hist(_CONSOLIDATED_DF, sel_co)

    def _make_inp_out(year):
        sd = dl.get_step_data(hist, year)
        sd_clean = {k: v for k, v in sd.items() if k in _VALID_TEMPLATE_FIELDS}
        inp = TemplateInputs(company=sel_co, year=year, **sd_clean)
        return inp, calculate(inp)

    inp,  out  = _make_inp_out(sel_yr)
    try:
        prev_inp, prev_out = _make_inp_out(sel_yr - 1)
    except Exception:
        prev_inp, prev_out = TemplateInputs(company=sel_co, year=sel_yr-1), None

    return sel_co, sel_yr, inp, out, prev_inp, prev_out, hist


def _compute_completeness(inp: "TemplateInputs", out: "TemplateOutputs") -> dict:
    """
    Returns {section_label: pct_complete} based on which fields have non-zero data.
    """
    def pct(*vals):
        filled = sum(1 for v in vals if v is not None and float(v) != 0)
        return int(filled / len(vals) * 100)

    fuel_vals = [inp.nat_gas, inp.coal_sub, inp.propane, inp.fuel_oil_heavy_a,
                 inp.diesel, inp.petrol, inp.biomass, inp.lpg, inp.other_fuels]
    fuel_filled = sum(1 for v in fuel_vals if v > 0)

    return {
        "ISO 14001":           pct(inp.total_sites, inp.iso_sites),
        "Production":          pct(inp.production),
        "Water":               pct(inp.water_withdrawals),
        "Energy — Electricity":pct(inp.renew_elec_purchased + inp.nonrenew_elec_purchased, inp.self_gen_elec),
        "Energy — Fuels":      min(100, int(fuel_filled / max(len(fuel_vals), 1) * 100)) if inp.nat_gas or inp.coal_sub else 0,
        "CO₂ Scope 1":         pct(out.total_co2_scope1),
        "CO₂ Scope 2":         pct(inp.co2_scope2_steam),
        "Waste":               pct(inp.waste_total, inp.waste_recovery),
        "Pathway 3 (SBTi)":    0,   # not captured in current template
        "Pathway 4 (H&S)":     0,   # not captured in current template
        "Pathway 4 (D&I)":     0,   # not captured in current template
    }


def _compute_readiness_score(completeness: dict, flags) -> tuple:
    """
    Score = weighted completeness average, minus penalties for flags.
    Returns (score: int, label: str)
    """
    weights = {
        "ISO 14001":1,"Production":2,"Water":2,
        "Energy — Electricity":3,"Energy — Fuels":3,
        "CO₂ Scope 1":3,"CO₂ Scope 2":2,"Waste":2,
        "Pathway 3 (SBTi)":1,"Pathway 4 (H&S)":1,"Pathway 4 (D&I)":1,
    }
    total_w  = sum(weights.values())
    raw      = sum(completeness.get(k,0) * w for k,w in weights.items()) / total_w
    n_errors   = sum(1 for f in flags if f.severity == "error")
    n_warnings = sum(1 for f in flags if f.severity == "warning")
    score = max(0, min(100, int(raw - n_errors * 10 - n_warnings * 3)))
    label = "Ready" if score >= 90 else "Review required" if score >= 70 else "Not ready"
    return score, label


# ─────────────────────────────────────────────────────────
# PAGE 4 -- VERIFICATION (dss+ only)  — fully live
# ─────────────────────────────────────────────────────────

def page_verification():
    if not st.session_state.is_dss:
        st.error("This section is restricted to dss+ analysts and managers."); return

    st.markdown("## Data Verification")
    st.caption("Select any company and reporting year from the consolidated dataset to review.")

    sel_co, sel_yr, inp, out, prev_inp, prev_out, hist = _dss_company_selector("verif")
    st.divider()

    # -- Real flags from formula_engine ----------------------------------------
    flags = validate_submission(inp, out, prev_out, threshold=20.0)

    # -- Also compute YoY field-level flags from the data ----------------------
    extra_flags = []
    yoy_fields = [
        ("nat_gas",              "Natural Gas"),
        ("renew_elec_purchased", "Renewable Electricity"),
        ("nonrenew_elec_purchased","Non-Renewable Electricity"),
        ("fuel_oil_heavy_a",     "Fuel Oil"),
        ("coal_sub",             "Coal"),
        ("water_withdrawals",    "Water Withdrawals"),
        ("production",           "Production"),
    ]
    from formula_engine import ValidationFlag
    for field, label in yoy_fields:
        cur  = getattr(inp,      field, 0) or 0
        prev = getattr(prev_inp, field, 0) or 0
        if prev > 0 and cur > 0:
            pct = (cur - prev) / abs(prev) * 100
            if abs(pct) > 20:
                direction = "increase" if pct > 0 else "decrease"
                extra_flags.append(ValidationFlag(
                    severity="warning",
                    message=f"{label} — >{20}% YoY {direction} ({pct:+.1f}%)",
                    detail=(f"{label}: {prev:,.0f} → {cur:,.0f} "
                            f"({'↑' if pct>0 else '↓'}{abs(pct):.1f}%). "
                            f"Verify with company documentation.")
                ))

    all_flags = flags + extra_flags

    # -- Completeness + summary metrics ----------------------------------------
    completeness = _compute_completeness(inp, out)
    avg_complete  = int(sum(completeness.values()) / len(completeness))
    n_err  = sum(1 for f in all_flags if f.severity == "error")
    n_warn = sum(1 for f in all_flags if f.severity == "warning")
    score, label = _compute_readiness_score(completeness, all_flags)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Company",      sel_co)
    c2.metric("Year",         str(sel_yr))
    c3.metric("Status",       "Ready" if score >= 90 else "Pending review")
    c4.metric("Completeness", f"{avg_complete}%")
    c5.metric("Open flags",   f"{n_warn} warning{'s' if n_warn!=1 else ''} · {n_err} error{'s' if n_err!=1 else ''}")
    st.divider()

    # -- Flag cards ------------------------------------------------------------
    if not st.session_state.get("flags_resolved_real"):
        st.session_state["flags_resolved_real"] = set()

    resolved_set = st.session_state["flags_resolved_real"]

    for i, flag in enumerate(all_flags):
        flag_id  = f"flag_{sel_co}_{sel_yr}_{i}"
        resolved = flag_id in resolved_set

        sev = "ok" if resolved else flag.severity
        icon_map  = {"ok":"✓", "warning":"!", "error":"✕", "warn":"!"}
        color_map = {"ok":"fc-ok fi-ok", "warning":"fc-warn fi-warn",
                     "error":"fc-error fi-error", "warn":"fc-warn fi-warn"}
        fc, fi = color_map.get(sev, "fc-ok fi-ok").split()
        icon   = icon_map.get(sev, "OK")
        title  = flag.message + (" — Approved" if resolved else "")
        detail = flag.detail

        # H3 FIX: escape flag content before injecting into HTML
        st.markdown(f"""<div class="flag-card {fc}">
          <div class="fc-icon {fi}">{_html.escape(icon)}</div>
          <div><div class="fc-title">{_html.escape(title)}</div>
               <div class="fc-detail">{_html.escape(detail)}</div></div>
        </div>""", unsafe_allow_html=True)

        if not resolved and flag.severity in ("warning","error"):
            cols = st.columns([6,1,1])
            with cols[1]:
                if st.button("Query", key=f"q_{flag_id}"):
                    st.toast(f"Query logged: {flag.message[:50]}...")
            with cols[2]:
                if flag.severity == "warning":
                    if st.button("Accept", key=f"a_{flag_id}", type="primary"):
                        resolved_set.add(flag_id)
                        st.session_state["flags_resolved_real"] = resolved_set
                        st.rerun()
                else:
                    if st.button("Send Back", key=f"sb_{flag_id}"):
                        st.toast(f"Submission returned to {sel_co} with error details.")

    st.divider()
    col_approve, col_export, _ = st.columns([1.5, 1.5, 3])
    with col_approve:
        warn_ids = [f"flag_{sel_co}_{sel_yr}_{i}"
                    for i, f in enumerate(all_flags) if f.severity == "warning"]
        if st.button("Approve All Warnings", type="primary"):
            resolved_set.update(warn_ids)
            st.session_state["flags_resolved_real"] = resolved_set
            st.rerun()
    with col_export:
        if st.button("Export Flag Report"):
            rows = [{"Flag": f.message, "Severity": f.severity, "Detail": f.detail,
                     "Status": "Resolved" if f"flag_{sel_co}_{sel_yr}_{i}" in resolved_set else "Open"}
                    for i, f in enumerate(all_flags)]
            export_df = pd.DataFrame(rows)
            st.download_button(
                "Download CSV", data=export_df.to_csv(index=False).encode(),
                file_name=f"flags_{sel_co.replace(' ','_')}_{sel_yr}.csv",
                mime="text/csv", key="dl_flags"
            )


# ─────────────────────────────────────────────────────────
# PAGE 5 -- AI READINESS (dss+ only) — fully live
# ─────────────────────────────────────────────────────────

def page_readiness():
    if not st.session_state.is_dss:
        st.error("This section is restricted to dss+ analysts and managers."); return

    st.markdown("## AI Readiness Check")
    st.caption("Select any company and reporting year to compute a live readiness score.")

    sel_co, sel_yr, inp, out, prev_inp, prev_out, hist = _dss_company_selector("ready")
    st.divider()

    # -- Compute live values ---------------------------------------------------
    flags        = validate_submission(inp, out, prev_out, threshold=20.0)
    completeness = _compute_completeness(inp, out)
    score, label = _compute_readiness_score(completeness, flags)
    n_errors     = sum(1 for f in flags if f.severity == "error")
    n_warnings   = sum(1 for f in flags if f.severity == "warning")

    renew_pct = (inp.renew_elec_purchased + inp.self_gen_elec) / max(out.total_electricity, 1) * 100
    prev_co2  = out.total_co2  # placeholder for YoY
    prev_e    = out.total_energy
    if prev_out:
        yoy_co2 = yoy_change(out.total_co2, prev_out.total_co2) or 0
        yoy_e   = yoy_change(out.total_energy, prev_out.total_energy) or 0
    else:
        yoy_co2 = yoy_e = 0

    # -- Gauge + summary -------------------------------------------------------
    col_score, col_info = st.columns([1, 3])
    with col_score:
        score_color = "#00916E" if score >= 80 else "#D97706" if score >= 60 else "#DC2626"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            number=dict(suffix="/100", font=dict(size=32, color=score_color)),
            gauge=dict(
                axis=dict(range=[0,100]),
                bar=dict(color=score_color, thickness=.25),
                steps=[
                    dict(range=[0,60],  color="#FEE2E2"),
                    dict(range=[60,80], color="#FEF3C7"),
                    dict(range=[80,100],color="#D1FAE5"),
                ],
                threshold=dict(line=dict(color="#065F46",width=3), thickness=.75, value=score)
            )
        ))
        fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown(f"### Report Readiness Score: **{score} / 100**")
        st.caption(f"{sel_co} · {sel_yr} Reporting Year · "
                   f"{n_errors} error{'s' if n_errors!=1 else ''}, "
                   f"{n_warnings} warning{'s' if n_warnings!=1 else ''}")
        if score >= 90:
            st.success(f"✅ {label} — submission can be included in consolidated report.")
        elif score >= 70:
            st.warning(f"⚠️ {label} — resolve open items before submission.")
        else:
            st.error(f"❌ {label} — significant data gaps must be addressed.")

        # Key live KPIs at a glance
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Energy KPI",   f"{out.energy_kpi:.2f} GJ/T")
        k2.metric("CO₂ KPI",      f"{out.co2_kpi:.3f} T/T")
        k3.metric("Water KPI",    f"{out.water_kpi:.2f} m³/T")
        k4.metric("Renewable %",  f"{renew_pct:.1f}%")

    st.divider()

    # -- Completeness by section -----------------------------------------------
    st.markdown("#### Data completeness by section")
    cols = st.columns(3)
    for i, (label_s, pct) in enumerate(completeness.items()):
        color = "#00916E" if pct == 100 else "#D97706" if pct >= 60 else "#DC2626"
        with cols[i % 3]:
            with st.container(border=True):
                st.caption(label_s)
                st.progress(pct / 100, text=f"{pct}%")

    st.divider()

    # ── ESG Analyst Chat (replaces old LLM insights) ──────────────────────────
    st.markdown("#### dss+ ESG Analyst")
    st.caption(f"Ask about {sel_co} {sel_yr} or any TIP company — powered by local AI (Ollama).")

    try:
        from chatbot.chatbot_engine import ESGChatbot
        _bot_key = f"_readiness_bot_{st.session_state.get('user_name','dss')}"
        if _bot_key not in st.session_state:
            st.session_state[_bot_key] = ESGChatbot(
                st.session_state.get("user_name", "dss_user"))
        _bot = st.session_state[_bot_key]

        _ok, _status = _bot.copilot.is_available()

        # Status bar
        _dot   = "🟢" if _ok else "🔴"
        _slabel = _bot.copilot.provider_label() if _ok else "Ollama not running — open from system tray"
        st.markdown(
            f'<div style="font-size:12px;color:#6B7280;margin-bottom:8px">'
            f'{_dot} {_slabel}</div>',
            unsafe_allow_html=True,
        )

        if not _ok:
            st.info("Start Ollama from your system tray, then reload this page.")
        else:
            # Quick-context chips for this company
            _chat_key = f"ai_msgs_{sel_co}_{sel_yr}"
            if _chat_key not in st.session_state:
                st.session_state[_chat_key] = []

            _msgs = st.session_state[_chat_key]

            # Render message history
            for _i, _m in enumerate(_msgs):
                _av = "👤" if _m["role"] == "user" else "🤖"
                with st.chat_message(_m["role"], avatar=_av):
                    st.markdown(_m["content"])
                    if _m.get("figure"):
                        st.plotly_chart(_m["figure"], use_container_width=True,
                                        key=f"ai_fig_{_i}")

            # Suggestion chips on empty state
            if not _msgs:
                _sugs = [
                    f"Summarise {sel_co} ESG performance in {sel_yr}",
                    f"Why did CO₂ intensity change for {sel_co.split()[0]}?",
                    f"Chart water intake for {sel_co.split()[0]} 2016–{sel_yr}",
                    f"Compare {sel_co.split()[0]} vs sector average in {sel_yr}",
                ]
                _sc = st.columns(2)
                for _si, _s in enumerate(_sugs):
                    with _sc[_si % 2]:
                        if st.button(_s, key=f"ai_chip_{_si}", use_container_width=True):
                            st.session_state[_chat_key].append(
                                {"role": "user", "content": _s, "figure": None})
                            with st.spinner("Thinking…"):
                                _resp = _bot.chat(_s)
                            st.session_state[_chat_key].append({
                                "role": "assistant",
                                "content": _resp.text,
                                "figure": _resp.figure,
                            })
                            st.rerun()

            # Chat input
            _q = st.chat_input(
                f"Ask about {sel_co} {sel_yr} or any ESG metric…",
                key=f"ai_input_{sel_co}_{sel_yr}",
            )
            if _q:
                st.session_state[_chat_key].append(
                    {"role": "user", "content": _q, "figure": None})
                with st.chat_message("user", avatar="👤"):
                    st.markdown(_q)

                with st.chat_message("assistant", avatar="🤖"):
                    _placeholder = st.empty()
                    _acc = ""
                    for _chunk in _bot.copilot.call_stream(
                        user_message  = _q,
                        data_context  = _bot.context.build_context_str(_q),
                        history       = _bot.history,
                        system_prompt = _bot.system_prompt,
                    ):
                        _acc += _chunk
                        _placeholder.markdown(_acc + "▌")

                    _placeholder.markdown(_acc)

                    _spec = _bot.graph.extract_spec(_acc)
                    _fig  = None
                    if _spec and not _bot.context.df.empty:
                        _fig = _bot.graph.build(_spec, _bot.context.df)
                        if _fig:
                            st.plotly_chart(_fig, use_container_width=True,
                                            key=f"ai_resp_fig_{len(_msgs)}")

                    _clean = _bot.graph.strip_spec(_acc)

                _bot.history.append({"role": "user",      "content": _q})
                _bot.history.append({"role": "assistant",  "content": _clean})
                if len(_bot.history) > 12:
                    _bot.history = _bot.history[-12:]
                _bot.logger.log(_q, _clean, _bot.classifier.classify(_q),
                                had_chart=(_fig is not None))

                st.session_state[_chat_key].append({
                    "role": "assistant", "content": _clean, "figure": _fig})
                st.rerun()

            # Clear button
            if _msgs:
                if st.button("🗑 Clear conversation", key="ai_clear_conv"):
                    st.session_state[_chat_key] = []
                    _bot.clear_history()
                    st.rerun()

    except ImportError:
        st.info("Chatbot module not available. Ensure chatbot/ folder is present.")
    except Exception as _e:
        st.error(f"Chat error: {_e}")


# ─────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    show_login()
else:
    show_sidebar()
    page = st.session_state.page
    if   page == "entry":         page_entry()
    elif page == "analysis":      page_analysis()
    elif page == "benchmarking":  page_benchmarking()
    elif page == "verification":  page_verification()
    elif page == "readiness":     page_readiness()