"""
TIP ESG Platform — Streamlit Frontend
======================================
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from formula_engine import (
    TemplateInputs, calculate, validate_submission,
    get_benchmarks, build_template_dataframe, fmt_num,
    yoy_change, ValidationFlag
)

from benchmark_loader import BenchmarkLoader

loader = BenchmarkLoader(data_dir=".")

df = loader.load_clean()
print(df.head())

# ─────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TIP ESG Platform · dss+",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0A2240 !important; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong
{ color: #ffffff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Sidebar nav buttons — always visible white text, no default box ── */
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
    box-shadow: none !important;
    outline: none !important;
    border: none !important;
    color: #ffffff !important;
}

/* ── Main area buttons ── */
.stButton > button {
    border-radius: 7px; font-weight: 500; font-size: 13px;
    border: 1.5px solid #D1D5DB; transition: all .15s;
}
.stButton > button:hover { border-color: #6B7280; }
div[data-testid="column"] .stButton > button.primary-btn {
    background: #00916E; color: #fff; border-color: #00916E;
}
/* ── Form inputs ── */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    border-radius: 7px; border: 1.5px solid #D1D5DB;
    font-size: 14px !important;
}
/* ── Cards ── */
.kpi-card {
    background: #fff; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 16px 18px; text-align: left;
}
.kpi-card .label { font-size: 11px; color: #6B7280;
    text-transform: uppercase; letter-spacing: .5px; font-weight: 500; }
.kpi-card .value { font-size: 26px; font-weight: 700;
    color: #111827; margin: 5px 0 2px; }
.kpi-card .unit  { font-size: 12px; color: #9CA3AF; }
.kpi-card .delta { font-size: 12px; font-weight: 600; margin-top: 4px; }
.delta-pos { color: #059669; }
.delta-neg { color: #DC2626; }
/* ── Stepper ── */
.step-bar { display:flex; align-items:center; gap:0;
    background:#fff; border:1px solid #E5E7EB; border-radius:10px;
    padding:16px 20px; margin-bottom:20px; }
.step-item { display:flex; align-items:center; flex:1; min-width:0; }
.step-circle { width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:12px; font-weight:700; flex-shrink:0; }
.sc-done   { background:#00916E; color:#fff; }
.sc-active { background:#1D4ED8; color:#fff; }
.sc-todo   { background:#F3F4F6; color:#9CA3AF;
    border:2px solid #E5E7EB; }
.step-label { font-size:11.5px; font-weight:500; margin-left:7px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sl-done   { color:#00916E; }
.sl-active { color:#1D4ED8; }
.sl-todo   { color:#9CA3AF; }
.step-line { flex:1; height:2px; background:#E5E7EB; margin:0 6px; min-width:8px; }
.sl-done-line { background:#00916E; }
/* ── Table legend ── */
.tbl-legend { display:flex; gap:14px; padding:10px 16px;
    background:#F9FAFB; border-top:1px solid #E5E7EB;
    border-radius:0 0 8px 8px; flex-wrap:wrap; }
.tl { display:flex; align-items:center; gap:5px;
    font-size:11px; color:#6B7280; }
.tl-sw { width:14px; height:14px; border-radius:3px;
    border:1px solid #D1D5DB; display:inline-block; }
/* ── Band chart ── */
.band-container { margin: 6px 0 12px; }
.band-row-wrap { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.band-lbl { font-size:12px; font-weight:500; color:#374151;
    width:170px; flex-shrink:0; }
.band-track { flex:1; height:18px; border-radius:4px;
    background:#F3F4F6; position:relative; }
.band-seg { position:absolute; top:0; height:100%; }
.band-pin { position:absolute; width:4px; height:28px;
    background:#0A2240; border-radius:2px; top:-5px;
    transform:translateX(-50%); }
.band-pin-val { position:absolute; font-size:10px; font-weight:700;
    color:#0A2240; top:-18px; transform:translateX(-50%);
    white-space:nowrap; background:#fff; padding:0 2px; }
.band-chip { font-size:11px; font-weight:600; padding:3px 9px;
    border-radius:10px; flex-shrink:0; }
.chip-top  { background:#D1FAE5; color:#065F46; }
.chip-mid  { background:#FEF3C7; color:#92400E; }
.chip-bot  { background:#FEE2E2; color:#991B1B; }
/* ── Flag cards ── */
.flag-card { display:flex; align-items:flex-start; gap:10px;
    padding:12px 14px; border-radius:8px; border:1px solid;
    margin-bottom:10px; }
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
/* ── AI card ── */
.ai-card { background:#fff; border:1px solid #E5E7EB;
    border-radius:10px; overflow:hidden; margin-bottom:12px; }
.ai-head  { display:flex; align-items:center; gap:8px;
    padding:10px 14px; background:#F9FAFB;
    border-bottom:1px solid #E5E7EB; }
.ai-pulse { width:8px; height:8px; border-radius:50%;
    background:#00916E; flex-shrink:0; }
.ai-title { font-size:13px; font-weight:600; color:#111827; }
.ai-badge { margin-left:auto; background:#E6F5F1; color:#007A5C;
    font-size:11px; padding:2px 9px; border-radius:10px;
    border:1px solid #6EE7B7; font-weight:500; }
.ai-body  { padding:12px 14px; font-size:13px; color:#374151;
    line-height:1.8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────────────────
HIST_YEARS = list(range(2009, 2023))   # 14 historical years: 2009–2022
CURR_YEAR  = 2023
LONG_YEARS = list(range(2009, 2024))
COMPANIES  = ["Bridgestone","Goodyear","Michelin","Continental",
              "Pirelli","Sumitomo","Yokohama","Cooper Tire","Hankook","Toyo Tires"]

# Historical raw data: 14 years 2009–2022
# Production scale factors from LONG_DATA["prod"] relative to 2019 (index 10=3.86M T)
_PROD_BASE = 3_860_798  # 2019 value
_PROD_LONG = [2.84, 3.51, 3.77, 3.52, 3.64, 3.62, 3.54, 3.63, 3.70, 3.91,
              3.86, 3.05, 3.32, 3.58]  # 2009-2022
_S = [p / 3.86 for p in _PROD_LONG]  # scale factors relative to 2019

def _scale(base_2019, custom=None):
    """Generate 14 values; last 4 override with exact data if provided."""
    vals = [round(base_2019 * s) for s in _S]
    if custom:
        for i, v in enumerate(custom):
            vals[10 + i] = v   # 2019=idx10, 2020=11, 2021=12, 2022=13
    return vals

# Renewable electricity grew from near-zero: 0% in 2009→ grows after 2013
_RENEW_PCT = [0, 0, 0, 0, 0, 0, 0, 2.3, 2.2, 9.7, 21.4, 31.4, 40.6, 48.3]  # 2009-2022
_TOTAL_ELEC_BASE = 12_978_503  # 2019 total electricity
_TOTAL_ELEC = [round(_TOTAL_ELEC_BASE * _S[i]) for i in range(14)]
_RENEW_HIST = [round(_TOTAL_ELEC[i] * _RENEW_PCT[i] / 100) for i in range(14)]
_NONRENEW_HIST = [_TOTAL_ELEC[i] - _RENEW_HIST[i] for i in range(14)]
# Override last 4 with exact values
_RENEW_HIST[10:] = [706_562, 1_528_836, 2_557_561, 4_082_923]
_NONRENEW_HIST[10:] = [12_271_131, 9_667_437, 10_297_758, 9_037_549]

HIST_RAW = {
    "total_sites":   [38,39,40,40,41,42,43,44,46,48,51,51,52,52],
    "iso_sites":     [36,38,39,39,40,41,42,43,45,47,51,51,52,52],
    "production":    _scale(_PROD_BASE, [3_860_798, 3_047_092, 3_320_000, 3_580_000]),
    "water_withdrawals": _scale(23_127_757, [23_127_757, 20_345_811, 21_147_924, 20_927_589]),
    "renew_elec_purchased":    _RENEW_HIST,
    "nonrenew_elec_purchased": _NONRENEW_HIST,
    "self_gen_elec":  [0,0,0,0,0,0,0,0,0,0,810,1_564,11_748,36_082],
    "purchased_steam": _scale(1_133_191, [1_133_191, 977_324, 1_036_304, 1_042_310]),
    "sold_electricity": [0,0,0,0,0,0,0,0,0,0,0,11_199,7_748,7_702],
    "nat_gas":       _scale(16_210_969, [16_210_969, 14_040_397, 15_939_109, 15_927_554]),
    "coal_sub":      _scale(456_997,    [456_997, 337_992, 360_848, 395_006]),
    "coal_brown":    [0]*14, "coal_other": [0]*14,
    "propane":       _scale(290_761,    [290_761, 214_440, 288_245, 334_563]),
    "fuel_oil_heavy_a": [round(781_730 * _S[i] * max(0.2, 1 - i*0.065)) for i in range(10)] + [781_730, 518_579, 440_915, 166_773],
    "fuel_oil_heavy_c": [0]*14,
    "diesel":  _scale(108_136,  [108_136, 91_916, 134_886, 184_346]),
    "petrol":  _scale(24_947,   [24_947, 12_976, 9_339, 8_032]),
    "biomass": [0]*14, "waste_tires_mt": [0]*14,
    "lpg":     _scale(1_237_839, [1_237_839, 1_124_479, 1_271_422, 1_329_571]),
    "other_fuels": [round(879 * _S[i]) for i in range(10)] + [879, 1_182, 1_256, 1_226],
    "co2_scope2_steam":       _scale(59_575, [59_575, 51_059, 48_013, 62_384]),
    "co2_scope2_electricity": [0]*14,
    "sold_steam": [0]*14,
    "waste_total":    _scale(352_000, [352_000, 295_000, 320_000, 335_000]),
    "waste_recovery": _scale(299_200, [299_200, 253_700, 275_200, 284_750]),
}

LONG_DATA = {
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
FUEL_MIX = {
    "Natural Gas": [46,46,47,47,47,46,47,47,48,49,49,49,49,49,50],
    "Electricity": [34.7,34.2,34.9,34.6,35.3,36.7,37.1,37.9,38.2,38.8,39.1,39.3,39.8,40.4,40.7],
    "Fuel Oil":    [8.5,6.7,6,5.8,5.1,4.8,3.7,3.2,3,2.6,2.4,1.8,1.4,0.5,0.5],
    "LPG":         [2.4,2.4,2.3,3.6,3.5,3.5,3.5,3.6,3.5,3.6,3.7,3.9,3.9,4.1,4.2],
    "Coal":        [3.2,3.1,2.8,2.8,3.7,3.6,2.3,2,2.1,1.6,1.4,1.2,1.1,1.2,1.2],
    "Other":       [5.2,7.6,7,5.4,5.4,5.4,6.4,5.3,4.7,4.4,4,4.8,4.3,4.8,3.4],
}

CLIENTS = {
    "bridgestone@tip-reporting.com": "Bridgestone",
    "michelin@tip-reporting.com":    "Michelin",
    "goodyear@tip-reporting.com":    "Goodyear",
    "yokohama@tip-reporting.com":    "Yokohama",
}

STEP_META = [
    ("ISO 14001","Certified sites and facility coverage"),
    ("Production","Annual production volume"),
    ("Water","Water withdrawals by source"),
    ("Energy","Electricity and fuel consumption"),
    ("CO₂","Emission inputs and auto-calculated totals"),
    ("Waste","Waste generated, recovered and eliminated"),
]

# ─────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "authenticated":   False,
        "user_name":       "",
        "user_company":    "",
        "user_email":      "",
        "is_dss":          False,
        "page":            "login",
        "step":            0,
        "template_done":   False,
        "consolidated_df": None,   # cached from SharePoint or local dummy
        "step_data": {
            "total_sites": 54, "iso_sites": 54,
            "production": 3_720_000,
            "water_withdrawals": 21_500_000,
            "renew_elec_purchased": 5_200_000,
            "nonrenew_elec_purchased": 8_500_000,
            "self_gen_elec": 45_000,
            "purchased_steam": 1_050_000,
            "sold_electricity": 8_000,
            "nat_gas": 16_100_000, "coal_sub": 380_000,
            "propane": 340_000, "fuel_oil_heavy_a": 150_000,
            "diesel": 190_000, "lpg": 1_350_000,
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
# CONSOLIDATED DATA LOADER
# Tries SharePoint first; falls back to built-in dummy data.
# Cached so SharePoint is only queried every 5 minutes.
# ─────────────────────────────────────────────────────────

def _generate_dummy_consolidated() -> pd.DataFrame:
    """Built-in dummy dataset — 10 TIP companies × 15 years."""
    import numpy as _np
    _np.random.seed(42)
    scales = [1.51,2.6,1.8,1.65,0.9,1.2,0.72,0.62,0.76,0.42]
    rows = []
    for company, scale in zip(COMPANIES, scales):
        rng = _np.random.RandomState(hash(company) % 2**31)
        for i, yr in enumerate(LONG_YEARS):
            prod  = scale*1e6*(1+0.025*i+rng.uniform(-0.04,0.04))
            if yr==2020: prod *= 0.79
            energy= prod*rng.uniform(8.5,10.2)
            co2_k = rng.uniform(0.68,0.92)*(1-0.018*i+rng.uniform(-0.015,0.015))
            co2   = prod*co2_k
            water = prod*rng.uniform(5.5,9.8)*(0.83 if yr==2020 else 1.0)
            renew = max(0.0,(yr-2013)*rng.uniform(2.5,6.5)) if yr>2013 else 0.0
            wt    = prod*rng.uniform(0.048,0.065)
            wr    = wt*rng.uniform(0.84,0.93)
            rows.append({"Company":company,"Year":int(yr),
                         "Prod_MT":prod,"Energy_GJ":energy,
                         "CO2_T":co2,"CO2_KPI":co2_k,
                         "Water_M3":water,"Renew_Pct":renew,
                         "Waste_MT":wt,"Recovery_MT":wr,
                         "Recovery_Pct":wr/wt*100,
                         "Energy_KPI":energy/prod,"Water_KPI":water/prod})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_consolidated_data():
    """
    
    Load consolidated benchmark data.
    IMPORTANT:
    - NO Streamlit UI calls allowed here
    - Cache-safe: pure data loading

    """
    
    from local_storage import get_storage
    storage = get_storage()
    return storage.load_benchmark_data()


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def get_current_outputs():
    sd = st.session_state.step_data
    inp = TemplateInputs(
        company=st.session_state.user_company,
        year=CURR_YEAR,
        **{k: sd.get(k, 0) for k in [
            "total_sites","iso_sites","production","water_withdrawals",
            "renew_elec_purchased","nonrenew_elec_purchased","self_gen_elec",
            "purchased_steam","sold_electricity","nat_gas","coal_sub",
            "propane","fuel_oil_heavy_a","diesel","lpg",
            "co2_scope2_steam","waste_total","waste_recovery",
        ]}
    )
    return inp, calculate(inp)

def get_hist_outputs():
    outs = []
    for i, yr in enumerate(HIST_YEARS):
        inp = TemplateInputs(
            company=st.session_state.user_company, year=yr,
            **{k: HIST_RAW[k][i] for k in HIST_RAW}
        )
        outs.append((yr, inp, calculate(inp)))
    return outs

def kpi_card_html(label, value, unit, delta, delta_positive=True):
    delta_cls = "delta-pos" if delta_positive else "delta-neg"
    arrow = "▼" if delta_positive else "▲"
    return f"""<div class="kpi-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="unit">{unit}</div>
        <div class="delta {delta_cls}">{arrow} {delta}</div>
    </div>"""

def band_html(label, val, q25, median, q75, unit, lower_better=True):
    span    = q75 - q25
    bmin    = q25 - span * 0.4
    bmax    = q75 + span * 0.4
    rng     = bmax - bmin if (bmax - bmin) != 0 else 1
    pos_pct = max(2, min(98, (val - bmin) / rng * 100))
    if lower_better:
        top_cls = "chip-top" if val <= q25 else "chip-mid" if val <= median else "chip-bot"
        top_lbl = "Top 25%" if val <= q25 else "Average"   if val <= median else "Below avg"
    else:
        top_cls = "chip-top" if val >= q75 else "chip-mid" if val >= median else "chip-bot"
        top_lbl = "Top 25%" if val >= q75 else "Average"   if val >= median else "Below avg"
    # Pre-compute display value — f-strings cannot contain conditionals in format specs
    val_str = f"{val:.3f}" if isinstance(val, float) and val < 10 else fmt_num(val)
    return f"""
    <div class="band-row-wrap">
      <div class="band-lbl">{label}</div>
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

            if st.button("Sign in →", type="primary", width="stretch"):
                email_l   = email.strip().lower()
                is_dss    = "@consultdss.com" in email_l
                is_client = email_l in CLIENTS
                if not is_dss and not is_client:
                    st.error("Email not recognised. "
                             "Try employee@consultdss.com or bridgestone@tip-reporting.com")
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

            st.caption("Demo: employee@consultdss.com · bridgestone@tip-reporting.com (any password)")

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
            st.markdown(f"""
            <div style="margin:10px 12px 0;padding:10px 12px;background:rgba(255,255,255,.06);border-radius:8px">
              <div style="color:rgba(255,255,255,.4);font-size:10px;text-transform:uppercase;letter-spacing:.6px">Company</div>
              <div style="color:#fff;font-size:13px;font-weight:500;margin-top:3px">{st.session_state.user_company}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="padding:12px 8px 4px;color:rgba(255,255,255,.3);font-size:10px;text-transform:uppercase;letter-spacing:.7px">Data & Reports</div>', unsafe_allow_html=True)

        pages = [
            ("entry",        "KPI Data Entry"),
            ("analysis",     "Analysis"),
            ("benchmarking", "Benchmarking"),
        ]
        for page_id, label in pages:
            active = st.session_state.page == page_id
            if active:
                st.markdown(f"""
                <div style="background:rgba(0,145,110,0.85);border-radius:8px;padding:9px 12px;
                    margin-bottom:2px;color:#fff;font-size:13.5px;font-weight:600;">
                  {label}
                </div>""", unsafe_allow_html=True)
            else:
                if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
                    st.session_state.page = page_id
                    st.rerun()

        if st.session_state.is_dss:
            st.markdown('<div style="padding:12px 8px 4px;color:rgba(255,255,255,.3);font-size:10px;text-transform:uppercase;letter-spacing:.7px;margin-top:8px">dss+ Internal</div>', unsafe_allow_html=True)
            for page_id, label in [("verification", "Verification"), ("readiness", "AI Readiness")]:
                active = st.session_state.page == page_id
                if active:
                    st.markdown(f"""
                    <div style="background:rgba(0,145,110,0.85);border-radius:8px;padding:9px 12px;
                        margin-bottom:2px;color:#fff;font-size:13.5px;font-weight:600;">
                      {label}
                    </div>""", unsafe_allow_html=True)
                else:
                    if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
                        st.session_state.page = page_id
                        st.rerun()

        st.markdown("---")
        name_init = "".join(p[0].upper() for p in st.session_state.user_name.split()[:2])
        role_lbl  = "dss+ Employee" if st.session_state.is_dss else f"Client · {st.session_state.user_company}"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:9px">
          <div style="width:32px;height:32px;border-radius:50%;background:#00916E;color:#fff;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0">{name_init}</div>
          <div>
            <div style="color:#fff;font-size:13px;font-weight:500">{st.session_state.user_name}</div>
            <div style="color:rgba(255,255,255,.4);font-size:11px">{role_lbl}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Sign out", width="stretch"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ─────────────────────────────────────────────────────────
# PAGE 1 — STEPPER + TEMPLATE TABLE
# ─────────────────────────────────────────────────────────
def render_stepper_bar():
    items = ""
    for i, (name, _) in enumerate(STEP_META):
        if i < st.session_state.step:
            sc, sl, icon = "sc-done", "sl-done", "✓"
        elif i == st.session_state.step:
            sc, sl, icon = "sc-active", "sl-active", str(i+1)
        else:
            sc, sl, icon = "sc-todo", "sl-todo", str(i+1)
        line = f'<div class="step-line {"sl-done-line" if i>0 and i<=st.session_state.step else ""}"></div>' if i > 0 else ""
        items += f'{line}<div class="step-item"><div class="step-circle {sc}">{icon}</div><span class="step-label {sl}">{name}</span></div>'
    st.markdown(f'<div class="step-bar">{items}</div>', unsafe_allow_html=True)

STEP_FIELDS = [
    # Step 0: ISO 14001
    [("total_sites","Total no. of sites","All facilities globally",None),
     ("iso_sites","ISO 14001 certified sites","Sites with active certification",None)],
    # Step 1: Production
    [("production","Annual production","Total weight of all products (metric T)",None)],
    # Step 2: Water
    [("water_withdrawals","Total water withdrawals","All sources: surface, well, municipal (m³)",None)],
    # Step 3: Energy
    [("renew_elec_purchased","Renewable electricity purchased","Grid-purchased green electricity (GJ)",None),
     ("nonrenew_elec_purchased","Non-renewable electricity purchased","Standard grid electricity (GJ)",None),
     ("self_gen_elec","Self-generated renewable electricity","On-site solar, wind (GJ)",None),
     ("purchased_steam","Purchased steam (GJ)",None,None),
     ("nat_gas","Natural gas (GJ LHV)",None,None),
     ("coal_sub","Coal – all types (GJ LHV)",None,None),
     ("fuel_oil_heavy_a","Fuel oil (GJ LHV)",None,None),
     ("diesel","Diesel (GJ LHV)",None,None),
     ("lpg","LPG (GJ LHV)",None,None)],
    # Step 4: CO2
    [("co2_scope2_steam","Scope 2 – Steam CO₂ (T.CO₂)","Company-provided Scope 2 from purchased steam",None)],
    # Step 5: Waste
    [("waste_total","Total waste generated (metric T)",None,None),
     ("waste_recovery","Waste sent to recovery (metric T)",None,None)],
]

def page_entry():
    st.markdown(f"## {'KPI Data Entry' if not st.session_state.template_done else 'ESG KPI Template — 2023'}")

    if st.session_state.template_done:
        col_a, col_b = st.columns([5,1])
        with col_b:
            if st.button("← Edit Inputs"):
                st.session_state.template_done = False
                st.session_state.step = 0
                st.rerun()

        # Excel-style sheet tabs — mirrors the template's multiple sheets
        tab_main, tab_elec, tab_waste, tab_qual, tab_conv = st.tabs([
            "📋 Main Data Input",
            "⚡ Electricity by Country",
            "🗑️ Waste",
            "💬 Qualitative Data",
            "🔢 Conversion Tables",
        ])
        with tab_main:
            render_template_table()
        with tab_elec:
            render_electricity_tab()
        with tab_waste:
            render_waste_tab()
        with tab_qual:
            render_qualitative_tab()
        with tab_conv:
            render_conversion_tab()
        return

    render_stepper_bar()
    step = st.session_state.step
    name, desc = STEP_META[step]
    fields = STEP_FIELDS[step]

    with st.container(border=True):
        st.markdown(f"### Step {step+1} of {len(STEP_META)} — {name}")
        st.caption(desc)
        st.divider()

        # Calc current outputs for live preview
        inp, out = get_current_outputs()

        n_cols = 2 if len(fields) > 3 else 1
        cols_list = st.columns(n_cols)
        for idx, fdef in enumerate(fields):
            key, label = fdef[0], fdef[1]
            sublabel = fdef[2] if fdef[2] else ""
            hist_val  = HIST_RAW.get(key, [None]*4)[-1] if key in HIST_RAW else None
            with cols_list[idx % n_cols]:
                if sublabel:
                    st.caption(sublabel)
                val = st.number_input(
                    label,
                    value=float(st.session_state.step_data.get(key, 0)),
                    step=1.0, format="%.0f",
                    key=f"input_{key}"
                )
                st.session_state.step_data[key] = val
                if hist_val:
                    st.caption(f"2022 reference: {fmt_num(hist_val)}")

        # Live-calculated preview for this step
        st.divider()
        inp2, out2 = get_current_outputs()
        if step == 0:
            c1, c2 = st.columns(2)
            c1.metric("% ISO Certified", f"{out2.pct_certified*100:.1f}%")
        elif step == 1:
            st.metric("Production entered", fmt_num(inp2.production) + " metric T")
        elif step == 2:
            c1, c2 = st.columns(2)
            c1.metric("Water withdrawals", fmt_num(inp2.water_withdrawals) + " m³")
            c2.metric("Water intensity KPI", f"{out2.water_kpi:.2f} m³/T")
        elif step == 3:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total electricity",  fmt_num(out2.total_electricity) + " GJ")
            c2.metric("Total energy",        fmt_num(out2.total_energy) + " GJ")
            c3.metric("Energy intensity KPI", f"{out2.energy_kpi:.2f} GJ/T")
        elif step == 4:
            c1, c2, c3 = st.columns(3)
            c1.metric("Scope 1 CO₂", fmt_num(out2.total_co2_scope1) + " T")
            c2.metric("Scope 2 CO₂", fmt_num(out2.total_co2_scope2) + " T")
            c3.metric("CO₂ intensity KPI", f"{out2.co2_kpi:.3f} T/T")
        elif step == 5:
            c1, c2, c3 = st.columns(3)
            c1.metric("Waste elimination",   fmt_num(out2.waste_elimination) + " T")
            c2.metric("Recovery rate",        f"{out2.waste_recovery_pct*100:.1f}%")
            ok_str = "✅ Consistent" if out2.check_waste else "❌ Inconsistent"
            c3.metric("Waste check", ok_str)

        st.divider()
        nav_l, nav_r = st.columns([1,1])
        with nav_l:
            if step > 0 and st.button("← Previous step"):
                st.session_state.step -= 1
                st.rerun()
        with nav_r:
            if step < len(STEP_META) - 1:
                if st.button("Save & Continue →", type="primary"):
                    st.session_state.step += 1
                    st.rerun()
            else:
                if st.button("✓ Generate Template", type="primary"):
                    st.session_state.template_done = True
                    st.rerun()

def render_template_table():
    raw_company = st.session_state.user_company or "Your Company"
    company = raw_company if raw_company != "All Companies" else "TIP Member Company"

    # ── Header matching the Excel template ──────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
        background:#fff;border:1px solid #E5E7EB;border-radius:10px;
        padding:18px 24px;margin-bottom:14px">
      <div>
        <div style="font-size:17px;font-weight:700;color:#0A2240;letter-spacing:-.2px">
          Tire Industry Project — Key Performance Indicators
        </div>
        <div style="font-size:26px;font-weight:800;color:#00916E;margin-top:5px;letter-spacing:-.4px">
          {company}
        </div>
        <div style="font-size:12px;color:#9CA3AF;margin-top:4px">Corporate units · ESG KPI Template — {CURR_YEAR}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:.5px">Reporting year</div>
        <div style="font-size:36px;font-weight:800;color:#0A2240;line-height:1">{CURR_YEAR}</div>
        <div style="font-size:11px;color:#9CA3AF;margin-top:3px">Data range: 2009–{CURR_YEAR}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("✅ Template generated from your inputs. Blue cells = company input · grey italic = auto-calculated formula.", icon="📋")

    inp, out = get_current_outputs()
    hist = get_hist_outputs()

    ROWS = [
        ("section","ISO 14001",None,None,None),
        ("input","Total no. of sites","no.", "total_sites", None),
        ("input","ISO 14001 certified sites","no.","iso_sites",None),
        ("calc","% certified sites","%",None, lambda i,o: f"{o.pct_certified*100:.1f}%"),
        ("section","Production",None,None,None),
        ("input","Production","metric T","production",None),
        ("section","Water",None,None,None),
        ("input","Water withdrawals","m³","water_withdrawals",None),
        ("calc","Water intensity KPI","m³/T",None,lambda i,o:f"{o.water_kpi:.2f}"),
        ("section","Energy",None,None,None),
        ("calc","Total Electricity","GJ",None,lambda i,o:f"{o.total_electricity:,.0f}"),
        ("input","— Renewable electricity","GJ","renew_elec_purchased",None),
        ("input","— Non-renewable electricity","GJ","nonrenew_elec_purchased",None),
        ("input","— Self-generated renewable","GJ","self_gen_elec",None),
        ("input","Purchased Steam","GJ","purchased_steam",None),
        ("input","Natural Gas","GJ LHV","nat_gas",None),
        ("input","Coal (all types)","GJ LHV","coal_sub",None),
        ("input","Fuel Oil","GJ LHV","fuel_oil_heavy_a",None),
        ("input","Diesel","GJ LHV","diesel",None),
        ("input","LPG","GJ LHV","lpg",None),
        ("calc","TOTAL ENERGY","GJ LHV",None,lambda i,o:f"{o.total_energy:,.0f}"),
        ("calc","Energy intensity KPI","GJ/T",None,lambda i,o:f"{o.energy_kpi:.2f}"),
        ("section","CO₂ Emissions",None,None,None),
        ("input","Scope 2 – Steam","T.CO₂","co2_scope2_steam",None),
        ("calc","CO₂ – Natural Gas","T.CO₂",None,lambda i,o:f"{o.co2_nat_gas:,.0f}"),
        ("calc","CO₂ – Coal","T.CO₂",None,lambda i,o:f"{o.co2_coal:,.0f}"),
        ("calc","CO₂ – Fuel Oil + Diesel","T.CO₂",None,lambda i,o:f"{o.co2_fuel_oil+o.co2_diesel:,.0f}"),
        ("calc","CO₂ – LPG","T.CO₂",None,lambda i,o:f"{o.co2_lpg:,.0f}"),
        ("calc","TOTAL CO₂ Scope 1","T.CO₂",None,lambda i,o:f"{o.total_co2_scope1:,.0f}"),
        ("calc","TOTAL CO₂ Scope 2","T.CO₂",None,lambda i,o:f"{o.total_co2_scope2:,.0f}"),
        ("calc","TOTAL CO₂ (S1+S2)","T.CO₂",None,lambda i,o:f"{o.total_co2:,.0f}"),
        ("calc","CO₂ intensity KPI","T.CO₂/T",None,lambda i,o:f"{o.co2_kpi:.3f}"),
        ("section","Waste",None,None,None),
        ("input","Total waste generated","metric T","waste_total",None),
        ("input","Waste to recovery","metric T","waste_recovery",None),
        ("calc","Waste to elimination","metric T",None,lambda i,o:f"{o.waste_elimination:,.0f}"),
        ("calc","% recovery rate","%",None,lambda i,o:f"{o.waste_recovery_pct*100:.1f}%"),
    ]

    data = []
    for rdef in ROWS:
        rtype, label, unit, key, fn = rdef
        if rtype == "section":
            row = {"Indicator": f"▸ {label}", "Unit": ""}
            for yr, hi, ho in hist: row[str(yr)] = ""
            row[str(CURR_YEAR)] = ""
            row["YoY %"] = ""
            data.append({"_type":"section","_row":row})
            continue

        row = {"Indicator": label, "Unit": unit or ""}
        hist_nums = []
        for yr, hi, ho in hist:
            v = getattr(hi, key, None) if key else None
            if v is None and fn: v = fn(hi, ho)
            row[str(yr)] = f"{float(v):,.0f}" if isinstance(v,(int,float)) and not isinstance(v,str) else (v if v else "—")
            try: hist_nums.append(float(str(v).replace(",","").replace("%","").replace("—","0")))
            except: hist_nums.append(0)

        cv = getattr(inp, key, None) if key else None
        if cv is None and fn: cv = fn(inp, out)
        row[str(CURR_YEAR)] = f"{float(cv):,.0f}" if isinstance(cv,(int,float)) and not isinstance(cv,str) else (cv if cv else "—")

        try:
            cn = float(str(cv).replace(",","").replace("%",""))
            pn = hist_nums[-1] if hist_nums else 0
            yoy = (cn - pn) / abs(pn) * 100 if pn else None
            row["YoY %"] = f"{yoy:+.1f}%" if yoy is not None else "—"
        except:
            row["YoY %"] = "—"
        data.append({"_type":rtype,"_row":row,"_key":key,"_label":label})

    all_rows  = [d["_row"] for d in data]
    all_types = [d["_type"] for d in data]
    df_tbl = pd.DataFrame(all_rows)

    # Freeze Indicator+Unit; highlight 2023 column differently
    curr_col = str(CURR_YEAR)

    def style_row(row, idx):
        rtype = all_types[idx]
        styles = []
        for col in df_tbl.columns:
            if rtype == "section":
                styles.append(
                    "background-color:#E8F5F0;color:#065F46;font-weight:800;"
                    "font-size:13px;border-top:2px solid #6EE7B7;padding-top:8px;padding-bottom:8px;"
                    "letter-spacing:.3px;text-transform:uppercase"
                )
            elif col == curr_col and rtype == "input":
                styles.append("background-color:#DBEAFE;color:#1E40AF;font-weight:700")
            elif col == curr_col and rtype == "calc":
                styles.append("background-color:#EFF6FF;color:#1D4ED8;font-style:italic")
            elif rtype == "calc":
                styles.append("background-color:#F8FAFC;color:#6B7280;font-style:italic")
            elif rtype == "input":
                styles.append("background-color:#F0F9FF;")
            else:
                styles.append("")
        return styles

    styled = df_tbl.style.apply(lambda row: style_row(row, row.name), axis=1)
    # Height: ~38px per row
    tbl_height = min(900, max(400, len(all_rows) * 36 + 60))
    st.dataframe(styled, hide_index=True, height=tbl_height, use_container_width=True)
    st.markdown("""<div class="tbl-legend">
      <div class="tl"><div class="tl-sw" style="background:#F0F9FF;border-color:#BAE6FD"></div>Company input (historical)</div>
      <div class="tl"><div class="tl-sw" style="background:#DBEAFE;border-color:#93C5FD"></div>Company input (2023)</div>
      <div class="tl"><div class="tl-sw" style="background:#EFF6FF;border-color:#A5B4FC"></div>Auto-calculated (2023)</div>
      <div class="tl"><div class="tl-sw" style="background:#F8FAFC;border-color:#E5E7EB"></div>Auto-calculated (historical)</div>
    </div>""", unsafe_allow_html=True)


def render_electricity_tab():
    """Editable electricity by country — mirrors the Excel 'Electricity data input' sheet."""
    st.markdown("#### Non-Renewable Electricity Purchased by Country")
    st.caption("Enter MWh values per country per year. Country-specific IEA emission factors are applied for Scope 2 calculations.")

    ELEC_COUNTRIES = [
        "Canada","Chile","Mexico","United States",
        "Australia","Japan","Korea","New Zealand",
        "Austria","Belgium","Czech Republic","Denmark","Finland","France",
        "Germany","Italy","Netherlands","Poland","Portugal","Spain",
        "Sweden","United Kingdom",
        "China","India","Indonesia","Malaysia","Thailand","Vietnam",
        "Brazil","South Africa","Turkey",
    ]

    # Use session state to persist edits across reruns
    if "elec_data" not in st.session_state:
        elec_rows = []
        for country in ELEC_COUNTRIES:
            row = {"Country": country, "Unit": "MWh"}
            for yr in range(2009, 2024):
                row[str(yr)] = 0
            elec_rows.append(row)
        st.session_state.elec_data = pd.DataFrame(elec_rows)

    df_elec = st.session_state.elec_data

    # Build column config: Country & Unit locked, year cols editable
    col_cfg = {
        "Country": st.column_config.TextColumn("Country", disabled=True, width="medium"),
        "Unit":    st.column_config.TextColumn("Unit", disabled=True, width="small"),
    }
    for yr in range(2009, 2024):
        col_cfg[str(yr)] = st.column_config.NumberColumn(
            str(yr), min_value=0, format="%d", width="small"
        )

    edited = st.data_editor(
        df_elec,
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
        height=900,
        key="elec_editor",
    )
    st.session_state.elec_data = edited

    col_a, col_b = st.columns([3,1])
    with col_b:
        if st.button("Reset to zero", key="elec_reset"):
            for yr in range(2009, 2024):
                st.session_state.elec_data[str(yr)] = 0
            st.rerun()

    st.info("ℹ️ Country-level data enables precise location-based Scope 2 calculations. Leave as 0 if the country has no operations.")
    total_2023 = edited["2023"].sum() if "2023" in edited.columns else 0
    st.metric("Total Non-Renewable Electricity 2023 (all countries)", f"{total_2023:,.0f} MWh")


def render_waste_tab():
    """Waste detail — mirrors the Excel 'Waste' sheet."""
    inp, out = get_current_outputs()
    hist = get_hist_outputs()
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
        ("calc","Consistency check","—",None,lambda i,o:"✅ OK" if o.check_waste else "❌ Error"),
        ("calc","Recovery rate","%",None,lambda i,o:f"{o.waste_recovery_pct*100:.1f}%"),
        ("calc","Waste intensity","kg/T prod",None,lambda i,o:f"{i.waste_total/i.production*1000:.2f}" if i.production else "—"),
    ]
    data = []
    for rdef in WASTE_ROWS:
        rtype, label, unit, key, fn = rdef
        if rtype == "section":
            row = {"Indicator": f"▸ {label}", "Unit": ""}
            for yr, hi, ho in hist: row[str(yr)] = ""
            row[str(CURR_YEAR)] = ""; row["YoY %"] = ""
            data.append({"_type":"section","_row":row}); continue
        row = {"Indicator": label, "Unit": unit or ""}
        hist_nums = []
        for yr, hi, ho in hist:
            v = getattr(hi, key, None) if key else None
            if v is None and fn: v = fn(hi, ho)
            row[str(yr)] = str(v) if v is not None else "—"
            try: hist_nums.append(float(str(v).replace(",","").replace("%","").replace("—","0")))
            except: hist_nums.append(0)
        cv = getattr(inp, key, None) if key else None
        if cv is None and fn: cv = fn(inp, out)
        row[str(CURR_YEAR)] = str(cv) if cv is not None else "—"
        try:
            cn = float(str(cv).replace(",","").replace("%",""))
            pn = hist_nums[-1] if hist_nums else 0
            yoy = (cn - pn) / abs(pn) * 100 if pn else None
            row["YoY %"] = f"{yoy:+.1f}%" if yoy is not None else "—"
        except: row["YoY %"] = "—"
        data.append({"_type":rtype,"_row":row})

    all_rows  = [d["_row"] for d in data]
    all_types = [d["_type"] for d in data]
    df_w = pd.DataFrame(all_rows)
    def _style_waste(row, idx):
        rt = all_types[idx]
        return [
            "background:#F0FDF8;font-weight:700;color:#065F46" if rt=="section"
            else "background:#DBEAFE;font-weight:600" if (rt=="input" and col==str(CURR_YEAR))
            else "background:#F0F9FF" if rt=="input"
            else "background:#F8FAFC;font-style:italic;color:#6B7280"
            for col in df_w.columns
        ]
    st.dataframe(df_w.style.apply(lambda row: _style_waste(row, row.name), axis=1),
                 hide_index=True, use_container_width=True, height=400)

    st.divider()
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Waste 2023", f"{inp.waste_total:,.0f} T")
    c2.metric("Recovery Rate",    f"{out.waste_recovery_pct*100:.1f}%")
    c3.metric("Consistency",      "✅ OK" if out.check_waste else "❌ Error")


def render_qualitative_tab():
    """Qualitative data entry — structured exactly as per the WBCSD TIP Excel 'Qualitative data' sheet."""
    st.markdown("""
    <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:14px 18px;margin-bottom:16px;font-size:13px;color:#374151;line-height:1.7">
    This section gathers qualitative data that will help gain additional insights for better interpretation of your 
    quantitative data, as well as better understanding of industry trends. Please report your company's main programs, 
    trends, or actions that are already implemented, under implementation or planned by your whole organization.<br>
    <span style="color:#9CA3AF;font-size:12px">Non-public information will be kept confidential and only used at an aggregated level with no mention to the company.</span>
    </div>
    """, unsafe_allow_html=True)

    def qual_section(icon, title, questions):
        """Renders one KPI section with Public / Non-public / Comments columns."""
        st.markdown(f"""
        <div style="background:#0A2240;color:#fff;font-size:13px;font-weight:700;
            padding:9px 16px;border-radius:8px 8px 0 0;margin-top:18px;margin-bottom:0">
          {icon} {title}
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            for q_label, q_hint, q_key in questions:
                st.markdown(f"**{q_label}**")
                if q_hint:
                    st.caption(q_hint)
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.text_area("📢 Public information",
                        key=f"pub_{title}_{q_key}", height=90,
                        placeholder="Information that can be included in the Global KPIs Report…")
                with c2:
                    st.text_area("🔒 Non-public (confidential)",
                        key=f"nonpub_{title}_{q_key}", height=90,
                        placeholder="Used only at aggregated level, no company attribution…")
                with c3:
                    st.text_area("💬 Other comments",
                        key=f"cmt_{title}_{q_key}", height=90,
                        placeholder="Any additional remarks…")
                st.divider()

    # ── ENERGY ─────────────────────────────────────────────
    qual_section("⚡", "Energy", [
        ("Program — Management approach",
         "Explain how your organization manages the energy topic: policies, commitments, ISO 50001 certifications, goals & targets, responsibilities, resources, grievance mechanisms, specific actions and initiatives.",
         "program"),
        ("Impacts",
         "Include the expected impacts related to the program initiatives. Do you expect the efforts to positively or negatively impact the Energy KPI?",
         "impacts"),
        ("Impacts — Covid-19 / disruptions",
         "How did the Covid-19 pandemic (or other major disruptions) affect your programs and KPIs on Energy?",
         "covid"),
        ("Specific projects completed / underway",
         "Report specific projects related to energy that you are currently running, implementing or planning — routine activities (closed loop systems, equipment upgrades, on-site renewable energy, etc.) or innovative projects. Include expected KPI impacts.",
         "projects"),
        ("Type of energy / renewable installations",
         "If you reported data in the 'self-generated and consumed renewable electricity on-site' field, please describe the installation type, number of sites, and any supplier contracts. You may also describe the source of purchased steam.",
         "type"),
    ])

    # ── CO₂ ────────────────────────────────────────────────
    qual_section("☁️", "CO₂ Emissions", [
        ("Program — Management approach",
         "Explain how your organization manages CO₂: policies, commitments, goals & targets, responsibilities, resources, grievance mechanisms, specific actions and initiatives.",
         "program"),
        ("Impacts",
         "Include the expected impacts related to the program initiatives. Do you expect the efforts to positively or negatively impact the CO₂ KPI?",
         "impacts"),
        ("Impacts — Covid-19 / disruptions",
         "How did the Covid-19 pandemic (or other disruptions) affect your programs and KPIs on CO₂?",
         "covid"),
        ("Specific projects completed / underway",
         "Report specific projects related to CO₂ emissions reduction — routine activities (equipment upgrades, fuel switching, etc.) or innovative actions. Include expected KPI impacts.",
         "projects"),
    ])

    # ── WATER ───────────────────────────────────────────────
    qual_section("💧", "Water", [
        ("Program — Management approach",
         "Explain how your organization manages water: policies, commitments, goals & targets, responsibilities, resources, grievance mechanisms, specific actions and initiatives.",
         "program"),
        ("Impacts",
         "Include the expected impacts related to the program initiatives. Do you expect the efforts to positively or negatively impact the Water KPI?",
         "impacts"),
        ("Impacts — Covid-19 / disruptions",
         "How did the Covid-19 pandemic (or other disruptions) affect your programs and KPIs on Water?",
         "covid"),
        ("Specific projects completed / underway",
         "Report specific projects related to water — routine activities (leak detection, closed loop systems, equipment upgrades, etc.) or innovative actions. Include expected KPI impacts.",
         "projects"),
    ])

    # ── ENVIRONMENTAL MANAGEMENT ────────────────────────────
    qual_section("🌿", "Environmental Management (ISO 14001)", [
        ("Program — Management approach",
         "Explain how your organization manages environmental management. Does your organisation have a 100% ISO 14001 certification rate target in your overall environmental policy?",
         "program"),
        ("Impacts — Covid-19 / disruptions",
         "How did the Covid-19 pandemic (or other disruptions) affect your programs and KPI on Environmental Management?",
         "covid"),
        ("Specific projects completed / underway",
         "Report specific projects related to environmental management — obtaining or maintaining ISO 14001 or other environmental certifications, and any innovative initiatives.",
         "projects"),
    ])

    # ── WASTE ───────────────────────────────────────────────
    qual_section("🗑️", "Waste", [
        ("Program — Management approach",
         "Explain how your organization manages waste: policies, commitments, ISO certifications, goals & targets, responsibilities, resources, grievance mechanisms, specific actions and initiatives.",
         "program"),
        ("Impacts",
         "Include the expected impacts related to the program initiatives. Do you expect the efforts to positively or negatively impact the Waste KPI?",
         "impacts"),
        ("Impacts — Covid-19 / disruptions",
         "How did the Covid-19 pandemic (or other disruptions) affect your programs and KPI on Waste?",
         "covid"),
        ("Specific projects completed / underway",
         "Report the specific projects related to waste that you are currently running, implementing or planning.",
         "projects"),
    ])

    # ── ADDITIONAL INFORMATION ──────────────────────────────
    st.markdown("""
    <div style="background:#0A2240;color:#fff;font-size:13px;font-weight:700;
        padding:9px 16px;border-radius:8px 8px 0 0;margin-top:18px;margin-bottom:0">
      📝 Additional Information
    </div>
    """, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Other information that may affect the five environmental KPIs**")
        st.caption("Please provide any other information about potential changes in your organization that can affect the five environmental KPIs in the TIP report. This could concern production variations or changes that could positively or negatively impact the performance of the 5 reported KPIs.")
        st.text_area("Additional comments",
            key="qual_additional", height=120,
            placeholder="e.g. major plant closures, acquisitions, production restructuring, changes in reporting scope, methodology changes…")


def render_conversion_tab():
    """Unit conversion tables — mirrors the Excel 'Conversion tables' sheet."""
    st.markdown("#### Unit Conversion Tables")
    st.caption("Reference factors used to normalise data to corporate units. Do not edit.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Energy conversion factors**")
        ef_data = {
            "Energy Type": ["Natural Gas","Propane","LPG","Diesel","Petrol","Fuel Oil","Coal","Biomass","Waste Tires"],
            "Unit": ["GJ LHV"]*9,
            "CO₂ EF (T.CO₂/GJ)": [0.0561,0.0631,0.0561,0.0741,0.0693,0.0774,0.0950,0.0,0.0475],
            "HHV/LHV ratio": [1.095,1.085,1.085,1.06,1.06,1.06,1.02,1.215,1.06],
        }
        st.dataframe(pd.DataFrame(ef_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Unit conversion factors (to corporate unit)**")
        unit_data = {
            "Indicator": ["Production","Production","Water","Energy (electric)","Energy (electric)","Waste","Waste"],
            "From unit": ["kg","lb","m³","MWh","TJ","kg","lb"],
            "To unit": ["metric T","metric T","m³","GJ","GJ","metric T","metric T"],
            "Factor": [0.001,0.000454,1.0,3.6,1000.0,0.001,0.000454],
        }
        st.dataframe(pd.DataFrame(unit_data), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Emission factors source**: WBCSD TIP methodology · IEA country factors (Scope 2)")
    st.markdown("**Reference**: IPCC 2006 Guidelines, IRES 2011, IEA World Energy Statistics")



# ─────────────────────────────────────────────────────────
# PAGE 2 — ANALYSIS
# ─────────────────────────────────────────────────────────
def page_analysis():
    st.markdown("## Analysis & Trends")
    yrs = [str(y) for y in LONG_YEARS]

    # KPI strip
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, unit, delta, pos in [
        (c1,"Total Energy 2023","32.4M","GJ","▼ 0.3% vs 2022",True),
        (c2,"Total CO₂ 2023","2.05M","T.CO₂","▼ 0.5% vs 2022",True),
        (c3,"CO₂ Intensity","0.551","T.CO₂/T","▼ 4.3% vs 2022",True),
        (c4,"Renewable Elec","48.3%","of total elec","▲ 19% vs 2022",True),
        (c5,"Waste Recovery","85.8%","of total waste","▲ 0.9% vs 2022",True),
    ]:
        col.markdown(kpi_card_html(label,val,unit,delta,pos), unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    def mkchart(title, traces, height=280, yformat=None):
        fig = go.Figure()
        for t in traces: fig.add_trace(t)
        fig.update_layout(margin=dict(l=10,r=10,t=36,b=10),height=height,
            title=dict(text=title,font=dict(size=13,color="#374151")),
            legend=dict(font=dict(size=11)),
            plot_bgcolor="#fff",paper_bgcolor="#fff",
            xaxis=dict(gridcolor="#F3F4F6"),
            yaxis=dict(gridcolor="#F3F4F6",tickformat=yformat or ""))
        return fig

    col_l, col_r = st.columns(2)
    with col_l:
        fig = mkchart("Total energy consumption (M GJ)",[
            go.Scatter(x=yrs,y=LONG_DATA["energy"],mode="lines+markers",
                name="Total energy",line=dict(color="#00916E",width=2),
                fill="tozeroy",fillcolor="rgba(0,145,110,.08)",marker=dict(size=4))])
        st.plotly_chart(fig,width="stretch")

    with col_r:
        fig = mkchart("CO₂ emissions — Scope 1 vs Scope 2 (M T.CO₂)",[
            go.Scatter(x=yrs,y=LONG_DATA["scope1"],mode="lines",name="Scope 1",
                line=dict(color="#DC2626",width=2),fill="tozeroy",fillcolor="rgba(220,38,38,.12)"),
            go.Scatter(x=yrs,y=LONG_DATA["scope2"],mode="lines",name="Scope 2",
                line=dict(color="#1D4ED8",width=2),fill="tozeroy",fillcolor="rgba(29,78,216,.12)")])
        st.plotly_chart(fig,width="stretch")

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig = mkchart("Energy & CO₂ intensity KPIs",[
            go.Scatter(x=yrs,y=LONG_DATA["energy_kpi"],mode="lines+markers",
                name="Energy KPI (GJ/T)",line=dict(color="#7C3AED",width=2),marker=dict(size=4),yaxis="y"),
            go.Scatter(x=yrs,y=LONG_DATA["co2_kpi"],mode="lines+markers",
                name="CO₂ KPI (T/T)",line=dict(color="#EA580C",width=2,dash="dot"),marker=dict(size=4),yaxis="y2")])
        fig.update_layout(yaxis2=dict(overlaying="y",side="right",gridcolor="#F3F4F6",
            title="CO₂ KPI",showgrid=False),yaxis_title="Energy KPI")
        st.plotly_chart(fig,width="stretch")

    with col_r2:
        fig = go.Figure()
        colors = {"Natural Gas":"#F59E0B","Electricity":"#3B82F6","Fuel Oil":"#EF4444",
                  "LPG":"#8B5CF6","Coal":"#6B7280","Other":"#D1D5DB"}
        for fuel, vals in FUEL_MIX.items():
            fig.add_trace(go.Bar(x=yrs,y=vals,name=fuel,marker_color=colors.get(fuel,"#ccc"),
                marker_line_width=0))
        fig.update_layout(barmode="stack",title=dict(text="Fuel mix evolution (%)",font=dict(size=13,color="#374151")),
            legend=dict(font=dict(size=10)),height=280,margin=dict(l=10,r=10,t=36,b=10),
            yaxis=dict(range=[0,100],ticksuffix="%",gridcolor="#F3F4F6"),
            plot_bgcolor="#fff",paper_bgcolor="#fff",xaxis=dict(tickangle=-45))
        st.plotly_chart(fig,width="stretch")

    col_l3,col_m3,col_r3 = st.columns(3)
    with col_l3:
        fig = mkchart("Water withdrawals (M m³)",[
            go.Scatter(x=yrs,y=LONG_DATA["water"],mode="lines+markers",
                name="Water",line=dict(color="#0EA5E9",width=2),
                fill="tozeroy",fillcolor="rgba(14,165,233,.08)",marker=dict(size=3))],height=230)
        st.plotly_chart(fig,width="stretch")

    with col_m3:
        fig = go.Figure(go.Bar(x=yrs,y=LONG_DATA["renew_pct"],
            marker_color=["rgba(0,145,110,.9)" if i>=12 else "rgba(0,145,110,.4)" for i in range(15)],
            marker_line_width=0))
        fig.update_layout(title=dict(text="Renewable electricity (%)",font=dict(size=13,color="#374151")),
            height=230,margin=dict(l=10,r=10,t=36,b=10),
            yaxis=dict(range=[0,100],ticksuffix="%",gridcolor="#F3F4F6"),
            plot_bgcolor="#fff",paper_bgcolor="#fff")
        st.plotly_chart(fig,width="stretch")

    with col_r3:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yrs,y=LONG_DATA["waste_recov"],name="Recovery %",
            marker_color="rgba(0,145,110,.7)"))
        fig.add_trace(go.Bar(x=yrs,y=[100-v for v in LONG_DATA["waste_recov"]],
            name="Elimination %",marker_color="rgba(220,38,38,.4)"))
        fig.update_layout(barmode="stack",title=dict(text="Waste recovery rate (%)",font=dict(size=13,color="#374151")),
            height=230,margin=dict(l=10,r=10,t=36,b=10),
            yaxis=dict(range=[0,100],ticksuffix="%",gridcolor="#F3F4F6"),
            plot_bgcolor="#fff",paper_bgcolor="#fff",legend=dict(font=dict(size=10)))
        st.plotly_chart(fig,width="stretch")

# ─────────────────────────────────────────────────────────
# PAGE 3 — BENCHMARKING
# ─────────────────────────────────────────────────────────
def page_benchmarking():
    st.markdown("## Peer Benchmarking")
    src_note = ("📡 Benchmarking data loaded live from SharePoint"
                if st.session_state.get("consolidated_source") == "sharepoint"
                else "📋 Benchmarking uses built-in TIP consolidated dataset (10 companies × 15 years)")
    st.info(src_note + " — Quartile bands derived from all TIP members. No individual competitor figures disclosed.", icon="🔍")

    inp, out = get_current_outputs()

    # ── Load consolidated data (SharePoint or dummy) ──────────────────
    
    bench_df = df.copy()
    bench_df = bench_df[bench_df["year"] == 2023]
    company = st.session_state.user_company

    # ── Compute live quartiles from actual consolidated data ──────────
    def live_bench(row_label, company_value, unit, lower_better):
        kpi_rows = bench_df[
            bench_df["row_label"] == row_label
        ]

        vals = kpi_rows["data"].dropna().values

        if len(vals) >= 4:
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
        else:
            q25, med, q75 = (
                company_value * 0.85,
                company_value,
                company_value * 1.15
            )

        from formula_engine import BenchmarkResult
        return BenchmarkResult(
            row_label,
            company_value,
            float(q25),
            float(med),
            float(q75),
            unit,
            lower_better,
        )

    renew_val = (inp.renew_elec_purchased + inp.self_gen_elec) / max(out.total_electricity, 1) * 100
    benchmarks = [
        live_bench("co2_kpi",        out.co2_kpi,               "T.CO₂/T", True),
        live_bench("energy_kpi",     out.energy_kpi,            "GJ/T", True),
        live_bench("water_kpi",      out.water_kpi,             "m³/T", True),
        live_bench("renewable_pct",  renew_val,                 "%", False),
        live_bench("waste_recovery", out.waste_recovery_pct*100,"%", False),
    ]
    # Use friendly names
    kpi_labels = ["CO₂ intensity","Energy intensity","Water intensity",
                  "Renewable electricity","Waste recovery rate"]
    for b, lbl in zip(benchmarks, kpi_labels):
        b.kpi_name = lbl

    col_l, col_r = st.columns(2)

    with col_l:
        with st.container(border=True):
            st.markdown("#### Industry band positioning — 2023")
            st.caption("Your KPI vs TIP member quartile ranges. Bands recalculate as new companies are added.")
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
                bands_html += band_html(b.kpi_name, b.company_value,
                                        b.q25, b.median, b.q75,
                                        b.unit, b.lower_is_better)
            bands_html += "</div>"
            st.markdown(bands_html, unsafe_allow_html=True)

    with col_r:
        with st.container(border=True):
            st.markdown("#### ESG profile — vs TIP industry average")
            st.caption("Normalised score 0–100. Six key ESG dimensions.")
            dims = ["CO₂ intensity","Energy efficiency","Water management",
                    "Waste recovery","Renewable energy","H&S performance"]
            # Live company scores derived from quartile positions
            company_scores = []
            for b in benchmarks[:5]:
                rng = max(b.q75 - b.q25, 0.001)
                raw = (b.company_value - b.q25) / rng
                company_scores.append(max(0, min(100, (1-raw)*100 if b.lower_is_better else raw*100)))
            company_scores += [85]   # H&S — not in KPI set, use representative value
            industry_scores = [65, 70, 65, 74, 52, 74]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=company_scores+[company_scores[0]],
                theta=dims+[dims[0]],fill="toself",name="Bridgestone 2023",
                line=dict(color="#00916E",width=2),
                fillcolor="rgba(0,145,110,.12)"))
            fig.add_trace(go.Scatterpolar(r=industry_scores+[industry_scores[0]],
                theta=dims+[dims[0]],fill="toself",name="TIP average",
                line=dict(color="#9CA3AF",width=1.5,dash="dot"),
                fillcolor="rgba(156,163,175,.08)"))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,100],tickfont=dict(size=9))),
                showlegend=True,height=340,margin=dict(l=40,r=40,t=20,b=20),
                legend=dict(font=dict(size=11)))
            st.plotly_chart(fig,width="stretch")

    with st.container(border=True):
        st.markdown("#### Improvement rate — your company vs TIP industry average since 2009")
        improve_data = {
            "KPI": ["CO₂ intensity","Energy intensity","Water intensity",
                    "Renewable electricity","Waste recovery rate"],
            "Your improvement": ["▼ 35.2%","▼ 12.1%","▼ 26.4%","▲ +48pp","▲ +4pp"],
            "Industry average":  ["▼ 22.8%","▼ 15.3%","▼ 19.1%","▲ +18pp","▲ +6pp"],
            "Lead vs peers":     ["+12.4pp ✅","−3.2pp ⚠️","+7.3pp ✅","+30pp ✅","−2pp ⚠️"],
            "Status":            ["Ahead","Lagging","Ahead","Ahead","Lagging"],
        }
        df_imp = pd.DataFrame(improve_data)
        st.dataframe(df_imp.style.apply(lambda row: [
            "background:#ECFDF5;color:#065F46" if row["Status"]=="Ahead"
            else "background:#FFFBEB;color:#92400E"]*len(row), axis=1),
            hide_index=True, width="stretch")

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        with st.container(border=True):
            st.markdown("#### ✅ Key strengths")
            st.success("**CO₂ intensity** at 0.551 T/T — top quartile of TIP members for 2nd consecutive year. 35% improvement since 2009, outpacing industry by 12pp.")
            st.success("**Renewable electricity** at 48.3% — highest in 2023 TIP cohort, driven by procurement agreements and on-site generation.")
    with col_r2:
        with st.container(border=True):
            st.markdown("#### ⚠️ Improvement areas")
            st.warning("**Energy intensity** at 8.7 GJ/T lags industry improvement rate by 3.2pp. Opportunity: steam system optimisation, waste heat recovery.")
            st.warning("**Waste recovery** at 85.8% is below top-quartile threshold (~88%). Industrial composting and circular material partnerships are common levers.")

# ─────────────────────────────────────────────────────────
# PAGE 4 — VERIFICATION (dss only)
# ─────────────────────────────────────────────────────────
def page_verification():
    if not st.session_state.is_dss:
        st.error("🔒 This section is restricted to dss+ analysts and managers.")
        return
    st.markdown("## Data Verification")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Company","Bridgestone")
    c2.metric("Year","2023")
    c3.metric("Status","Pending review")
    c4.metric("Completeness","87%")
    c5.metric("Open flags","2 warnings · 1 error")

    st.divider()
    flags_def = [
        ("warn","Fuel Oil — >20% YoY decrease (−62.2%)",
         "Bridgestone 2023 vs 2022: Fuel Oil dropped from 166,773 GJ to 150,000 GJ. Comment: 'Transition to natural gas.' Plausible — verify against procurement records.","flag1"),
        ("warn","Renewable electricity — >20% YoY increase (+27.3%)",
         "Renewable electricity rose from 4,082,923 GJ to 5,200,000 GJ. No comment provided. Requires explanation from company.","flag2"),
        ("error","Waste consistency check FAILED",
         "Total waste (338,000 MT) ≠ Recovery (290,000) + Elimination. Discrepancy of 2,000 MT. Must be corrected before acceptance.","flag3"),
        ("ok","Energy totals — all values consistent",
         "Total energy (32.4M GJ) reconciles with all sub-category inputs. Unit conversions verified. All YoY changes within ±20%.","flag4"),
        ("ok","CO₂ calculations verified",
         "Scope 1 matches emission factor outputs. Scope 2 consistent with IEA country factors. Total CO₂ 2.05M T.CO₂ internally consistent.","flag5"),
    ]
    for severity, title, detail, flag_id in flags_def:
        icon_map = {"warn":"!","error":"✕","ok":"✓"}
        color_map = {"warn":"fc-warn fi-warn","error":"fc-error fi-error","ok":"fc-ok fi-ok"}
        fc, fi = color_map[severity].split()
        resolved = flag_id in st.session_state.flags_resolved
        if resolved:
            fc, fi = "fc-ok","fi-ok"
            title += " ✓ Approved"
        st.markdown(f"""<div class="flag-card {fc}">
          <div class="fc-icon {fi}">{icon_map.get("ok" if resolved else severity,"")}</div>
          <div><div class="fc-title">{title}</div><div class="fc-detail">{detail}</div></div>
        </div>""", unsafe_allow_html=True)
        if not resolved and severity in ("warn","error"):
            cols = st.columns([6,1,1])
            with cols[1]:
                if st.button("Query", key=f"q_{flag_id}"):
                    st.toast(f"Query sent to Bridgestone contact for: {title[:40]}...", icon="📧")
            with cols[2]:
                if severity == "warn" and st.button("Accept ✓", key=f"a_{flag_id}", type="primary"):
                    st.session_state.flags_resolved.add(flag_id)
                    st.rerun()
                elif severity == "error" and st.button("Send Back", key=f"sb_{flag_id}"):
                    st.toast("Submission returned to Bridgestone with error details.", icon="↩️")

    st.divider()
    col_approve, _ = st.columns([1,3])
    with col_approve:
        if st.button("✓ Approve All Warnings", type="primary"):
            st.session_state.flags_resolved.update({"flag1","flag2"})
            st.rerun()

# ─────────────────────────────────────────────────────────
# PAGE 5 — AI READINESS (dss only)
# ─────────────────────────────────────────────────────────
def page_readiness():
    if not st.session_state.is_dss:
        st.error("🔒 This section is restricted to dss+ analysts and managers.")
        return
    st.markdown("## AI Readiness Check")

    col_score, col_info = st.columns([1,3])
    with col_score:
        fig = go.Figure(go.Indicator(mode="gauge+number",value=82,
            number=dict(suffix="/100",font=dict(size=32,color="#00916E")),
            gauge=dict(axis=dict(range=[0,100]),bar=dict(color="#00916E",thickness=.25),
                steps=[dict(range=[0,60],color="#FEE2E2"),
                       dict(range=[60,80],color="#FEF3C7"),
                       dict(range=[80,100],color="#D1FAE5")],
                threshold=dict(line=dict(color="#065F46",width=3),thickness=.75,value=82))))
        fig.update_layout(height=200,margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch")

    with col_info:
        st.markdown("### Report Readiness Score: **82 / 100**")
        st.caption("Bridgestone · 2023 Reporting Year · 3 items pending resolution")
        st.warning("Review required before submission to consolidated database")

    st.divider()
    st.markdown("#### Data completeness by section")
    sections = [
        ("ISO 14001",100,"#00916E"),("Production",100,"#00916E"),
        ("Water",100,"#00916E"),("Energy — Electricity",100,"#00916E"),
        ("Energy — Fuels",95,"#00916E"),("CO₂ Scope 1",100,"#00916E"),
        ("CO₂ Scope 2",85,"#D97706"),("Waste",88,"#D97706"),
        ("Pathway 3 (SBTi / Water)",60,"#DC2626"),
        ("Pathway 4 (H&S)",55,"#DC2626"),("Pathway 4 (D&I)",50,"#DC2626"),
        ("Electricity by country",80,"#D97706"),
    ]
    cols = st.columns(3)
    for i, (label, pct, color) in enumerate(sections):
        with cols[i%3]:
            with st.container(border=True):
                st.caption(label)
                st.progress(pct/100, text=f"{pct}%")

    st.divider()
    st.markdown("#### AI-generated insights *(for analyst review — not final output)*")
    for title, body in [
        ("Energy & Emissions Summary",
         "Bridgestone's **total energy consumption** in 2023 (32.4M GJ) remained broadly stable year-on-year (−0.3%), masking a significant structural shift: **fuel oil usage fell 10%**, continuing the multi-year transition toward natural gas and renewables. **Renewable electricity** reached 48.3% of total electricity — a 19% improvement on 2022 and the highest reported in the 2023 TIP cohort. **CO₂ intensity** declined to 0.551 T/T, placing Bridgestone in the **top quartile** for the second consecutive year."),
        ("Data Gaps & Actions Required",
         "**3 items must be resolved before submission is accepted:**\n\n1. **Waste consistency error** — Total waste does not reconcile with recovery + elimination sub-totals. Discrepancy of 2,000 MT. Return to client for correction.\n\n2. **Pathway 4 incomplete** — Lost-time injury rate, total recordable injury rate, female representation at workforce and board levels are missing. Mandatory for 2023 TIP report.\n\n3. **Scope 3 emissions not reported** — Bridgestone has committed to SBTi net-zero. Category 11 (rolling resistance use-phase) should be flagged for voluntary disclosure."),
    ]:
        st.markdown(f"""<div class="ai-card">
          <div class="ai-head"><div class="ai-pulse"></div>
            <span class="ai-title">{title}</span>
            <span class="ai-badge">AI insight · review before use</span>
          </div>
          <div class="ai-body">{body.replace(chr(10),'<br>')}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# MAIN
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