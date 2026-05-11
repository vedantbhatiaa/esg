"""
TIP ESG Platform — Formula Engine
===================================
Pure Python implementation of all Excel template calculations.
No UI dependencies. Can be used by Streamlit, FastAPI, or any other frontend.

Formula source: WBCSD TIP KPI Collection Tool methodology
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd

# ─────────────────────────────────────────────────────────
# CONSTANTS  (TIP standard methodology, same as Excel EFs)
# ─────────────────────────────────────────────────────────

EMISSION_FACTORS = {
    # Source: TIP Conversion Tables tab  (T.CO2 per GJ LHV)
    "nat_gas":           56.1,
    "coal_sub":          95.0,   # Sub-bituminous
    "coal_brown":        95.0,   # Brown coal briquettes
    "coal_other":        95.0,   # Other bituminous
    "propane":           63.1,
    "fuel_oil_heavy_a":  77.4,
    "fuel_oil_heavy_c":  77.4,
    "diesel":            74.1,
    "petrol":            69.3,
    "biomass":            0.0,   # Biogenic CO2 excluded per GHG Protocol
    "waste_tires":       47.5,   # per GJ (after heat value conversion)
    "lpg":               56.1,
    "other_fuels":       71.9,
}

WASTE_TIRE_HEAT_VALUE = 36.23   # GJ per metric T of waste tires
GJ_TO_MWH = 1 / 3.6             # 1 GJ = 0.2778 MWh

# Default Scope 2 electricity emission factor (location-based, IEA global avg)
# Production value uses country-specific factors from Electricity data input tab
DEFAULT_ELEC_EF = 0.45          # T.CO2 per MWh

# ─────────────────────────────────────────────────────────
# INPUT DATA MODEL
# ─────────────────────────────────────────────────────────

@dataclass
class TemplateInputs:
    """
    All company-provided raw input fields for one reporting year.
    These are the ~20 fields the company actually fills in the Excel template.
    All other values are calculated.
    """
    company:    str = ""
    year:       int = 2023

    # ISO 14001
    total_sites: float = 0.0
    iso_sites:   float = 0.0

    # Production
    production:  float = 0.0     # metric T

    # Water
    water_withdrawals: float = 0.0   # m3

    # Energy — Electricity
    renew_elec_purchased:  float = 0.0   # GJ
    nonrenew_elec_purchased: float = 0.0  # GJ
    self_gen_elec:         float = 0.0   # GJ
    purchased_steam:       float = 0.0   # GJ
    sold_electricity:      float = 0.0   # GJ (negative contribution)
    sold_steam:            float = 0.0   # GJ (negative contribution)

    # Energy — Fuels (GJ LHV unless noted)
    nat_gas:           float = 0.0
    coal_sub:          float = 0.0
    coal_brown:        float = 0.0
    coal_other:        float = 0.0
    propane:           float = 0.0
    fuel_oil_heavy_a:  float = 0.0
    fuel_oil_heavy_c:  float = 0.0
    diesel:            float = 0.0
    petrol:            float = 0.0
    biomass:           float = 0.0
    waste_tires_mt:    float = 0.0   # metric T (converted internally to GJ)
    lpg:               float = 0.0
    other_fuels:       float = 0.0

    # CO2 — only Scope 2 steam is company-provided; rest is calculated
    co2_scope2_steam: float = 0.0   # T.CO2
    # co2_scope2_electricity is derived from Electricity data input tab
    co2_scope2_electricity: float = 0.0   # T.CO2 (calculated from country EFs)

    # Waste
    waste_total:    float = 0.0   # metric T
    waste_recovery: float = 0.0   # metric T

    def validate_inputs(self) -> list[str]:
        """Basic sanity checks before calculation."""
        errors = []
        if self.total_sites <= 0:
            errors.append("Total sites must be > 0")
        if self.iso_sites > self.total_sites:
            errors.append("ISO certified sites cannot exceed total sites")
        if self.production <= 0:
            errors.append("Production must be > 0")
        if self.waste_recovery > self.waste_total:
            errors.append("Waste to recovery cannot exceed total waste")
        return errors


# ─────────────────────────────────────────────────────────
# OUTPUT DATA MODEL
# ─────────────────────────────────────────────────────────

@dataclass
class TemplateOutputs:
    """
    All calculated fields derived from TemplateInputs.
    These mirror the formula cells in the Excel template.
    """
    inputs: TemplateInputs = field(default_factory=TemplateInputs)

    # ISO 14001
    pct_certified: float = 0.0

    # Water
    water_kpi: float = 0.0      # m3 / metric T

    # Energy — Electricity subtotals
    total_electricity: float = 0.0   # GJ
    waste_tires_gj:    float = 0.0   # GJ LHV (converted from metric T)

    # Energy — Totals
    total_energy: float = 0.0        # GJ LHV
    energy_kpi:   float = 0.0        # GJ LHV / metric T

    # CO2 — Scope 1 breakdown
    co2_nat_gas:    float = 0.0
    co2_coal:       float = 0.0
    co2_propane:    float = 0.0
    co2_fuel_oil:   float = 0.0
    co2_diesel:     float = 0.0
    co2_petrol:     float = 0.0
    co2_biomass:    float = 0.0
    co2_waste_tires: float = 0.0
    co2_lpg:        float = 0.0
    co2_other:      float = 0.0
    total_co2_scope1: float = 0.0

    # CO2 — Scope 2 total
    total_co2_scope2: float = 0.0

    # CO2 — Totals
    total_co2: float = 0.0
    co2_kpi:   float = 0.0           # T.CO2 / metric T

    # Waste
    waste_elimination:    float = 0.0
    waste_recovery_pct:   float = 0.0
    waste_elimination_pct: float = 0.0

    # Consistency flags (True = OK)
    check_waste:       bool = True
    check_electricity: bool = True
    check_iso:         bool = True


# ─────────────────────────────────────────────────────────
# CORE CALCULATION FUNCTION
# ─────────────────────────────────────────────────────────

def calculate(inputs: TemplateInputs) -> TemplateOutputs:
    """
    Run all formula calculations on a set of company inputs.
    Mirrors the Excel template formula structure exactly.

    Args:
        inputs: TemplateInputs dataclass with all company-provided values

    Returns:
        TemplateOutputs dataclass with all derived values
    """
    out = TemplateOutputs(inputs=inputs)
    d   = inputs   # shorthand

    # ── ISO 14001 ───────────────────────────────────────
    out.pct_certified = safe_div(d.iso_sites, d.total_sites)

    # ── Water ───────────────────────────────────────────
    out.water_kpi = safe_div(d.water_withdrawals, d.production)

    # ── Energy — Electricity ────────────────────────────
    out.total_electricity = (
        d.renew_elec_purchased +
        d.nonrenew_elec_purchased +
        d.self_gen_elec
    )

    # Waste tires: metric T → GJ LHV
    out.waste_tires_gj = d.waste_tires_mt * WASTE_TIRE_HEAT_VALUE

    # ── Energy — Total ──────────────────────────────────
    coal_total_gj = d.coal_sub + d.coal_brown + d.coal_other
    fuel_oil_total_gj = d.fuel_oil_heavy_a + d.fuel_oil_heavy_c

    out.total_energy = (
        out.total_electricity
        + d.purchased_steam
        + d.nat_gas
        + coal_total_gj
        + d.propane
        + fuel_oil_total_gj
        + d.diesel
        + d.petrol
        + d.biomass
        + out.waste_tires_gj
        + d.lpg
        + d.other_fuels
        - d.sold_electricity
        - d.sold_steam
    )
    out.energy_kpi = safe_div(out.total_energy, d.production)

    # ── CO2 — Scope 1 (fuel combustion) ────────────────
    out.co2_nat_gas    = d.nat_gas       * EMISSION_FACTORS["nat_gas"]           / 1000
    out.co2_coal       = coal_total_gj   * EMISSION_FACTORS["coal_sub"]          / 1000
    out.co2_propane    = d.propane       * EMISSION_FACTORS["propane"]           / 1000
    out.co2_fuel_oil   = fuel_oil_total_gj * EMISSION_FACTORS["fuel_oil_heavy_a"] / 1000
    out.co2_diesel     = d.diesel        * EMISSION_FACTORS["diesel"]            / 1000
    out.co2_petrol     = d.petrol        * EMISSION_FACTORS["petrol"]            / 1000
    out.co2_biomass    = 0.0             # biogenic excluded
    out.co2_waste_tires = out.waste_tires_gj * EMISSION_FACTORS["waste_tires"]  / 1000
    out.co2_lpg        = d.lpg          * EMISSION_FACTORS["lpg"]               / 1000
    out.co2_other      = d.other_fuels  * EMISSION_FACTORS["other_fuels"]       / 1000

    out.total_co2_scope1 = (
        out.co2_nat_gas + out.co2_coal + out.co2_propane +
        out.co2_fuel_oil + out.co2_diesel + out.co2_petrol +
        out.co2_biomass + out.co2_waste_tires + out.co2_lpg + out.co2_other
    )

    # ── CO2 — Scope 2 ───────────────────────────────────
    # If country-specific electricity EF not provided, use default
    if d.co2_scope2_electricity == 0 and d.nonrenew_elec_purchased > 0:
        # Fallback: convert GJ to MWh, apply default EF
        nonrenew_mwh = d.nonrenew_elec_purchased * GJ_TO_MWH
        d.co2_scope2_electricity = nonrenew_mwh * DEFAULT_ELEC_EF

    out.total_co2_scope2 = d.co2_scope2_steam + d.co2_scope2_electricity

    # ── CO2 — Totals ────────────────────────────────────
    out.total_co2 = out.total_co2_scope1 + out.total_co2_scope2
    out.co2_kpi   = safe_div(out.total_co2, d.production)

    # ── Waste ───────────────────────────────────────────
    out.waste_elimination     = d.waste_total - d.waste_recovery
    out.waste_recovery_pct    = safe_div(d.waste_recovery, d.waste_total)
    out.waste_elimination_pct = 1.0 - out.waste_recovery_pct

    # ── Consistency checks ──────────────────────────────
    out.check_waste = abs(
        d.waste_total - d.waste_recovery - out.waste_elimination
    ) < 1.0

    out.check_electricity = abs(
        out.total_electricity
        - d.renew_elec_purchased
        - d.nonrenew_elec_purchased
        - d.self_gen_elec
    ) < 1.0

    out.check_iso = d.iso_sites <= d.total_sites

    return out


# ─────────────────────────────────────────────────────────
# YEAR-OVER-YEAR CALCULATIONS
# ─────────────────────────────────────────────────────────

def yoy_change(current: float, previous: float) -> Optional[float]:
    """Return % change. None if previous is 0 or None."""
    if not previous or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def yoy_flag(pct_change: Optional[float], threshold: float = 20.0) -> str:
    """Return 'OK', 'WARN' (>threshold%), or 'N/A'."""
    if pct_change is None:
        return "N/A"
    return "WARN" if abs(pct_change) > threshold else "OK"


# ─────────────────────────────────────────────────────────
# VALIDATION ENGINE
# ─────────────────────────────────────────────────────────

@dataclass
class ValidationFlag:
    field:       str
    severity:    str          # 'error' | 'warning' | 'info'
    message:     str
    detail:      str = ""
    yoy_pct:     Optional[float] = None
    auto_pass:   bool = False  # can be auto-approved


def validate_submission(
    current: TemplateOutputs,
    previous: Optional[TemplateOutputs] = None,
    yoy_threshold: float = 20.0
) -> list[ValidationFlag]:
    """
    Run all validation rules on a submission.
    Returns a list of ValidationFlag objects sorted by severity.

    Rules:
    1. Consistency checks (errors if failed)
    2. Year-over-year variation > threshold (warnings)
    3. Missing or zero values in key fields (warnings)
    """
    flags: list[ValidationFlag] = []
    d  = current.inputs
    o  = current

    # ── Consistency errors ──────────────────────────────
    if not o.check_waste:
        flags.append(ValidationFlag(
            field="waste_total",
            severity="error",
            message="Waste consistency check FAILED",
            detail=f"Total waste ({fmt_num(d.waste_total)} MT) ≠ Recovery ({fmt_num(d.waste_recovery)}) + Elimination ({fmt_num(o.waste_elimination)}). "
                   f"Difference: {fmt_num(abs(d.waste_total - d.waste_recovery - o.waste_elimination))} MT. Must be corrected."
        ))

    if not o.check_iso:
        flags.append(ValidationFlag(
            field="iso_sites",
            severity="error",
            message="ISO 14001 sites exceeds total sites",
            detail=f"ISO sites ({d.iso_sites}) > Total sites ({d.total_sites}). Data entry error."
        ))

    # ── YoY variation warnings ──────────────────────────
    if previous:
        p = previous.inputs
        po = previous

        yoy_checks = [
            ("production",         d.production,         p.production,         "Production"),
            ("water_withdrawals",  d.water_withdrawals,  p.water_withdrawals,  "Water withdrawals"),
            ("total_energy",       o.total_energy,       po.total_energy,      "Total energy"),
            ("total_co2",          o.total_co2,          po.total_co2,         "Total CO₂"),
            ("nat_gas",            d.nat_gas,            p.nat_gas,            "Natural gas"),
            ("fuel_oil",           d.fuel_oil_heavy_a + d.fuel_oil_heavy_c,
                                   p.fuel_oil_heavy_a + p.fuel_oil_heavy_c,    "Fuel oil"),
            ("renew_elec",         d.renew_elec_purchased, p.renew_elec_purchased, "Renewable electricity"),
            ("waste_total",        d.waste_total,        p.waste_total,        "Total waste"),
        ]

        for key, curr_val, prev_val, label in yoy_checks:
            pct = yoy_change(curr_val, prev_val)
            if pct is not None and abs(pct) > yoy_threshold:
                flags.append(ValidationFlag(
                    field=key,
                    severity="warning",
                    message=f"{label} — >{yoy_threshold:.0f}% YoY change ({pct:+.1f}%)",
                    detail=f"Current: {fmt_num(curr_val)} · Previous year: {fmt_num(prev_val)}. "
                           f"Please provide a comment explaining this variation.",
                    yoy_pct=pct,
                    auto_pass=True
                ))

    # ── Missing value warnings ──────────────────────────
    if d.co2_scope2_steam == 0:
        flags.append(ValidationFlag(
            field="co2_scope2_steam",
            severity="warning",
            message="Scope 2 steam CO₂ not provided",
            detail="Defaulted to 0. Please confirm whether purchased steam is used and provide CO₂ value.",
            auto_pass=True
        ))

    if d.waste_total == 0:
        flags.append(ValidationFlag(
            field="waste_total",
            severity="warning",
            message="Total waste not reported",
            detail="Waste data is mandatory for TIP reporting from 2022 onwards.",
        ))

    # Sort: errors first, then warnings
    flags.sort(key=lambda f: 0 if f.severity == "error" else 1)
    return flags


# ─────────────────────────────────────────────────────────
# BENCHMARKING ENGINE
# ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    kpi_name:      str
    company_value: float
    q25:           float    # top 25% threshold (best practice boundary)
    median:        float
    q75:           float    # bottom 25% boundary
    unit:          str
    lower_is_better: bool = True

    @property
    def quartile(self) -> str:
        if self.lower_is_better:
            if self.company_value <= self.q25:
                return "top25"
            elif self.company_value <= self.median:
                return "mid_upper"
            elif self.company_value <= self.q75:
                return "mid_lower"
            else:
                return "bottom25"
        else:
            if self.company_value >= self.q75:
                return "top25"
            elif self.company_value >= self.median:
                return "mid_upper"
            elif self.company_value >= self.q25:
                return "mid_lower"
            else:
                return "bottom25"

    @property
    def quartile_label(self) -> str:
        return {
            "top25": "Top 25%",
            "mid_upper": "Above average",
            "mid_lower": "Average",
            "bottom25": "Below average"
        }.get(self.quartile, "—")

    @property
    def position_pct(self) -> float:
        """0–100 position on the band track (for rendering)."""
        rng = self.q75 - self.q25
        if rng == 0:
            return 50.0
        pos = (self.company_value - self.q25) / rng * 50 + 25
        if not self.lower_is_better:
            pos = 100 - pos
        return max(2, min(98, pos))


def get_benchmarks(outputs: TemplateOutputs) -> list[BenchmarkResult]:
    """
    Compare a company's KPIs against TIP industry quartile bands.
    Quartile ranges sourced from TIP 2023 published aggregate report.
    Individual company data is never exposed.
    """
    o = outputs
    return [
        BenchmarkResult("CO₂ intensity",       o.co2_kpi,      0.55, 0.68, 0.82, "T.CO₂/T",    lower_is_better=True),
        BenchmarkResult("Energy intensity",     o.energy_kpi,   8.0,  9.2,  10.5, "GJ/T",        lower_is_better=True),
        BenchmarkResult("Water intensity",      o.water_kpi,    5.5,  7.0,  9.0,  "m³/T",        lower_is_better=True),
        BenchmarkResult("Renewable elec %",     safe_div(o.inputs.renew_elec_purchased + o.inputs.self_gen_elec, o.total_electricity) * 100,
                                                40.0, 20.0, 10.0, "%", lower_is_better=False),
        BenchmarkResult("Waste recovery %",     o.waste_recovery_pct * 100,
                                                86.0, 80.0, 74.0, "%", lower_is_better=False),
    ]


# ─────────────────────────────────────────────────────────
# TEMPLATE TABLE BUILDER (for display in any frontend)
# ─────────────────────────────────────────────────────────

def build_template_dataframe(
    historical: list[tuple[int, TemplateInputs, TemplateOutputs]],
    current_inputs:  TemplateInputs,
    current_outputs: TemplateOutputs,
    previous_outputs: Optional[TemplateOutputs] = None
) -> pd.DataFrame:
    """
    Build a pandas DataFrame matching the Excel template layout.
    Columns: Indicator, Unit, Type, [hist years...], [curr year], YoY%
    Type column: 'input' | 'calc' | 'section'
    """
    all_years = [(yr, inp, out) for yr, inp, out in historical]
    curr_yr = current_inputs.year

    def row(label, unit, row_type, vals_fn, calc_fn=None, bold=False):
        hist_vals = {yr: vals_fn(inp, out) for yr, inp, out in all_years}
        curr_val  = (calc_fn or vals_fn)(current_inputs, current_outputs)
        prev_val  = vals_fn(*all_years[-1][1:]) if all_years else None
        yoy = yoy_change(float(curr_val or 0), float(prev_val or 0)) if prev_val else None
        return {
            "Indicator": label,
            "Unit": unit,
            "Type": row_type,
            "Bold": bold,
            **hist_vals,
            curr_yr: curr_val,
            "YoY %": f"{yoy:+.1f}%" if yoy is not None else "—"
        }

    def section(label):
        return {"Indicator": label, "Unit": "", "Type": "section", "Bold": True,
                **{yr: "" for yr, _, _ in all_years}, curr_yr: "", "YoY %": ""}

    rows = [
        section("ISO 14001"),
        row("Total no. of sites",             "no.",       "input", lambda i,o: i.total_sites),
        row("ISO 14001 certified sites",      "no.",       "input", lambda i,o: i.iso_sites),
        row("% certified sites",              "%",         "calc",  lambda i,o: f"{o.pct_certified*100:.1f}%"),
        section("Production"),
        row("Production",                     "metric T",  "input", lambda i,o: i.production),
        section("Water"),
        row("Water withdrawals",              "m³",        "input", lambda i,o: i.water_withdrawals),
        row("Water intensity KPI",            "m³/T",      "calc",  lambda i,o: f"{o.water_kpi:.2f}"),
        section("Energy"),
        row("Total Electricity",              "GJ",        "calc",  lambda i,o: o.total_electricity),
        row("— Renewable electricity",        "GJ",        "input", lambda i,o: i.renew_elec_purchased),
        row("— Non-renewable electricity",    "GJ",        "input", lambda i,o: i.nonrenew_elec_purchased),
        row("— Self-generated renewable",     "GJ",        "input", lambda i,o: i.self_gen_elec),
        row("Purchased Steam",                "GJ",        "input", lambda i,o: i.purchased_steam),
        row("Sold Electricity",               "GJ",        "input", lambda i,o: i.sold_electricity),
        row("Natural Gas",                    "GJ LHV",    "input", lambda i,o: i.nat_gas),
        row("Coal (all types)",               "GJ LHV",    "input", lambda i,o: i.coal_sub+i.coal_brown+i.coal_other),
        row("Propane",                        "GJ LHV",    "input", lambda i,o: i.propane),
        row("Fuel Oil",                       "GJ LHV",    "input", lambda i,o: i.fuel_oil_heavy_a+i.fuel_oil_heavy_c),
        row("Diesel",                         "GJ LHV",    "input", lambda i,o: i.diesel),
        row("LPG",                            "GJ LHV",    "input", lambda i,o: i.lpg),
        row("Total Energy",                   "GJ LHV",    "calc",  lambda i,o: o.total_energy, bold=True),
        row("Energy intensity KPI",           "GJ/T",      "calc",  lambda i,o: f"{o.energy_kpi:.2f}"),
        section("CO₂ Emissions"),
        row("Scope 2 – Steam",                "T.CO₂",     "input", lambda i,o: i.co2_scope2_steam),
        row("CO₂ – Natural Gas",              "T.CO₂",     "calc",  lambda i,o: o.co2_nat_gas),
        row("CO₂ – Coal",                     "T.CO₂",     "calc",  lambda i,o: o.co2_coal),
        row("CO₂ – Fuel Oil + Diesel",        "T.CO₂",     "calc",  lambda i,o: o.co2_fuel_oil+o.co2_diesel),
        row("CO₂ – LPG",                      "T.CO₂",     "calc",  lambda i,o: o.co2_lpg),
        row("Total CO₂ Scope 1",              "T.CO₂",     "calc",  lambda i,o: o.total_co2_scope1, bold=True),
        row("Total CO₂ Scope 2",              "T.CO₂",     "calc",  lambda i,o: o.total_co2_scope2),
        row("Total CO₂ (S1 + S2)",            "T.CO₂",     "calc",  lambda i,o: o.total_co2, bold=True),
        row("CO₂ intensity KPI",              "T.CO₂/T",   "calc",  lambda i,o: f"{o.co2_kpi:.3f}"),
        section("Waste"),
        row("Total waste generated",          "metric T",  "input", lambda i,o: i.waste_total),
        row("Waste sent to recovery",         "metric T",  "input", lambda i,o: i.waste_recovery),
        row("Waste sent to elimination",      "metric T",  "calc",  lambda i,o: o.waste_elimination),
        row("Waste recovery rate",            "%",         "calc",  lambda i,o: f"{o.waste_recovery_pct*100:.1f}%"),
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b and b != 0 else default


def fmt_num(n: float, decimals: int = 0) -> str:
    if n is None:
        return "—"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.0f}K"
    return f"{n:.{decimals}f}"


def inputs_to_dict(inp: TemplateInputs) -> dict:
    return asdict(inp)


def outputs_to_dict(out: TemplateOutputs) -> dict:
    d = asdict(out)
    d.pop("inputs", None)   # avoid nesting
    return d


# ─────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_input = TemplateInputs(
        company="Bridgestone", year=2023,
        total_sites=54, iso_sites=54,
        production=3_720_000,
        water_withdrawals=21_500_000,
        renew_elec_purchased=5_200_000,
        nonrenew_elec_purchased=8_500_000,
        self_gen_elec=45_000,
        purchased_steam=1_050_000,
        nat_gas=16_100_000,
        coal_sub=380_000,
        fuel_oil_heavy_a=150_000,
        diesel=190_000,
        lpg=1_350_000,
        co2_scope2_steam=60_000,
        waste_total=338_000,
        waste_recovery=290_000,
    )

    errors = test_input.validate_inputs()
    if errors:
        print("Input validation errors:", errors)
    else:
        result = calculate(test_input)
        print(f"Total Energy  : {fmt_num(result.total_energy)} GJ")
        print(f"Energy KPI    : {result.energy_kpi:.2f} GJ/T")
        print(f"Total CO₂     : {fmt_num(result.total_co2)} T.CO₂")
        print(f"CO₂ KPI       : {result.co2_kpi:.3f} T.CO₂/T")
        print(f"Water KPI     : {result.water_kpi:.2f} m³/T")
        print(f"Recovery rate : {result.waste_recovery_pct*100:.1f}%")
        print(f"Waste OK      : {result.check_waste}")

        flags = validate_submission(result)
        print(f"\nValidation flags: {len(flags)}")
        for f in flags:
            print(f"  [{f.severity.upper()}] {f.message}")

        benchmarks = get_benchmarks(result)
        print(f"\nBenchmarking:")
        for b in benchmarks:
            print(f"  {b.kpi_name:<25} {b.company_value:.3f} {b.unit:<10} → {b.quartile_label}")
