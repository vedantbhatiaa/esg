"""
config.py — TIP ESG Platform · Central Configuration
======================================================
Single source of truth for ALL file paths, year bounds, and tuneable
constants. Nothing is hardcoded anywhere else — every module imports from here.

To override any value, set it in .streamlit/secrets.toml or as an env var.

Example secrets.toml additions:
    DATA_YEAR_START   = 2009
    DATA_YEAR_END     = 2023
    MASTER_CSV_NAME   = "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"
    OLLAMA_URL        = "http://localhost:11434"
    OLLAMA_MODEL      = "phi3"
    LOG_RETENTION_DAYS = 8
    DSS_EMAIL_DOMAIN  = "@consultdss.com"
"""

from __future__ import annotations
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets → env var → default (in that priority)."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


# ── Root paths ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
DATA_DIR     = BASE_DIR / "data_storage"

# ── Master data ───────────────────────────────────────────────────────────────
MASTER_CSV_NAME  = _secret("MASTER_CSV_NAME",
                            "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv")
MASTER_XLSX_NAME = _secret("MASTER_XLSX_NAME",
                            "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.xlsx")
CONSOLIDATED_XLSX_NAME = _secret("CONSOLIDATED_XLSX_NAME",
                                  "CONSOLIDATED_DUMMY_2009_2023.xlsx")
SECTOR_CSV_NAME  = _secret("SECTOR_CSV_NAME",
                            "ESG_SECTOR_AGGREGATED_2009_2023.csv")
TIP_MEMBERS_CSV_NAME = _secret("TIP_MEMBERS_CSV_NAME",
                                "ESG_MASTER_WIDE_TIP_MEMBERS_2009_2023.csv")

MASTER_DIR   = DATA_DIR / "master"
RAW_DIR      = DATA_DIR / "raw"
MEMBERS_DIR  = DATA_DIR / "members" / "TIP"
VERSIONS_DIR = DATA_DIR / "versions"
LOGS_DIR     = DATA_DIR / "chat_logs"
REPORTS_DIR  = DATA_DIR / "reports" / "TIP"
VALIDATED_DIR = DATA_DIR / "validated"

MASTER_CSV   = MASTER_DIR / MASTER_CSV_NAME
RAW_CSV      = RAW_DIR    / MASTER_CSV_NAME
TIP_MASTER_CSV = MEMBERS_DIR / TIP_MEMBERS_CSV_NAME

# ── Year bounds — derived dynamically from real data; hardcoded fallback ──────
DATA_YEAR_START: int = int(_secret("DATA_YEAR_START", "2009"))
DATA_YEAR_END:   int = int(_secret("DATA_YEAR_END",   "2023"))


def refresh_year_bounds(df=None) -> tuple[int, int]:
    """
    Compute (start_year, end_year) from the loaded DataFrame.
    Call this once after data is loaded; updates the module globals
    DATA_YEAR_START and DATA_YEAR_END so HIST_YEARS / LONG_YEARS / CURR_YEAR
    are accurate.
    Returns (start, end).
    """
    global DATA_YEAR_START, DATA_YEAR_END
    if df is not None and not df.empty and "Year" in df.columns:
        try:
            years = sorted(df["Year"].dropna().astype(int).unique())
            if years:
                DATA_YEAR_START = years[0]
                DATA_YEAR_END   = years[-1]
                logger.info("[config] Year bounds from data: %d–%d",
                            DATA_YEAR_START, DATA_YEAR_END)
                return DATA_YEAR_START, DATA_YEAR_END
        except Exception as e:
            logger.warning("[config] Could not derive year bounds from data: %s", e)
    return DATA_YEAR_START, DATA_YEAR_END


def hist_years() -> list[int]:
    """Historical years list (excludes current/reporting year)."""
    return list(range(DATA_YEAR_START, DATA_YEAR_END))


def long_years() -> list[int]:
    """Full years list including reporting year."""
    return list(range(DATA_YEAR_START, DATA_YEAR_END + 1))


def curr_year() -> int:
    """The most recent data year (reporting year)."""
    return DATA_YEAR_END


# ── Authentication ─────────────────────────────────────────────────────────────
DSS_EMAIL_DOMAIN = _secret("DSS_EMAIL_DOMAIN", "@consultdss.com")

def load_clients() -> dict[str, str]:
    """
    Load email → company mapping.
    Production: provide CLIENTS_JSON in secrets.toml:
        CLIENTS_JSON = '{"co@firm.com": "Company Name", ...}'
    Testing: falls back to the built-in demo dict below.
    """
    import json
    raw = _secret("CLIENTS_JSON", "")
    if raw:
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning("[config] CLIENTS_JSON parse error: %s", e)
    # Demo fallback (testing only — replace with CLIENTS_JSON in production)
    return {
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


# ── LLM / AI ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL      = _secret("OLLAMA_URL",      "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = _secret("OLLAMA_MODEL",    "phi3")
HF_DEFAULT_MODEL     = _secret("HF_MODEL",        "mistralai/Mistral-7B-Instruct-v0.3")

OLLAMA_NUM_PREDICT: int = int(_secret("OLLAMA_NUM_PREDICT", "500"))
OLLAMA_NUM_CTX:     int = int(_secret("OLLAMA_NUM_CTX",     "2048"))
OLLAMA_TEMPERATURE: float = float(_secret("OLLAMA_TEMPERATURE", "0.3"))
OLLAMA_TIMEOUT:     int = int(_secret("OLLAMA_TIMEOUT", "300"))

AZURE_OPENAI_ENDPOINT   = _secret("AZURE_OPENAI_ENDPOINT",   "https://YOUR-RESOURCE.openai.azure.com")
AZURE_OPENAI_KEY        = _secret("AZURE_OPENAI_KEY",        "")
AZURE_OPENAI_DEPLOYMENT = _secret("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ── Chatbot / logging ─────────────────────────────────────────────────────────
LOG_RETENTION_DAYS: int = int(_secret("LOG_RETENTION_DAYS", "8"))
FILELOCK_TIMEOUT:   int = int(_secret("FILELOCK_TIMEOUT",   "10"))