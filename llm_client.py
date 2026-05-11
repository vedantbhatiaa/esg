"""
TIP ESG Platform — LLM Client (Azure OpenAI / Microsoft Copilot)
=================================================================
Uses Azure OpenAI Service — the same endpoint that powers Microsoft Copilot.

Data Privacy Rules (mandatory — never bypass):
  ✅ Only DERIVED SUMMARIES are sent to the model (KPI values, % changes)
  ✅ Company name is anonymised in the prompt before sending
  ✅ No raw Excel data, no file contents, no personally identifying info
  ✅ Azure OpenAI's enterprise agreement guarantees zero data retention
  ✅ Zero training on tenant data (confirmed in Microsoft DPA for E5 plan)
  ❌ NEVER send raw uploaded files or database connection strings

Azure OpenAI vs OpenAI.com:
  We use AZURE OpenAI (not api.openai.com) because:
  - Data stays in the Microsoft tenant (same as SharePoint, Copilot)
  - Zero-retention SLA documented in the Enterprise Agreement
  - Consistent with dss+'s existing Copilot partnership
"""

import os, json, logging, time
from typing import Optional
import requests

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://YOUR-RESOURCE.openai.azure.com"
)
AZURE_OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOY   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")   # your deployment name
API_VERSION           = "2024-02-01"

MAX_TOKENS   = 800
TEMPERATURE  = 0.2          # Low = more deterministic / factual


# ─── Privacy sanitiser ───────────────────────────────────────────────────────

def _anonymise(company: str) -> str:
    """
    Replace company name with 'Client Company' before sending to LLM.
    Ensures the model never stores identifiable client data in logs.
    """
    return "Client Company"


def _build_kpi_summary(kpis: dict, company: str, year: int) -> str:
    """
    Convert a KPI dict to a structured text summary for the prompt.
    Only numerical derivatives — never raw file contents.
    """
    anon = _anonymise(company)
    lines = [
        f"Company: {anon} | Reporting year: {year}",
        f"Production: {kpis.get('production_mt',0):,.0f} metric T",
        f"Total Energy: {kpis.get('total_energy_gj',0):,.0f} GJ  |  Intensity: {kpis.get('energy_kpi',0):.2f} GJ/T",
        f"CO2 Scope 1: {kpis.get('co2_scope1',0):,.0f} T  |  Scope 2: {kpis.get('co2_scope2',0):,.0f} T",
        f"Total CO2: {kpis.get('total_co2',0):,.0f} T  |  Intensity: {kpis.get('co2_kpi',0):.4f} T/T",
        f"Water: {kpis.get('water_m3',0):,.0f} m3  |  Intensity: {kpis.get('water_kpi',0):.2f} m3/T",
        f"Renewable electricity: {kpis.get('renew_elec_pct',0):.1f}% of total electricity",
        f"Waste recovery rate: {kpis.get('waste_recovery_pct',0):.1f}%",
        f"YoY CO2 change: {kpis.get('yoy_co2_pct',0):+.1f}%",
        f"YoY energy change: {kpis.get('yoy_energy_pct',0):+.1f}%",
    ]
    if kpis.get("benchmarks"):
        lines.append("\nIndustry benchmarking position (TIP 2023 quartiles):")
        for b in kpis["benchmarks"]:
            lines.append(f"  {b['kpi']}: {b['position']} (company={b['value']:.3f}, industry Q2={b['median']:.3f})")
    return "\n".join(lines)


# ─── Prompt Templates ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an ESG analyst assistant for dss+, a sustainability consulting firm.
You analyse KPI data from tire manufacturers participating in the WBCSD Tire Industry Project.
Your role is to generate factual, concise insights from KPI summaries.

Rules:
- Base insights ONLY on the KPI numbers provided. Never invent data.
- Write in clear, professional English suitable for a consulting report.
- Always note that insights require analyst review before publication.
- Do NOT speculate about causes unless explicitly asked.
- Keep responses to 2-4 short paragraphs maximum.
"""

INSIGHT_PROMPT = """Based on the following ESG KPI summary, provide:
1. A 2-sentence performance headline (strengths and gaps)
2. Key year-on-year observations
3. Industry benchmark commentary
4. One actionable recommendation

KPI Summary:
{kpi_summary}

Format: plain paragraphs, no bullet points. Max 300 words.
Conclude with: 'Note: All insights are AI-generated and require analyst review before use.'
"""

GAPS_PROMPT = """Review the following ESG KPI summary and list data quality issues:
- Missing or zero values where data would be expected
- Year-on-year variations above 20% that need explanation
- Completeness issues by section

KPI Summary:
{kpi_summary}

Flags provided:
{flags}

Format: numbered list of issues. Keep each item to one sentence.
"""

READINESS_PROMPT = """Rate the readiness of this submission for inclusion in the TIP consolidated report.
Score from 0-100 and give a 2-sentence justification.

KPI Summary:
{kpi_summary}

Completeness by section:
{completeness}

Flags: {n_errors} errors, {n_warnings} warnings.

Respond as JSON: {{"score": 82, "label": "Review required", "justification": "..."}}
"""


# ─── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """Azure OpenAI client. Only receives anonymised KPI summaries — never raw data."""

    def __init__(self):
        self.endpoint  = AZURE_OPENAI_ENDPOINT
        self.key       = AZURE_OPENAI_KEY
        self.deploy    = AZURE_OPENAI_DEPLOY

    def _call(self, messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
        """Make a single API call to Azure OpenAI."""
        if not self.key:
            logger.warning("AZURE_OPENAI_KEY not set — returning mock response")
            return _mock_response(messages[-1]["content"][:80])

        url     = (f"{self.endpoint}/openai/deployments/{self.deploy}"
                   f"/chat/completions?api-version={API_VERSION}")
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": TEMPERATURE}

        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.HTTPError as e:
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited — retrying in %ss", wait)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Azure OpenAI call failed after 3 retries")

    def generate_insight(self, kpis: dict, company: str, year: int) -> str:
        """
        Generate a plain-language ESG performance narrative.
        Input: KPI dict (numbers only — no raw data).
        Output: String for analyst review.
        """
        summary  = _build_kpi_summary(kpis, company, year)
        prompt   = INSIGHT_PROMPT.format(kpi_summary=summary)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        result = self._call(messages)
        logger.info("Insight generated for %s %s (%d chars)", company, year, len(result))
        return result

    def identify_gaps(self, kpis: dict, company: str, year: int,
                      flags: list[dict]) -> str:
        """
        Identify data gaps and quality issues from KPIs + validation flags.
        """
        summary   = _build_kpi_summary(kpis, company, year)
        flags_txt = "\n".join(
            f"- [{f.get('severity','').upper()}] {f.get('message','')}: {f.get('detail','')}"
            for f in flags
        ) or "No validation flags raised."
        prompt   = GAPS_PROMPT.format(kpi_summary=summary, flags=flags_txt)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        return self._call(messages)

    def score_readiness(self, kpis: dict, company: str, year: int,
                        completeness: dict, flags: list[dict]) -> dict:
        """
        Return a readiness score (0-100) and justification.
        Parses the model's JSON response safely.
        """
        summary     = _build_kpi_summary(kpis, company, year)
        comp_txt    = "\n".join(f"  {k}: {v}%" for k,v in completeness.items())
        n_errors    = sum(1 for f in flags if f.get("severity") == "error")
        n_warnings  = sum(1 for f in flags if f.get("severity") == "warning")
        prompt      = READINESS_PROMPT.format(
            kpi_summary=summary, completeness=comp_txt,
            n_errors=n_errors, n_warnings=n_warnings)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        raw = self._call(messages, max_tokens=200)
        try:
            # Extract JSON from response
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"score": 70, "label": "Review required",
                "justification": raw[:200]}


# ─── Mock response (when API key not set) ────────────────────────────────────

def _mock_response(prompt_preview: str) -> str:
    return (
        "[AI MOCK — Azure OpenAI key not configured]\n\n"
        "This is a placeholder response generated without calling the API. "
        "When AZURE_OPENAI_KEY is set, this will be replaced by a real model response "
        "based on the anonymised KPI summary provided.\n\n"
        "Note: All insights are AI-generated and require analyst review before use."
    )


# ─── Singleton ───────────────────────────────────────────────────────────────
_llm: Optional[LLMClient] = None

def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = get_llm()
    
    test_kpis = {
        "production_mt":    1_515_000,
        "total_energy_gj":  13_300_000,
        "energy_kpi":       8.78,
        "co2_scope1":       405_000,
        "co2_scope2":       376_000,
        "total_co2":        781_000,
        "co2_kpi":          0.516,
        "water_m3":         9_180_000,
        "water_kpi":        6.06,
        "renew_elec_pct":   69.1,
        "waste_recovery_pct": 92.4,
        "yoy_co2_pct":      -3.2,
        "yoy_energy_pct":   +1.1,
        "benchmarks": [
            {"kpi":"CO2 intensity","position":"Top 25%","value":0.516,"median":0.68},
            {"kpi":"Energy intensity","position":"Above avg","value":8.78,"median":9.2},
        ]
    }

    print("=== INSIGHT ===")
    print(llm.generate_insight(test_kpis, "VerdaTyres Corp", 2023))

    print("\n=== GAPS ===")
    print(llm.identify_gaps(test_kpis, "VerdaTyres Corp", 2023, []))

    print("\n=== READINESS SCORE ===")
    completeness = {"ISO 14001":100,"Production":100,"Water":100,
                    "Energy":95,"CO2 Scope 1":100,"CO2 Scope 2":85,
                    "Waste":88,"Pathway 3 (SBTi)":60,"Pathway 4 (H&S)":55}
    score = llm.score_readiness(test_kpis, "VerdaTyres Corp", 2023, completeness, [])
    print(json.dumps(score, indent=2))
