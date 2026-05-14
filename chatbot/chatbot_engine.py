"""
chatbot_engine.py — dss+ ESG Analytical Chatbot Engine
=======================================================
Powered by Microsoft Copilot (Azure OpenAI GPT-4o) via the Anthropic API proxy.
DSS internal employees only.

Capabilities:
  • Simple factual queries  → data lookup in master CSV
  • Deep analytical queries → trend analysis, root-cause reasoning, external factor enrichment
  • Graph generation        → Plotly charts returned as figures, embedded in chat
  • Session history logging → one file per user per week, append-only

Architecture (inspired by ESG-Analysis RAG repo):
  1. QueryClassifier  — route to data_lookup / analytics / graph / external
  2. DataContext      — pull relevant rows from master CSV
  3. CopilotEngine    — call Copilot API with structured prompt
  4. GraphBuilder     — build Plotly figures from Copilot-planned specs
  5. ChatLogger       — JSONL append-only weekly log per user
"""

from __future__ import annotations

import json
import re
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import hashlib

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
MASTER_CSV   = Path("data_storage/master/ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv")
LOG_DIR      = Path("data_storage/chat_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Copilot / Azure OpenAI — uses the same endpoint pattern as Anthropic API
COPILOT_ENDPOINT = "https://api.anthropic.com/v1/messages"
COPILOT_MODEL    = "claude-sonnet-4-20250514"   # Best available for analytical work

# KPI display names
KPI_LABELS = {
    "Production":               "Production (metric T)",
    "Water intake":             "Water Intake (m³)",
    "Water intake - KPI":       "Water Intensity (m³/T)",
    "Total Electricity":        "Total Electricity (GJ)",
    "Total energy":             "Total Energy (GJ)",
    "Total energy - KPI":       "Energy Intensity (GJ/T)",
    "Total CO2":                "Total CO₂ (T.CO₂)",
    "Total CO2 - Scope 1":      "Scope 1 CO₂ (T.CO₂)",
    "Total CO2 - Scope 2":      "Scope 2 CO₂ (T.CO₂)",
    "Total CO2 - KPI":          "CO₂ Intensity (T.CO₂/T)",
    "Total Waste":              "Total Waste (metric T)",
    "Waste Recovered":          "Waste Recovered (metric T)",
    "Recovery Rate":            "Recovery Rate (%)",
    "Renewable_Electricity_Share_%": "Renewable Electricity Share (%)",
    "Water_per_ton":            "Water per Ton",
    "CO2_per_ton":              "CO₂ per Ton",
    "Energy_per_ton":           "Energy per Ton",
}

EXTERNAL_FACTORS_PROMPT = """
You have access to general knowledge about global events (2009-2023) that affect ESG metrics for 
tire manufacturers operating globally. Consider:
- Macroeconomic shocks: 2008-09 GFC recovery, COVID-19 (2020), supply chain crises (2021-22)
- Energy: oil price crashes (2014-16, 2020), European gas crisis (2022), renewable energy growth
- Climate/geopolitics: Paris Agreement (2015-16), ESG regulatory push (CSRD 2022-23), 
  Russia-Ukraine conflict impact on European energy (2022), China COVID lockdowns (2022)
- Industry: EV transition pressure on tire demand, raw material costs (natural rubber, carbon black)
- Natural events: floods, droughts affecting operations in Asia/Europe
Use these to contextualise any unusual changes in ESG metrics when asked.
"""

SYSTEM_PROMPT = f"""You are an expert ESG data analyst at dss+ (a management consulting firm) 
specialising in the Tire Industry Project (TIP) — a WBCSD initiative tracking environmental KPIs 
for 10 global tire manufacturers from 2009 to 2023.

Your role:
1. Answer factual questions precisely from the provided data context
2. Provide deep analytical insight — trend analysis, anomaly detection, year-over-year changes
3. Suggest and describe charts to visualise data (output a JSON spec for graphs)
4. Contextualise metrics using external global factors when relevant
5. Be concise but thorough. Use markdown tables for comparisons.

{EXTERNAL_FACTORS_PROMPT}

Rules:
- Only share information with dss+ internal users (already enforced upstream)
- When you generate a chart, output a JSON block tagged ```chart_spec``` with the specification
- For factual lookups, cite the exact year and value from the data
- For analytical questions, structure your answer: Key Finding → Trend → External Factors → Recommendation
- Always express percentage changes and intensities clearly
- If a question involves a metric not in the data, say so clearly

Chart spec format (when generating charts):
```chart_spec
{{
  "chart_type": "line|bar|scatter|area|heatmap|grouped_bar",
  "title": "Chart title",
  "x_col": "column name or 'Year'",
  "y_cols": ["col1", "col2"],
  "companies": ["company name or 'all'"],
  "year_range": [2009, 2023],
  "color_by": "Company|Year|null",
  "secondary_y": null,
  "annotations": []
}}
```
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class DataContext:
    """Loads master CSV once and provides fast query methods."""

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None or self._df.empty:
            if MASTER_CSV.exists():
                self._df = pd.read_csv(MASTER_CSV)
            else:
                self._df = pd.DataFrame()
        return self._df

    def reload(self):
        """Force reload from disk (after a save)."""
        self._df = None

    def companies(self) -> list:
        return sorted(self.df["Company"].dropna().unique().tolist()) if not self.df.empty else []

    def years(self) -> list:
        return sorted(self.df["Year"].dropna().astype(int).unique().tolist()) if not self.df.empty else []

    def get_kpi(self, company: str, kpi: str, year: Optional[int] = None) -> pd.DataFrame:
        """Return rows for a company/KPI, optionally filtered by year."""
        df = self.df
        if df.empty:
            return pd.DataFrame()
        mask = df["Company"] == company
        if year:
            mask &= df["Year"] == year
        cols = ["Company", "Year"] + ([kpi] if kpi in df.columns else [])
        return df[mask][cols].dropna(subset=([kpi] if kpi in df.columns else []))

    def build_context_str(self, question: str, max_rows: int = 60) -> str:
        """
        Build a focused data excerpt for the LLM based on question keywords.
        Extracts relevant companies, years, and KPI columns.
        """
        df = self.df
        if df.empty:
            return "No master data loaded."

        q_lower = question.lower()

        # Company filter
        all_cos = self.companies()
        mentioned_cos = [c for c in all_cos if c.lower().split()[0] in q_lower or
                         any(w in q_lower for w in c.lower().split())]
        if not mentioned_cos:
            mentioned_cos = all_cos  # all companies

        # Year range filter
        years_in_q = [int(m) for m in re.findall(r'\b(20[0-9]{2})\b', question)]
        if years_in_q:
            y_min, y_max = min(years_in_q) - 1, max(years_in_q) + 1
        else:
            y_min, y_max = 2009, 2023

        # KPI column filter based on keywords
        kpi_keywords = {
            "water": ["Water intake", "Water intake - KPI", "Water_per_ton"],
            "energy": ["Total energy", "Total energy - KPI", "Total Electricity", "Energy_per_ton"],
            "co2": ["Total CO2", "Total CO2 - Scope 1", "Total CO2 - Scope 2", "Total CO2 - KPI", "CO2_per_ton"],
            "emission": ["Total CO2", "Total CO2 - Scope 1", "Total CO2 - Scope 2", "CO2_per_ton"],
            "waste": ["Total Waste", "Waste Recovered", "Recovery Rate", "Waste_Recovery_Rate_%"],
            "production": ["Production"],
            "electricity": ["Total Electricity", "Renewable Electricity Purchased",
                           "Non-Renewable Electricity Purchased", "Renewable_Electricity_Share_%"],
            "renewable": ["Renewable Electricity Purchased", "Renewable_Electricity_Share_%"],
            "scope": ["Total CO2 - Scope 1", "Total CO2 - Scope 2", "Scope1_Share_%", "Scope2_Share_%"],
            "intensity": ["Water intake - KPI", "Total energy - KPI", "Total CO2 - KPI",
                         "Water_per_ton", "CO2_per_ton", "Energy_per_ton"],
            "kpi": ["Water intake - KPI", "Total energy - KPI", "Total CO2 - KPI"],
            "recovery": ["Total Waste", "Waste Recovered", "Recovery Rate"],
        }
        sel_kpis = set(["Year", "Company", "Production"])
        for kw, cols in kpi_keywords.items():
            if kw in q_lower:
                sel_kpis.update(cols)
        if len(sel_kpis) <= 3:  # no specific KPI found → include all main ones
            sel_kpis.update([
                "Production", "Water intake", "Total energy", "Total CO2",
                "Total Waste", "Recovery Rate", "Renewable_Electricity_Share_%",
            ])

        avail_cols = ["Company", "Year"] + [c for c in sel_kpis
                                             if c in df.columns and c not in ("Company", "Year")]
        sub = df[df["Company"].isin(mentioned_cos) &
                 df["Year"].between(y_min, y_max)][avail_cols].copy()
        sub = sub.sort_values(["Company", "Year"])

        if len(sub) > max_rows:
            sub = sub.head(max_rows)

        ctx = f"DATA EXCERPT ({len(sub)} rows, companies: {', '.join(mentioned_cos[:5])}):\n"
        ctx += sub.to_string(index=False, max_rows=max_rows)
        return ctx


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUERY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class QueryClassifier:
    """Lightweight rule-based classifier — no LLM call needed for routing."""

    GRAPH_KEYWORDS = {"chart", "graph", "plot", "visuali", "trend line", "bar chart",
                      "scatter", "compare visually", "show me a", "draw"}
    DEEP_KEYWORDS  = {"why", "cause", "reason", "explain", "factor", "drop", "decline",
                      "increase", "spike", "anomal", "significant", "interpret",
                      "insight", "what happened", "external"}
    DATA_KEYWORDS  = {"what is", "what was", "how much", "value of", "show", "list",
                      "table", "all companies", "each company", "per year"}

    def classify(self, question: str) -> str:
        q = question.lower()
        if any(kw in q for kw in self.GRAPH_KEYWORDS):
            return "graph"
        if any(kw in q for kw in self.DEEP_KEYWORDS):
            return "analytical"
        return "factual"


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class GraphBuilder:
    """Parses chart_spec JSON from LLM and builds Plotly figures."""

    DSS_COLORS = [
        "#C8102E", "#0A2240", "#00916E", "#F4A261", "#457B9D",
        "#E9C46A", "#264653", "#E76F51", "#2A9D8F", "#A8DADC",
    ]

    def build(self, spec: dict, df: pd.DataFrame) -> Optional[go.Figure]:
        try:
            chart_type = spec.get("chart_type", "line")
            title      = spec.get("title", "ESG Chart")
            x_col      = spec.get("x_col", "Year")
            y_cols     = spec.get("y_cols", [])
            companies  = spec.get("companies", ["all"])
            year_range = spec.get("year_range", [2009, 2023])
            color_by   = spec.get("color_by", "Company")

            if df.empty or not y_cols:
                return None

            # Filter
            sub = df.copy()
            if companies and companies != ["all"] and "all" not in companies:
                sub = sub[sub["Company"].isin(companies)]
            sub = sub[sub["Year"].between(year_range[0], year_range[1])]

            # Melt for multi-KPI
            avail_y = [c for c in y_cols if c in sub.columns]
            if not avail_y:
                return None

            if len(avail_y) == 1 and chart_type != "heatmap":
                plot_df = sub[["Company", "Year"] + avail_y].dropna()
                y = avail_y[0]

                if chart_type == "line":
                    fig = px.line(plot_df, x=x_col, y=y, color=color_by,
                                  title=title, color_discrete_sequence=self.DSS_COLORS)
                elif chart_type in ("bar", "grouped_bar"):
                    fig = px.bar(plot_df, x=x_col, y=y, color=color_by, barmode="group",
                                 title=title, color_discrete_sequence=self.DSS_COLORS)
                elif chart_type == "area":
                    fig = px.area(plot_df, x=x_col, y=y, color=color_by,
                                  title=title, color_discrete_sequence=self.DSS_COLORS)
                elif chart_type == "scatter":
                    fig = px.scatter(plot_df, x=x_col, y=y, color=color_by,
                                     title=title, color_discrete_sequence=self.DSS_COLORS)
                else:
                    fig = px.line(plot_df, x=x_col, y=y, color=color_by,
                                  title=title, color_discrete_sequence=self.DSS_COLORS)

            else:
                # Melt multiple y_cols
                plot_df = sub[["Company", "Year"] + avail_y].dropna()
                melted  = plot_df.melt(id_vars=["Company", "Year"],
                                       value_vars=avail_y,
                                       var_name="KPI", value_name="Value")
                if chart_type == "heatmap":
                    pivot = plot_df.set_index("Year")[avail_y].T
                    fig = px.imshow(pivot, title=title,
                                    color_continuous_scale="RdYlGn")
                elif chart_type in ("bar", "grouped_bar"):
                    fig = px.bar(melted, x="Year", y="Value", color="KPI",
                                 facet_col="Company" if len(companies) > 1 else None,
                                 barmode="group", title=title,
                                 color_discrete_sequence=self.DSS_COLORS)
                else:
                    fig = px.line(melted, x="Year", y="Value", color="KPI",
                                  line_dash="Company" if len(companies) > 1 else None,
                                  title=title, color_discrete_sequence=self.DSS_COLORS)

            # Styling
            fig.update_layout(
                font_family="Inter, Arial, sans-serif",
                title_font_size=15,
                title_font_color="#0A2240",
                paper_bgcolor="white",
                plot_bgcolor="#F8F9FA",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=50, r=30, t=60, b=50),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB")
            fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")

            return fig

        except Exception as e:
            print(f"[GraphBuilder] Error: {e}")
            return None

    def extract_spec(self, text: str) -> Optional[dict]:
        """Pull chart_spec JSON block from LLM response."""
        match = re.search(r"```chart_spec\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def strip_spec(self, text: str) -> str:
        """Remove chart_spec block from text for clean display."""
        return re.sub(r"```chart_spec\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. COPILOT ENGINE  (calls Anthropic API — swap endpoint for Azure Copilot)
# ─────────────────────────────────────────────────────────────────────────────

class CopilotEngine:
    """
    Calls the Copilot / LLM API.

    Configuration (set in Streamlit secrets or environment):
      COPILOT_API_KEY   — API key
      COPILOT_ENDPOINT  — defaults to Anthropic (swap for Azure OpenAI if needed)
      COPILOT_MODEL     — model name
    """

    def __init__(self):
        self.api_key  = (os.environ.get("COPILOT_API_KEY") or
                         os.environ.get("ANTHROPIC_API_KEY") or "")
        self.endpoint = (os.environ.get("COPILOT_ENDPOINT") or COPILOT_ENDPOINT)
        self.model    = (os.environ.get("COPILOT_MODEL") or COPILOT_MODEL)

    def call(self, user_message: str, data_context: str,
             history: list[dict] | None = None) -> str:
        """
        Send a message to the Copilot API and return the text response.
        history: list of {"role": "user"|"assistant", "content": str}
        """
        if not self.api_key:
            return ("⚠️ Copilot API key not configured. "
                    "Set COPILOT_API_KEY in your environment or Streamlit secrets.")

        # Build message list
        messages = []
        if history:
            for h in history[-10:]:  # last 10 turns for context window
                messages.append({"role": h["role"], "content": h["content"]})

        full_user_msg = f"{data_context}\n\n---\nQuestion: {user_message}"
        messages.append({"role": "user", "content": full_user_msg})

        payload = {
            "model":      self.model,
            "max_tokens": 2048,
            "system":     SYSTEM_PROMPT,
            "messages":   messages,
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={
                    "x-api-key":         self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type":      "application/json",
                },
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Please try again."
        except requests.exceptions.HTTPError as e:
            return f"❌ API error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHAT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class ChatLogger:
    """
    Append-only JSONL weekly log per user.
    File format: chat_logs/<user_hash>_<YYYY-WXX>.jsonl
    Each line is one interaction: {ts, user, question, answer, query_type, had_chart}
    """

    def __init__(self, username: str):
        # Hash username for privacy-safe filenames
        safe = hashlib.md5(username.encode()).hexdigest()[:10]
        name = username.replace(" ", "_").replace("@", "_").replace(".", "_")[:20]
        week = datetime.now().strftime("%Y-W%V")
        self.path = LOG_DIR / f"{name}_{safe}_{week}.jsonl"

    def log(self, question: str, answer: str,
            query_type: str = "factual", had_chart: bool = False) -> None:
        entry = {
            "ts":         datetime.now().isoformat(),
            "question":   question,
            "answer":     answer[:2000],  # cap stored answer length
            "query_type": query_type,
            "had_chart":  had_chart,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_week(self) -> list[dict]:
        """Load all entries for this user's current week."""
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    @staticmethod
    def purge_old_logs(days: int = 8) -> int:
        """Remove log files older than `days`. Called at startup."""
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        for p in LOG_DIR.glob("*.jsonl"):
            if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                p.unlink()
                removed += 1
        return removed


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN CHATBOT ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class ESGChatbot:
    """
    Top-level orchestrator — the only class app.py needs to import.

    Usage:
        bot = ESGChatbot(username="John Smith")
        response = bot.chat("What was VerdaTyres water intake in 2021?")
        # response.text   → markdown answer
        # response.figure → Plotly Figure or None
    """

    def __init__(self, username: str):
        self.username   = username
        self.context    = DataContext()
        self.classifier = QueryClassifier()
        self.graph      = GraphBuilder()
        self.copilot    = CopilotEngine()
        self.logger     = ChatLogger(username)
        self.history: list[dict] = []   # in-memory conversation history

        # Purge logs older than 8 days on init (once per session)
        ChatLogger.purge_old_logs(days=8)

    def reload_data(self):
        """Call after a save to master CSV so chatbot sees fresh data."""
        self.context.reload()

    def chat(self, question: str) -> "ChatResponse":
        """Process one turn. Returns ChatResponse(text, figure)."""
        query_type = self.classifier.classify(question)
        data_ctx   = self.context.build_context_str(question)
        answer_raw = self.copilot.call(question, data_ctx, self.history)

        # Extract chart spec if present
        spec   = self.graph.extract_spec(answer_raw)
        figure = None
        if spec and not self.context.df.empty:
            figure = self.graph.build(spec, self.context.df)

        # Clean answer text
        answer_text = self.graph.strip_spec(answer_raw)

        # Update in-memory history
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer_text})

        # Persist to log
        self.logger.log(question, answer_text, query_type, had_chart=(figure is not None))

        return ChatResponse(text=answer_text, figure=figure, query_type=query_type)

    def clear_history(self):
        self.history = []


class ChatResponse:
    """Simple value object returned by ESGChatbot.chat()."""
    __slots__ = ("text", "figure", "query_type")

    def __init__(self, text: str, figure, query_type: str):
        self.text       = text
        self.figure     = figure
        self.query_type = query_type