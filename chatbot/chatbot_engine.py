"""
chatbot_engine.py  —  dss+ ESG Analytical Chatbot Engine
=========================================================
Local-first: Ollama → HuggingFace → Azure

Key design for local 7B models (Mistral/LLaMA):
  - Context is PRE-AGGREGATED into a short summary, never raw CSV rows
  - Max ~800 tokens sent to model (fits 4K context window comfortably)
  - System prompt is short and direct
  - Chart specs are simple JSON the model can reliably produce
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import logging

from chatbot.llm_local import LocalLLMEngine
from chatbot.web_search import ESGWebSearch
import config as cfg

logger = logging.getLogger(__name__)

MASTER_CSV = cfg.MASTER_CSV   # single source of truth — no hardcoded paths
LOG_DIR    = cfg.LOGS_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _build_system_prompt(companies: list[str], year_start: int, year_end: int) -> str:
    """
    Build the LLM system prompt dynamically from real data.
    Company names and year bounds are never hardcoded — they come from the
    loaded DataFrame so the prompt is always accurate after data updates.
    """
    co_list = ", ".join(companies) if companies else "TIP member companies"
    return (
        f"You are an ESG data analyst at dss+ consulting, analysing tire manufacturer "
        f"KPIs ({year_start}–{year_end}) for the WBCSD Tire Industry Project (TIP).\n\n"
        f"Companies: {co_list}.\n\n"
        "When asked to create a chart, output ONLY this JSON block (replace values):\n"
        "```chart_spec\n"
        '{"chart_type":"line","title":"Chart title","x_col":"Year",'
        '"y_cols":["Total CO2"],"companies":["all"],'
        f'"year_range":[{year_start},{year_end}],"color_by":"Company"}}\n'
        "```\n"
        "chart_type: line, bar, grouped_bar, area, scatter\n"
        "y_cols must be exact column names from the data summary provided.\n\n"
        "For analysis questions: Key Finding → Trend → External Factors → Recommendation.\n"
        "Be concise. Use numbers from the data provided."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA CONTEXT  — produces SHORT summaries, not raw row dumps
# ─────────────────────────────────────────────────────────────────────────────
class DataContext:
    def __init__(self):
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if MASTER_CSV.exists():
                self._df = pd.read_csv(MASTER_CSV)
            else:
                self._df = pd.DataFrame()
        return self._df

    def reload(self):
        self._df = None

    def companies(self) -> list:
        return sorted(self.df["Company"].dropna().unique().tolist()) if not self.df.empty else []

    def years(self) -> list:
        return sorted(self.df["Year"].dropna().astype(int).unique().tolist()) if not self.df.empty else []

    # KPI keyword → column name mapping
    KW_COLS = {
        "water":      ["Water intake", "Water intake - KPI"],
        "energy":     ["Total energy", "Total energy - KPI"],
        "co2":        ["Total CO2", "Total CO2 - Scope 1", "Total CO2 - Scope 2", "Total CO2 - KPI"],
        "emission":   ["Total CO2", "Total CO2 - KPI"],
        "carbon":     ["Total CO2", "Total CO2 - KPI"],
        "scope":      ["Total CO2 - Scope 1", "Total CO2 - Scope 2"],
        "waste":      ["Total Waste", "Waste Recovered", "Recovery Rate"],
        "recovery":   ["Total Waste", "Waste Recovered", "Recovery Rate"],
        "electricit": ["Total Electricity", "Renewable_Electricity_Share_%"],
        "renewable":  ["Renewable Electricity Purchased", "Renewable_Electricity_Share_%"],
        "intensity":  ["Water intake - KPI", "Total energy - KPI", "Total CO2 - KPI"],
        "production": ["Production"],
    }

    DEFAULT_COLS = ["Production", "Water intake", "Total energy", "Total CO2", "Total Waste"]

    def _pick_cols(self, question: str) -> list:
        q = question.lower()
        cols = set()
        for kw, c in self.KW_COLS.items():
            if kw in q:
                cols.update(c)
        if not cols:
            cols.update(self.DEFAULT_COLS)
        return [c for c in cols if c in self.df.columns]

    def _pick_companies(self, question: str) -> list:
        q = question.lower()
        all_cos = self.companies()
        mentioned = [c for c in all_cos
                     if any(w in q for w in c.lower().split() if len(w) > 3)]
        return mentioned if mentioned else all_cos

    def _pick_years(self, question: str) -> tuple:
        found = [int(m) for m in re.findall(r'\b(20[0-9]{2})\b', question)]
        if found:
            return max(2009, min(found) - 1), min(2023, max(found) + 1)
        return 2009, 2023

    def build_context_str(self, question: str) -> str:
        """
        Build a SHORT, pre-aggregated data summary.
        For local 7B models: aim for < 600 tokens total context.
        Strategy:
          - Single year asked → table of all companies for that year
          - Trend asked → pivot: companies as rows, years as cols (5 years max)
          - All companies asked → aggregated stats (min/mean/max per KPI)
        """
        df = self.df
        if df.empty:
            return "No ESG data loaded."

        q          = question.lower()
        kpi_cols   = self._pick_cols(question)
        companies  = self._pick_companies(question)
        y_min, y_max = self._pick_years(question)

        if not kpi_cols:
            kpi_cols = self.DEFAULT_COLS[:2]

        sub = df[df["Company"].isin(companies) &
                 df["Year"].between(y_min, y_max)].copy()

        if sub.empty:
            return f"No data found for {companies} between {y_min}-{y_max}."

        # ── Case 1: Single specific year → compact company comparison table ───
        specific_years = [int(m) for m in re.findall(r'\b(20[0-9]{2})\b', question)]
        if specific_years:
            yr = specific_years[-1]
            yr_df = sub[sub["Year"] == yr][["Company"] + kpi_cols].dropna(how="all")
            if not yr_df.empty:
                yr_df = yr_df.set_index("Company")
                # Round for readability
                for c in yr_df.columns:
                    yr_df[c] = yr_df[c].apply(
                        lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A")
                lines = [f"ESG data for {yr} ({', '.join(kpi_cols)}):"]
                lines.append(yr_df.to_string())
                return "\n".join(lines)

        # ── Case 2: Trend / range → pivot with max 6 years ───────────────────
        years_range = sorted(sub["Year"].unique().astype(int).tolist())
        # Pick evenly spaced years if too many
        if len(years_range) > 6:
            step = max(1, len(years_range) // 6)
            years_range = years_range[::step]
            if years_range[-1] != sorted(sub["Year"].unique().astype(int).tolist())[-1]:
                years_range.append(sorted(sub["Year"].unique().astype(int).tolist())[-1])

        # One KPI at a time for trend (pick the most relevant)
        kpi = kpi_cols[0]
        pivot = (sub[sub["Year"].isin(years_range)][["Company", "Year", kpi]]
                 .dropna()
                 .pivot_table(index="Company", columns="Year", values=kpi, aggfunc="first"))
        pivot = pivot.round(1)

        # Shorten company names to fit
        pivot.index = [n.split()[0] for n in pivot.index]

        lines = [f"ESG trend — {kpi} ({y_min}–{y_max}):"]
        lines.append(pivot.to_string())

        # Add YoY change for last year if available
        if len(pivot.columns) >= 2:
            last_yr = pivot.columns[-1]
            prev_yr = pivot.columns[-2]
            changes = ((pivot[last_yr] - pivot[prev_yr]) / pivot[prev_yr].abs() * 100).round(1)
            lines.append(f"\nYoY change {prev_yr}→{last_yr} (%):")
            lines.append(changes.to_string())

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUERY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
class QueryClassifier:
    GRAPH_KW = {"chart", "graph", "plot", "visuali", "draw", "bar chart",
                "line chart", "compare visual", "trend line", "show me a graph",
                "show me a chart"}
    DEEP_KW  = {"why", "cause", "reason", "explain", "factor", "drop",
                "decline", "increase", "spike", "anomal", "interpret",
                "insight", "what happened", "external", "geopolit", "impact"}

    def classify(self, q: str) -> str:
        q = q.lower()
        if any(kw in q for kw in self.GRAPH_KW):
            return "graph"
        if any(kw in q for kw in self.DEEP_KW):
            return "analytical"
        return "factual"

    def extract_years(self, q: str) -> list:
        return [int(m) for m in re.findall(r'\b(20[0-9]{2})\b', q)]

    def extract_companies(self, q: str, all_cos: list) -> list:
        q = q.lower()
        return [c for c in all_cos if any(w in q for w in c.lower().split() if len(w) > 3)]


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class GraphBuilder:
    COLORS = ["#C8102E", "#0A2240", "#00916E", "#F4A261", "#457B9D",
              "#E9C46A", "#264653", "#E76F51", "#2A9D8F", "#A8DADC"]

    def extract_spec(self, text: str) -> Optional[dict]:
        match = re.search(r"```chart_spec\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def strip_spec(self, text: str) -> str:
        return re.sub(r"```chart_spec\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()

    def build(self, spec: dict, df: pd.DataFrame) -> Optional[go.Figure]:
        if df.empty:
            return None
        try:
            chart_type = spec.get("chart_type", "line")
            title      = spec.get("title", "ESG Chart")
            x_col      = spec.get("x_col", "Year")
            y_cols     = spec.get("y_cols", [])
            companies  = spec.get("companies", ["all"])
            year_range = spec.get("year_range", [2009, 2023])
            color_by   = spec.get("color_by", "Company")

            sub = df.copy()
            if companies and "all" not in [str(c).lower() for c in companies]:
                sub = sub[sub["Company"].isin(companies)]
            sub = sub[sub["Year"].between(year_range[0], year_range[1])]

            avail_y = [c for c in y_cols if c in sub.columns]
            if not avail_y:
                return None

            if len(avail_y) == 1:
                plot_df = sub[["Company", "Year"] + avail_y].dropna()
                y = avail_y[0]
                kw = dict(color=color_by, title=title,
                          color_discrete_sequence=self.COLORS)
                fns = {
                    "line":        lambda: px.line(plot_df, x=x_col, y=y, **kw),
                    "bar":         lambda: px.bar(plot_df, x=x_col, y=y,
                                                  barmode="group", **kw),
                    "grouped_bar": lambda: px.bar(plot_df, x=x_col, y=y,
                                                  barmode="group", **kw),
                    "area":        lambda: px.area(plot_df, x=x_col, y=y, **kw),
                    "scatter":     lambda: px.scatter(plot_df, x=x_col, y=y, **kw),
                }
                fig = fns.get(chart_type, fns["line"])()
            else:
                plot_df = sub[["Company", "Year"] + avail_y].dropna()
                melted  = plot_df.melt(id_vars=["Company", "Year"],
                                       value_vars=avail_y,
                                       var_name="KPI", value_name="Value")
                fig = px.line(melted, x="Year", y="Value", color="KPI",
                              title=title, color_discrete_sequence=self.COLORS)

            fig.update_layout(
                font_family="Inter, Arial, sans-serif",
                title_font_size=15, title_font_color="#0A2240",
                paper_bgcolor="white", plot_bgcolor="#F8F9FA",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(l=50, r=30, t=70, b=50),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", dtick=1)
            fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB")
            return fig
        except Exception as e:
            logger.warning("[GraphBuilder] Chart build error: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHAT LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class ChatLogger:
    def __init__(self, username: str):
        safe = username.replace(" ", "_").replace("@", "_").replace(".", "_")[:20]
        # SHA-256 is the standard; MD5 is cryptographically broken even for non-secret use
        import hashlib
        uid  = hashlib.sha256(username.encode()).hexdigest()[:8]
        week = datetime.now().strftime("%Y-W%V")
        self.path = LOG_DIR / f"{safe}_{uid}_{week}.jsonl"

    def log(self, question: str, answer: str,
            query_type: str = "factual", had_chart: bool = False) -> None:
        entry = {
            "ts":         datetime.now().isoformat(),
            "question":   question,
            "answer":     answer[:1500],
            "query_type": query_type,
            "had_chart":  had_chart,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[ChatLogger] Could not write log: %s", e)

    @staticmethod
    def purge_old_logs(days: int = None) -> None:
        """Remove log files older than `days` days. Default from config."""
        retention = days if days is not None else cfg.LOG_RETENTION_DAYS
        cutoff = datetime.now() - timedelta(days=retention)
        for p in LOG_DIR.glob("*.jsonl"):
            try:
                if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                    p.unlink()
                    logger.info("[ChatLogger] Purged old log: %s", p.name)
            except Exception as e:
                logger.warning("[ChatLogger] Could not purge %s: %s", p.name, e)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
class ESGChatbot:
    def __init__(self, username: str):
        self.username      = username
        self.context       = DataContext()
        self.classifier    = QueryClassifier()
        self.graph         = GraphBuilder()
        self.copilot       = LocalLLMEngine()
        self.search        = ESGWebSearch()
        self.history: list = []

        # Build system prompt dynamically from actual data — no hardcoded company names
        self._rebuild_system_prompt()

        ChatLogger.purge_old_logs()   # uses cfg.LOG_RETENTION_DAYS
        self.logger = ChatLogger(username)

    def _rebuild_system_prompt(self) -> None:
        """Regenerate SYSTEM_PROMPT from current data context. Call after data reload."""
        companies  = self.context.companies()
        year_start = cfg.DATA_YEAR_START
        year_end   = cfg.DATA_YEAR_END
        if not companies:
            # Fallback if data isn't loaded yet — will be rebuilt on first chat()
            companies = []
        self.system_prompt = _build_system_prompt(companies, year_start, year_end)

    def reload_data(self):
        self.context.reload()
        self._rebuild_system_prompt()  # keep prompt in sync with new data

    def chat(self, question: str) -> "ChatResponse":
        query_type = self.classifier.classify(question)
        data_ctx   = self.context.build_context_str(question)

        # External context only for analytical questions (adds tokens)
        ext_ctx = ""
        if query_type == "analytical":
            years     = self.classifier.extract_years(question)
            companies = self.classifier.extract_companies(
                question, self.context.companies())
            ext_ctx = self.search.build_context_for_question(
                question, companies, years)
            # Keep external context short too
            if len(ext_ctx) > 400:
                ext_ctx = ext_ctx[:400] + "..."

        full_context = data_ctx
        if ext_ctx:
            full_context += f"\n\nEXTERNAL CONTEXT:\n{ext_ctx}"

        answer_raw = self.copilot.call(
            user_message  = question,
            data_context  = full_context,
            history       = self.history,
            system_prompt = SYSTEM_PROMPT,
        )

        spec   = self.graph.extract_spec(answer_raw)
        figure = None
        if spec and not self.context.df.empty:
            figure = self.graph.build(spec, self.context.df)

        answer_text = self.graph.strip_spec(answer_raw)

        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer_text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        self.logger.log(question, answer_text, query_type, had_chart=(figure is not None))
        return ChatResponse(text=answer_text, figure=figure, query_type=query_type)

    def clear_history(self):
        self.history = []


class ChatResponse:
    __slots__ = ("text", "figure", "query_type")

    def __init__(self, text: str, figure, query_type: str):
        self.text       = text
        self.figure     = figure
        self.query_type = query_type