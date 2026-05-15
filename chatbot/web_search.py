"""
web_search.py  —  Public ESG data retrieval for dss+ chatbot
=============================================================
Uses DuckDuckGo search (no API key, no account, completely free).
Fetches real public information about:
  - Industry events affecting ESG metrics (energy crisis, COVID, etc.)
  - Company-specific ESG news and reports
  - Regulatory changes (CSRD, Paris Agreement, etc.)
  - Commodity prices (rubber, oil) affecting tire manufacturer costs/emissions

No scraping of paywalled content. Uses only public search snippets.
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Optional


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a DuckDuckGo search and return list of {title, snippet, url}.
    Falls back gracefully if duckduckgo-search is not installed.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddg:
            for r in ddg.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
        return results
    except ImportError:
        return [{"title": "Search unavailable",
                 "snippet": "Install duckduckgo-search: pip install duckduckgo-search",
                 "url": ""}]
    except Exception as e:
        return [{"title": "Search error", "snippet": str(e), "url": ""}]


class ESGWebSearch:
    """
    Retrieves public context relevant to ESG metric shifts.
    All methods return a formatted string ready to inject into LLM context.
    """

    # Known external events that commonly explain ESG metric shifts
    KNOWN_EVENTS = {
        2009: ["Global financial crisis recovery", "Low industrial output post-GFC"],
        2010: ["Post-GFC industrial rebound", "Energy demand surge"],
        2014: ["Oil price crash (-50%)", "Lower energy costs for manufacturers"],
        2015: ["Paris Agreement signed (COP21)", "ESG reporting push begins"],
        2016: ["Oil price bottom", "EU ETS reforms"],
        2019: ["Pre-COVID supply chain tension", "EU Green Deal proposed"],
        2020: ["COVID-19 lockdowns (-15 to -40% industrial output)",
               "Global energy demand crash", "Oil price went negative (April 2020)",
               "Many factories shut Q1-Q2"],
        2021: ["Post-COVID demand surge", "Global supply chain crisis",
               "Energy price spike", "Shipping cost +300%"],
        2022: ["Russia-Ukraine war → European gas crisis",
               "Energy prices +200-300% for European manufacturers",
               "China COVID lockdowns (Shanghai, Shenzhen)",
               "CSRD regulation adopted by EU"],
        2023: ["Energy prices stabilising", "CSRD mandatory reporting begins",
               "EV transition pressure on tire demand",
               "ESG disclosure requirements tightening globally"],
    }

    def get_year_context(self, year: int) -> str:
        """Return known external context for a given year."""
        events = self.KNOWN_EVENTS.get(year, [])
        if not events:
            # Check adjacent years
            events = (self.KNOWN_EVENTS.get(year - 1, []) +
                      self.KNOWN_EVENTS.get(year + 1, []))
        if events:
            return f"Known external factors for {year}:\n" + "\n".join(f"• {e}" for e in events)
        return f"No specific known events recorded for {year}."

    def get_year_range_context(self, year_start: int, year_end: int) -> str:
        """Return context for a range of years."""
        lines = []
        for yr in range(year_start, year_end + 1):
            events = self.KNOWN_EVENTS.get(yr)
            if events:
                lines.append(f"{yr}: " + "; ".join(events))
        return "\n".join(lines) if lines else "No major events recorded for this period."

    def search_company_esg(self, company: str, year: Optional[int] = None) -> str:
        """
        Search for public ESG news about a specific company.
        Uses the company name without the dummy suffix for real searches.
        """
        # Map dummy company names to realistic industry terms for better results
        industry_term = "tire manufacturer ESG sustainability"
        year_str = str(year) if year else ""
        query = f"{company} ESG sustainability report {year_str} {industry_term}"

        results = _ddg_search(query, max_results=4)
        if not results or results[0]["title"] == "Search unavailable":
            return f"Web search unavailable. Using built-in context only."

        lines = [f"Public search results for '{company}' ESG {year_str}:"]
        for r in results[:3]:
            if r["snippet"]:
                # Truncate long snippets
                snippet = r["snippet"][:200] + "..." if len(r["snippet"]) > 200 else r["snippet"]
                lines.append(f"• {r['title']}: {snippet}")
        return "\n".join(lines)

    def search_industry_event(self, topic: str, year: Optional[int] = None) -> str:
        """
        Search for industry-level events (energy prices, regulations, etc.).
        """
        year_str = str(year) if year else ""
        query = f"tire manufacturing industry {topic} {year_str} impact energy emissions"

        results = _ddg_search(query, max_results=4)
        if not results or results[0]["title"] == "Search unavailable":
            return self.get_year_context(year) if year else ""

        lines = [f"Industry context — {topic} {year_str}:"]
        for r in results[:3]:
            if r["snippet"]:
                snippet = r["snippet"][:200] + "..." if len(r["snippet"]) > 200 else r["snippet"]
                lines.append(f"• {r['title']}: {snippet}")
        return "\n".join(lines)

    def build_context_for_question(self, question: str,
                                   companies: list[str],
                                   years: list[int]) -> str:
        """
        Intelligently build external context based on what the question is about.
        Called before sending to LLM for deep analytical questions.
        """
        q_lower = question.lower()
        context_parts = []

        # Year context (built-in knowledge, always fast)
        if years:
            yr_min, yr_max = min(years), max(years)
            if yr_max - yr_min > 2:
                context_parts.append(self.get_year_range_context(yr_min, yr_max))
            else:
                for yr in years:
                    context_parts.append(self.get_year_context(yr))

        # Decide whether to do a live web search based on question type
        needs_web = any(kw in q_lower for kw in [
            "why", "cause", "reason", "explain", "factor", "external",
            "geopolit", "crisis", "war", "covid", "pandemic", "regulation",
            "policy", "market", "price", "industry", "news"
        ])

        if needs_web:
            # Detect topic for targeted search
            topics = []
            if any(kw in q_lower for kw in ["energy", "electricity", "emission"]):
                topics.append("energy prices")
            if any(kw in q_lower for kw in ["water", "drought"]):
                topics.append("water scarcity")
            if any(kw in q_lower for kw in ["waste", "recycl"]):
                topics.append("waste management regulations")
            if any(kw in q_lower for kw in ["co2", "carbon", "emission", "scope"]):
                topics.append("carbon emissions regulations")

            year_for_search = years[-1] if years else None
            for topic in topics[:2]:  # max 2 searches to stay fast
                result = self.search_industry_event(topic, year_for_search)
                if result:
                    context_parts.append(result)

        return "\n\n".join(context_parts) if context_parts else ""