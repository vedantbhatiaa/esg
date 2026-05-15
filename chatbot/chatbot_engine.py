"""
chatbot_panel.py  —  dss+ ESG Analyst · Option 5 split panel
=============================================================
Renders the right-side analyst panel with:
  - Red header + context badge (auto-detects current page/company/year)
  - 4-tab nav: Chat | Charts | History | Config
  - Streaming chat output
  - Charts tab: pre-generated charts from chat history
  - History tab: this week's log entries
  - Config tab: model/provider info

Usage in app.py:
    from chatbot.chatbot_panel import analyst_panel_css, render_analyst_panel

    # At top of page functions that want the panel:
    left_col, panel_col = analyst_panel_layout()
    with left_col:
        ... your existing page content ...
    with panel_col:
        render_analyst_panel(page="analysis", company="VerdaTyres Corp", year=2023)
"""
from __future__ import annotations
import streamlit as st
from typing import Optional

DSS_RED  = "#C8102E"
DSS_NAVY = "#0A2240"

# Context labels per page
PAGE_CONTEXT = {
    "analysis":      ("chart-line",     "Analysis"),
    "benchmarking":  ("layout-columns", "Benchmarking"),
    "verification":  ("shield-check",   "Verification"),
    "entry":         ("file-text",      "Data Entry"),
    "readiness":     ("brain",          "AI Readiness"),
}


def analyst_panel_css() -> None:
    """Inject panel CSS once per page load."""
    st.markdown(f"""
    <style>
    /* ── Panel outer shell ─────────────────────────────────── */
    .ap-shell {{
        background: var(--color-background-primary);
        border: 0.5px solid var(--color-border-tertiary);
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: calc(100vh - 120px);
        position: sticky;
        top: 16px;
    }}

    /* ── Red header ────────────────────────────────────────── */
    .ap-hd {{
        background: {DSS_RED};
        padding: 10px 12px;
        display: flex;
        align-items: center;
        gap: 9px;
        flex-shrink: 0;
    }}
    .ap-hd-avatar {{
        width: 28px; height: 28px;
        background: rgba(255,255,255,.2);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }}
    .ap-hd-title {{ color: #fff; font-size: 13px; font-weight: 500; line-height: 1.2; }}
    .ap-hd-sub   {{ color: rgba(255,255,255,.7); font-size: 10px; margin-top: 1px; }}

    /* ── Context bar ───────────────────────────────────────── */
    .ap-ctx {{
        background: var(--color-background-secondary);
        border-bottom: 0.5px solid var(--color-border-tertiary);
        padding: 5px 10px;
        display: flex; align-items: center; gap: 5px;
        font-size: 10px; color: var(--color-text-secondary);
        flex-shrink: 0;
    }}
    .ap-ctx-pill {{
        background: rgba(200,16,46,.1);
        color: #7F1D1D;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 10px; font-weight: 500;
    }}

    /* ── Nav tabs ──────────────────────────────────────────── */
    .ap-nav {{
        display: flex;
        border-bottom: 0.5px solid var(--color-border-tertiary);
        flex-shrink: 0;
    }}
    .ap-nav-item {{
        flex: 1; padding: 7px 4px 6px;
        text-align: center;
        font-size: 10px;
        color: var(--color-text-secondary);
        cursor: pointer;
        border-bottom: 2px solid transparent;
        display: flex; flex-direction: column;
        align-items: center; gap: 2px;
        transition: color .1s;
    }}
    .ap-nav-item.active {{
        color: {DSS_RED};
        border-bottom-color: {DSS_RED};
        font-weight: 500;
    }}
    .ap-nav-item i {{ font-size: 14px; }}

    /* ── Message bubbles ───────────────────────────────────── */
    .ap-msg-u {{
        align-self: flex-end;
        background: {DSS_RED};
        color: #fff;
        border-radius: 10px 10px 2px 10px;
        padding: 6px 9px;
        font-size: 12px;
        max-width: 88%;
        line-height: 1.4;
        margin-bottom: 2px;
    }}
    .ap-msg-b {{
        align-self: flex-start;
        background: var(--color-background-secondary);
        color: var(--color-text-primary);
        border-radius: 10px 10px 10px 2px;
        padding: 6px 9px;
        font-size: 12px;
        max-width: 92%;
        line-height: 1.4;
        margin-bottom: 2px;
    }}

    /* ── Suggestion chips ──────────────────────────────────── */
    .ap-chip {{
        background: var(--color-background-secondary);
        border: 0.5px solid var(--color-border-tertiary);
        border-radius: 6px;
        padding: 5px 9px;
        font-size: 11px;
        color: var(--color-text-primary);
        cursor: pointer;
        line-height: 1.3;
        margin-bottom: 4px;
    }}
    .ap-chip:hover {{
        border-color: {DSS_RED};
        color: {DSS_RED};
    }}

    /* ── History entry ─────────────────────────────────────── */
    .ap-hist {{
        background: var(--color-background-secondary);
        border: 0.5px solid var(--color-border-tertiary);
        border-radius: 7px;
        padding: 7px 9px;
        margin-bottom: 5px;
        font-size: 11px;
    }}
    .ap-hist-q  {{ color: var(--color-text-primary); font-weight: 500; margin-bottom: 2px; }}
    .ap-hist-ts {{ color: var(--color-text-secondary); font-size: 9px; }}

    /* ── Config row ────────────────────────────────────────── */
    .ap-cfg-row {{
        display: flex; justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 0.5px solid var(--color-border-tertiary);
        font-size: 11px;
    }}
    .ap-cfg-label {{ color: var(--color-text-secondary); }}
    .ap-cfg-val   {{ color: var(--color-text-primary); font-weight: 500; }}

    /* ── Status dot ────────────────────────────────────────── */
    .ap-dot-green {{ color: #22C55E; font-size: 9px; }}
    .ap-dot-red   {{ color: #EF4444; font-size: 9px; }}
    </style>
    """, unsafe_allow_html=True)


def analyst_panel_layout():
    """
    Return (left_col, right_col) — left gets page content, right gets panel.
    Widths: 67% content, 33% panel.
    Only shown to dss+ employees.
    """
    if not st.session_state.get("is_dss", False):
        # Non-DSS users: full-width single column
        return st.columns([1, 0.001])  # effectively one column
    return st.columns([2.1, 1], gap="medium")


def render_analyst_panel(
    page: str = "analysis",
    company: Optional[str] = None,
    year: Optional[int] = None,
) -> None:
    """
    Render the full Option 5 split analyst panel in the right column.
    Call this inside `with panel_col:` after analyst_panel_layout().
    """
    if not st.session_state.get("is_dss", False):
        return

    # ── Get bot instance ──────────────────────────────────────────────────────
    username = st.session_state.get("user_name", "dss_user")
    try:
        from chatbot.chatbot_engine import ESGChatbot
        bot_key = f"_chatbot_{username}"
        if bot_key not in st.session_state:
            st.session_state[bot_key] = ESGChatbot(username)
        bot = st.session_state[bot_key]
    except Exception as e:
        st.error(f"Analyst panel unavailable: {e}")
        return

    # ── Session state for this panel ──────────────────────────────────────────
    if "ap_tab"      not in st.session_state: st.session_state.ap_tab      = "chat"
    if "ap_messages" not in st.session_state: st.session_state.ap_messages = []
    if "ap_charts"   not in st.session_state: st.session_state.ap_charts   = []

    ok, status_msg = bot.copilot.is_available()

    # ── Context labels ────────────────────────────────────────────────────────
    icon, page_label = PAGE_CONTEXT.get(page, ("robot", page.title()))
    company  = company or st.session_state.get("reporting_company") or \
               st.session_state.get("user_company") or "All Companies"
    year     = year    or st.session_state.get("reporting_year") or 2023
    sub_text = f"{company} · {year}" if company != "All Companies" else \
               f"All companies · {year}"

    # Inject CSS
    analyst_panel_css()

    # ── Build header HTML ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ap-hd">
      <div class="ap-hd-avatar">
        <i class="ti ti-robot" style="color:#fff;font-size:14px" aria-hidden="true"></i>
      </div>
      <div>
        <div class="ap-hd-title">dss+ ESG Analyst</div>
        <div class="ap-hd-sub">{sub_text}</div>
      </div>
    </div>
    <div class="ap-ctx">
      <span>Context:</span>
      <span class="ap-ctx-pill">{company.split()[0]}</span>
      <span class="ap-ctx-pill">{year}</span>
      <span class="ap-ctx-pill">{page_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab navigation ────────────────────────────────────────────────────────
    tab_defs = [
        ("chat",    "ti-message-circle", "Chat"),
        ("charts",  "ti-chart-bar",      "Charts"),
        ("history", "ti-history",        "History"),
        ("config",  "ti-settings",       "Config"),
    ]

    tab_cols = st.columns(4)
    for i, (tab_id, tab_icon, tab_label) in enumerate(tab_defs):
        with tab_cols[i]:
            active_class = "active" if st.session_state.ap_tab == tab_id else ""
            st.markdown(
                f'<div class="ap-nav-item {active_class}">'
                f'<i class="ti {tab_icon}" aria-hidden="true"></i>'
                f'<span>{tab_label}</span></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                tab_label,
                key=f"ap_tab_{tab_id}",
                use_container_width=True,
            ):
                st.session_state.ap_tab = tab_id
                st.rerun()

    st.markdown("<hr style='margin:0;border:none;border-top:0.5px solid var(--color-border-tertiary)'>",
                unsafe_allow_html=True)

    # ── Tab content ───────────────────────────────────────────────────────────
    current_tab = st.session_state.ap_tab

    # ════════════════════════════════════════════════════════
    # CHAT TAB
    # ════════════════════════════════════════════════════════
    if current_tab == "chat":
        if not ok:
            st.warning(f"**{bot.copilot.provider_label()} not reachable.**  \n{status_msg}")
        else:
            # Render message history
            for i, msg in enumerate(st.session_state.ap_messages):
                is_user = msg["role"] == "user"
                css     = "ap-msg-u" if is_user else "ap-msg-b"
                st.markdown(
                    f'<div class="{css}">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("figure"):
                    st.plotly_chart(
                        msg["figure"],
                        use_container_width=True,
                        key=f"ap_fig_{i}",
                    )

            # Suggestions on empty chat
            if not st.session_state.ap_messages:
                page_suggestions = {
                    "analysis":     [
                        f"Why did CO₂ change in 2020?",
                        f"Chart water intake for {company.split()[0]} 2016–2023",
                        f"What is the energy intensity trend?",
                    ],
                    "benchmarking": [
                        "Who has the best CO₂ intensity in 2023?",
                        "Which company improved most in renewable electricity?",
                        "Compare water KPI across all companies",
                    ],
                    "verification": [
                        f"Explain the flags for {company.split()[0]}",
                        "What could cause a 24% Scope 2 spike?",
                        "Is this submission ready for consolidation?",
                    ],
                }.get(page, [
                    "What was the CO₂ intensity in 2023?",
                    "Chart total energy trend",
                    "Compare water intake year on year",
                ])

                st.markdown(
                    "<div style='font-size:11px;color:var(--color-text-secondary);"
                    "margin:6px 0 4px'>Try asking:</div>",
                    unsafe_allow_html=True,
                )
                for sug in page_suggestions:
                    if st.button(sug, key=f"ap_sug_{sug[:20]}", use_container_width=True):
                        _handle_ap_chat(sug, bot, company, year)
                        st.rerun()

            # Chat input
            user_input = st.chat_input(
                f"Ask about {company.split()[0]}...",
                key="ap_chat_input",
            )
            if user_input:
                _handle_ap_chat(user_input, bot, company, year)
                st.rerun()

    # ════════════════════════════════════════════════════════
    # CHARTS TAB
    # ════════════════════════════════════════════════════════
    elif current_tab == "charts":
        if not st.session_state.ap_charts:
            st.markdown(
                "<div style='font-size:12px;color:var(--color-text-secondary);"
                "padding:12px 0;text-align:center'>"
                "No charts yet.<br>Ask me to chart something in the Chat tab.</div>",
                unsafe_allow_html=True,
            )
            # Quick chart suggestions
            st.markdown("<div style='font-size:11px;color:var(--color-text-secondary);margin:8px 0 4px'>Generate a chart:</div>",
                        unsafe_allow_html=True)
            chart_sugs = [
                f"Chart CO₂ trend for {company.split()[0]} 2016–2023",
                "Compare renewable electricity share all companies",
                f"Chart waste recovery rate for {company.split()[0]}",
            ]
            for cs in chart_sugs:
                if st.button(cs, key=f"ap_cs_{cs[:20]}", use_container_width=True):
                    st.session_state.ap_tab = "chat"
                    _handle_ap_chat(cs, bot, company, year)
                    st.rerun()
        else:
            for i, (title, fig) in enumerate(st.session_state.ap_charts):
                st.markdown(
                    f"<div style='font-size:11px;font-weight:500;color:var(--color-text-primary);"
                    f"margin:6px 0 3px'>{title}</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"ap_chart_{i}")

    # ════════════════════════════════════════════════════════
    # HISTORY TAB
    # ════════════════════════════════════════════════════════
    elif current_tab == "history":
        try:
            from chatbot.chatbot_engine import ChatLogger
            logger  = ChatLogger(username)
            entries = logger.load_week()
        except Exception:
            entries = []

        if not entries:
            st.markdown(
                "<div style='font-size:12px;color:var(--color-text-secondary);"
                "padding:12px 0;text-align:center'>No interactions this week yet.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:11px;color:var(--color-text-secondary);"
                f"margin-bottom:8px'>{len(entries)} interactions this week</div>",
                unsafe_allow_html=True,
            )
            for entry in reversed(entries[-15:]):
                from datetime import datetime
                ts_str = ""
                try:
                    dt     = datetime.fromisoformat(entry["ts"])
                    ts_str = dt.strftime("%d %b · %H:%M")
                except Exception:
                    ts_str = entry.get("ts", "")[:16]

                chart_icon = " · 📊" if entry.get("had_chart") else ""
                qtype      = entry.get("query_type", "")
                type_badge = {"graph": "chart", "analytical": "analysis",
                              "factual": "factual"}.get(qtype, qtype)
                st.markdown(
                    f'<div class="ap-hist">'
                    f'<div class="ap-hist-q">{entry.get("question","")[:80]}</div>'
                    f'<div class="ap-hist-ts">{ts_str} · {type_badge}{chart_icon}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════════════════════════════
    # CONFIG TAB
    # ════════════════════════════════════════════════════════
    elif current_tab == "config":
        status_dot  = "🟢" if ok else "🔴"
        status_text = "Connected" if ok else "Disconnected"

        st.markdown(
            f'<div class="ap-cfg-row">'
            f'<span class="ap-cfg-label">Provider</span>'
            f'<span class="ap-cfg-val">{bot.copilot.provider_label()}</span>'
            f'</div>'
            f'<div class="ap-cfg-row">'
            f'<span class="ap-cfg-label">Status</span>'
            f'<span class="ap-cfg-val">{status_dot} {status_text}</span>'
            f'</div>'
            f'<div class="ap-cfg-row">'
            f'<span class="ap-cfg-label">Context window</span>'
            f'<span class="ap-cfg-val">2 048 tokens</span>'
            f'</div>'
            f'<div class="ap-cfg-row">'
            f'<span class="ap-cfg-label">Max output</span>'
            f'<span class="ap-cfg-val">500 tokens</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Test connection", key="ap_test_conn",
                         use_container_width=True):
                ok2, msg2 = bot.copilot.is_available()
                if ok2:
                    st.success(msg2)
                else:
                    st.error(msg2)
        with col2:
            if st.button("🗑 Clear history", key="ap_clear_hist",
                         use_container_width=True):
                st.session_state.ap_messages = []
                st.session_state.ap_charts   = []
                bot.clear_history()
                st.rerun()

        if st.button("🔄 Reload master data", key="ap_reload_data",
                     use_container_width=True):
            bot.reload_data()
            st.toast("Master data reloaded ✅")

        if not ok:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.warning(
                f"{status_msg}\n\n"
                "Make sure Ollama is running in your system tray."
            )


def _handle_ap_chat(question: str, bot, company: str, year: int) -> None:
    """
    Process one chat turn with streaming output rendered in the panel.
    Updates ap_messages and ap_charts in session_state.
    """
    # Add user message
    st.session_state.ap_messages.append(
        {"role": "user", "content": question, "figure": None}
    )

    # Build compact context for this page
    data_ctx = bot.context.build_context_str(question)

    # Collect streaming response
    accumulated = ""
    placeholder = st.empty()

    try:
        for chunk in bot.copilot.call_stream(
            user_message  = question,
            data_context  = data_ctx,
            history       = bot.history,
            system_prompt = bot.system_prompt,
        ):
            accumulated += chunk
            # Render streaming text — simple markdown so it looks clean in panel
            placeholder.markdown(
                f'<div class="ap-msg-b">{accumulated}▌</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        accumulated = f"Error: {str(e)}"

    placeholder.markdown(
        f'<div class="ap-msg-b">{accumulated}</div>',
        unsafe_allow_html=True,
    )

    # Extract chart spec if present
    spec   = bot.graph.extract_spec(accumulated)
    figure = None
    if spec and not bot.context.df.empty:
        figure = bot.graph.build(spec, bot.context.df)
        if figure:
            st.session_state.ap_charts.append(
                (spec.get("title", "Chart"), figure)
            )

    clean_text = bot.graph.strip_spec(accumulated)

    # Update history and messages
    bot.history.append({"role": "user",      "content": question})
    bot.history.append({"role": "assistant",  "content": clean_text})
    if len(bot.history) > 12:
        bot.history = bot.history[-12:]

    bot.logger.log(
        question, clean_text,
        bot.classifier.classify(question),
        had_chart=(figure is not None),
    )

    st.session_state.ap_messages.append({
        "role":    "assistant",
        "content": clean_text,
        "figure":  figure,
    })