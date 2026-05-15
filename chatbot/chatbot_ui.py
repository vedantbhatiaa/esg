"""
chatbot_ui.py  —  dss+ ESG Analyst Chat
========================================
• Floating red bubble (bottom-right) — click to open/close
• Streaming output — tokens appear as they're generated (feels instant)
• Clean chat UI — fixed height scrollable window, no page layout disruption
• DSS internal employees only
"""
from __future__ import annotations
import streamlit as st

DSS_RED   = "#C8102E"
DSS_NAVY  = "#0A2240"
DSS_LIGHT = "#F8F9FA"


def _get_bot(username: str):
    from chatbot.chatbot_engine import ESGChatbot
    key = f"_chatbot_{username}"
    if key not in st.session_state:
        st.session_state[key] = ESGChatbot(username)
    return st.session_state[key]


def _render_messages(messages: list) -> None:
    """Render all messages in the history."""
    for i, msg in enumerate(messages):
        is_user = msg["role"] == "user"
        avatar  = "👤" if is_user else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("figure") is not None:
                st.plotly_chart(
                    msg["figure"],
                    use_container_width=True,
                    key=f"fig_{i}_{id(msg['figure'])}",
                )


def _suggestions() -> str | None:
    """Quick-start chips shown on empty chat."""
    items = [
        "Water intake trend for VerdaTyres 2016-2023",
        "CO2 emission trend for DeltaGrip 2020-2022",
        "Why did energy consumption drop in 2020?",
        "Chart total waste vs recovery rate for GammaTire SA",
        "Which company improved renewable electricity share most?",
        "Compare CO2 intensity across all companies in 2022",
    ]
    cols = st.columns(2)
    for i, s in enumerate(items):
        with cols[i % 2]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                return s
    return None


def render_chatbot() -> None:
    # ── Auth guard ─────────────────────────────────────────────────────────────
    if not st.session_state.get("authenticated"):
        return
    if not st.session_state.get("is_dss", False):
        return

    username = st.session_state.get("user_name", "dss_user")
    bot      = _get_bot(username)

    if "chat_open"     not in st.session_state:
        st.session_state.chat_open = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # ── CSS: FAB + chat panel styling ─────────────────────────────────────────
    st.markdown(f"""
    <style>
    /* ── Floating Action Button ── */
    div.fab-wrap > div[data-testid="stButton"] > button {{
        position: fixed !important;
        bottom: 28px !important;
        right: 28px !important;
        width: 58px !important;
        height: 58px !important;
        border-radius: 50% !important;
        background: {DSS_RED} !important;
        color: #fff !important;
        font-size: 26px !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(200,16,46,.45) !important;
        z-index: 99999 !important;
        padding: 0 !important;
        min-width: unset !important;
        transition: transform .15s, box-shadow .15s !important;
    }}
    div.fab-wrap > div[data-testid="stButton"] > button:hover {{
        transform: scale(1.12) !important;
        box-shadow: 0 6px 28px rgba(200,16,46,.60) !important;
        background: {DSS_RED} !important;
        color: #fff !important;
    }}
    div.fab-wrap > div[data-testid="stButton"] > button:focus:not(:active) {{
        box-shadow: 0 4px 20px rgba(200,16,46,.45) !important;
    }}
    /* ── Chat panel container ── */
    .chat-panel {{
        background: #fff;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px 20px 12px 20px;
        margin-top: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,.08);
    }}
    /* ── Message bubbles ── */
    .stChatMessage {{
        border-radius: 10px !important;
        margin-bottom: 6px !important;
    }}
    /* ── Input bar ── */
    .stChatInput textarea {{
        border-radius: 10px !important;
        border: 1.5px solid #E5E7EB !important;
        font-size: 14px !important;
    }}
    .stChatInput textarea:focus {{
        border-color: {DSS_RED} !important;
        box-shadow: 0 0 0 2px rgba(200,16,46,.12) !important;
    }}
    /* ── Toolbar buttons ── */
    .toolbar-btn > div[data-testid="stButton"] > button {{
        border-radius: 8px !important;
        font-size: 12px !important;
        padding: 4px 10px !important;
        border: 1px solid #E5E7EB !important;
        background: #fff !important;
        color: #374151 !important;
    }}
    .toolbar-btn > div[data-testid="stButton"] > button:hover {{
        background: {DSS_LIGHT} !important;
        border-color: #D1D5DB !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Floating bubble — real Streamlit button styled via CSS ────────────────
    label = "✕" if st.session_state.chat_open else "💬"
    st.markdown('<div class="fab-wrap">', unsafe_allow_html=True)
    if st.button(label, key="fab_btn"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.chat_open:
        return

    # ── Chat panel ─────────────────────────────────────────────────────────────
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

    # Header row
    ok, status_msg = bot.copilot.is_available()
    dot   = "🟢" if ok else "🔴"
    sbg   = "#ECFDF5" if ok else "#FEF2F2"
    scol  = "#065F46" if ok else "#991B1B"

    hcol, tcol = st.columns([3, 2])
    with hcol:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:2px 0 10px 0">
          <div style="width:38px;height:38px;background:{DSS_RED};border-radius:50%;
                      display:flex;align-items:center;justify-content:center;
                      font-size:18px;flex-shrink:0">🤖</div>
          <div>
            <div style="font-size:14px;font-weight:700;color:{DSS_NAVY};
                        line-height:1.2">dss+ ESG Analyst</div>
            <div style="background:{sbg};color:{scol};border-radius:4px;
                        padding:1px 7px;font-size:11px;display:inline-block;
                        margin-top:2px">{dot} {bot.copilot.provider_label()}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with tcol:
        st.markdown('<div style="display:flex;gap:6px;justify-content:flex-end;'
                    'padding-top:4px">', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown('<div class="toolbar-btn">', unsafe_allow_html=True)
            if st.button("🗑", key="chat_clear", help="Clear chat",
                         use_container_width=True):
                st.session_state.chat_messages = []
                bot.clear_history()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="toolbar-btn">', unsafe_allow_html=True)
            if st.button("🔄", key="chat_reload", help="Reload data",
                         use_container_width=True):
                bot.reload_data()
                st.toast("Data reloaded ✅")
            st.markdown('</div>', unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="toolbar-btn">', unsafe_allow_html=True)
            if st.button("🔌", key="chat_test", help="Test connection",
                         use_container_width=True):
                ok2, msg2 = bot.copilot.is_available()
                if ok2:
                    st.success(msg2, icon="✅")
                else:
                    st.error(msg2)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Not available warning
    if not ok:
        st.warning(
            f"**{bot.copilot.provider_label()} not reachable.**  \n"
            f"{status_msg}  \n"
            "Make sure Ollama is running in your system tray."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("<hr style='margin:0 0 10px 0;border:none;"
                "border-top:1px solid #F3F4F6'>", unsafe_allow_html=True)

    # Message history
    _render_messages(st.session_state.chat_messages)

    # Suggestions (empty state)
    clicked = None
    if not st.session_state.chat_messages:
        st.markdown(
            "<div style='font-size:12px;color:#9CA3AF;margin:8px 0 6px 0'>"
            "Try asking:</div>",
            unsafe_allow_html=True,
        )
        clicked = _suggestions()

    # Input
    user_input = st.chat_input(
        "Ask about ESG data, trends, causes, or request a chart...",
        key="chat_input",
    )
    question = clicked or user_input

    if question:
        # Show user message immediately
        st.session_state.chat_messages.append(
            {"role": "user", "content": question, "figure": None}
        )
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        # ── Streaming response ─────────────────────────────────────────────────
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            accumulated = ""

            try:
                for chunk in bot.copilot.call_stream(
                    user_message  = question,
                    data_context  = bot.context.build_context_str(question),
                    history       = bot.history,
                    system_prompt = bot.system_prompt,
                ):
                    accumulated += chunk
                    # Show live text with blinking cursor
                    placeholder.markdown(accumulated + "▌")

            except Exception as e:
                accumulated = f"❌ Error: {str(e)}"

            # Final render without cursor
            placeholder.markdown(accumulated)

            # Extract and render chart if present
            spec   = bot.graph.extract_spec(accumulated)
            figure = None
            if spec and not bot.context.df.empty:
                figure = bot.graph.build(spec, bot.context.df)
                if figure:
                    st.plotly_chart(figure, use_container_width=True,
                                    key=f"stream_fig_{len(st.session_state.chat_messages)}")

            # Clean text (remove chart_spec block)
            clean_text = bot.graph.strip_spec(accumulated)

        # Update history
        bot.history.append({"role": "user",      "content": question})
        bot.history.append({"role": "assistant",  "content": clean_text})
        if len(bot.history) > 12:
            bot.history = bot.history[-12:]

        # Log
        bot.logger.log(question, clean_text,
                       bot.classifier.classify(question),
                       had_chart=(figure is not None))

        st.session_state.chat_messages.append({
            "role":    "assistant",
            "content": clean_text,
            "figure":  figure,
        })

    st.markdown('</div>', unsafe_allow_html=True)