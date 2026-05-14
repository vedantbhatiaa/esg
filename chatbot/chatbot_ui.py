"""
chatbot_ui.py — dss+ ESG Chatbot UI Component
===============================================
Renders the floating chat bubble (bottom-right) and full chat panel.
DSS internal employees only — enforced here and in the engine.

Import in app.py:
    from chatbot.chatbot_ui import render_chatbot
Then call at the end of the main router (after show_sidebar):
    render_chatbot()
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

# Lazy import to avoid loading pandas/plotly at module import time
def _get_bot(username: str):
    from chatbot.chatbot_engine import ESGChatbot
    key = f"_chatbot_{username}"
    if key not in st.session_state:
        st.session_state[key] = ESGChatbot(username)
    return st.session_state[key]


# ── DSS brand colours ─────────────────────────────────────────────────────────
DSS_RED   = "#C8102E"
DSS_NAVY  = "#0A2240"
DSS_GREEN = "#00916E"

# ── Floating bubble CSS + JS ──────────────────────────────────────────────────
BUBBLE_CSS = f"""
<style>
/* ── Chatbot floating bubble ─────────────────────────────── */
#dss-chat-bubble {{
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 54px;
    height: 54px;
    background: {DSS_RED};
    border-radius: 50%;
    box-shadow: 0 4px 18px rgba(0,0,0,0.22);
    cursor: pointer;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform .2s, box-shadow .2s;
    border: none;
}}
#dss-chat-bubble:hover {{
    transform: scale(1.08);
    box-shadow: 0 6px 24px rgba(0,0,0,0.30);
}}
#dss-chat-bubble svg {{
    width: 26px; height: 26px; fill: #fff;
}}
/* ── Notification badge ──────────────────────────────────── */
#dss-chat-badge {{
    position: fixed;
    bottom: 68px;
    right: 24px;
    background: {DSS_NAVY};
    color: #fff;
    border-radius: 10px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 700;
    z-index: 9999;
    display: none;
}}
</style>
<div id="dss-chat-badge"></div>
"""

BUBBLE_HTML = """
<button id="dss-chat-bubble" title="dss+ ESG Analyst Chatbot"
        onclick="window.parent.document.dispatchEvent(new CustomEvent('dss_chat_open'))">
  <!-- Chat icon SVG -->
  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
  </svg>
</button>
"""


def _render_messages(messages: list[dict]) -> None:
    """Render chat message history with avatars."""
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        figure  = msg.get("figure")

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                if figure is not None:
                    st.plotly_chart(figure, use_container_width=True, key=f"fig_{id(figure)}")


def _suggestion_buttons(bot) -> str | None:
    """Render suggestion chips; return text if one was clicked."""
    SUGGESTIONS = [
        "Water intake trend for VerdaTyres 2016–2023",
        "Compare CO₂ intensity across all companies in 2022",
        "Why did energy consumption drop in 2020?",
        "Chart total waste vs recovery rate for GammaTire SA",
        "Which company improved renewable electricity share most?",
        "Show scope 1 vs scope 2 emissions for all companies 2023",
    ]
    cols = st.columns(3)
    for i, sug in enumerate(SUGGESTIONS[:6]):
        with cols[i % 3]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                return sug
    return None


def render_chatbot() -> None:
    """
    Main entry point. Call this from app.py after show_sidebar() and page routing.
    Only shown to authenticated DSS employees.
    """
    # ── Auth guard ────────────────────────────────────────────────────────────
    if not st.session_state.get("authenticated"):
        return
    if not st.session_state.get("is_dss", False):
        return  # clients cannot see the chatbot

    username = st.session_state.get("user_name", "dss_user")
    bot      = _get_bot(username)

    # ── Session state for chat UI ─────────────────────────────────────────────
    if "chat_open"     not in st.session_state: st.session_state.chat_open     = False
    if "chat_messages" not in st.session_state: st.session_state.chat_messages = []

    # ── Floating bubble (injected into every page) ────────────────────────────
    st.markdown(BUBBLE_CSS, unsafe_allow_html=True)
    st.markdown(BUBBLE_HTML, unsafe_allow_html=True)

    # ── Toggle button (sidebar bottom — reliable Streamlit way) ──────────────
    with st.sidebar:
        st.markdown("---")
        chat_label = "💬 Close Analyst Chat" if st.session_state.chat_open else "💬 Open Analyst Chat"
        if st.button(chat_label, key="toggle_chatbot", use_container_width=True):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()

    if not st.session_state.chat_open:
        return

    # ── Chat panel ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
      <div style="width:38px;height:38px;background:{DSS_RED};border-radius:50%;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <span style="color:#fff;font-size:18px">🤖</span>
      </div>
      <div>
        <div style="font-size:16px;font-weight:700;color:{DSS_NAVY}">dss+ ESG Analyst</div>
        <div style="font-size:11px;color:#6B7280">Powered by Copilot · Internal use only</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Toolbar ───────────────────────────────────────────────────────────────
    tc1, tc2, tc3 = st.columns([2, 1, 1])
    with tc2:
        if st.button("🗑 Clear chat", key="chat_clear", use_container_width=True):
            st.session_state.chat_messages = []
            bot.clear_history()
            st.rerun()
    with tc3:
        if st.button("🔄 Reload data", key="chat_reload", use_container_width=True):
            bot.reload_data()
            st.success("Data refreshed.")

    # ── Message history ───────────────────────────────────────────────────────
    _render_messages(st.session_state.chat_messages)

    # ── Suggestion chips (show only when chat is empty) ───────────────────────
    clicked_suggestion = None
    if not st.session_state.chat_messages:
        st.markdown("**Quick questions to get started:**")
        clicked_suggestion = _suggestion_buttons(bot)

    # ── Input box ─────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask about ESG data, trends, causes, or request a specific chart…",
        key="chat_input_box",
    )

    # Prefer clicked suggestion over typed input
    final_input = clicked_suggestion or user_input

    if final_input:
        # Show user message immediately
        st.session_state.chat_messages.append({
            "role": "user", "content": final_input, "figure": None
        })

        with st.spinner("Analysing…"):
            response = bot.chat(final_input)

        st.session_state.chat_messages.append({
            "role":    "assistant",
            "content": response.text,
            "figure":  response.figure,
        })
        st.rerun()

    # ── Usage hint ────────────────────────────────────────────────────────────
    st.caption(
        "Ask factual questions, request trend analysis, or say "
        "**'chart water intake for VerdaTyres 2016–2023'** to generate a graph."
    )