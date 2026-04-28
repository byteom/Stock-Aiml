"""Streamlit dashboard — main entry point with sidebar navigation."""
from __future__ import annotations

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock-AIML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 📈 Stock-AIML")
st.sidebar.markdown("### Deep Learning Backtesting & Strategy Optimization")
st.sidebar.divider()
st.sidebar.caption("v1.0.0")

PAGES = {
    "📁 Dataset Upload":   "dataset_upload",
    "⚙️ Strategy Config":  "strategy_config",
    "▶️ Run Backtest":      "backtest_run",
    "📊 Results Summary":  "results_summary",
    "⚡ Stress Test":      "stress_test",
    "🔍 Explanation":      "explanation",
    "📤 Export":           "export",
}

# Store selected page
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "📁 Dataset Upload"

st.session_state["current_page"] = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("**Quick Links**")
st.sidebar.markdown("- [API Docs](/docs)" if True else "")
st.sidebar.markdown("- [MLflow](/)" if True else "")

# ── Load the selected page ───────────────────────────────────────────────────
page_module_name = PAGES[st.session_state["current_page"]]
page_module = __import__(f"dashboard.pages.{page_module_name}", fromlist=["render"])

with st.spinner(f"Loading {st.session_state['current_page']}..."):
    page_module.render()
