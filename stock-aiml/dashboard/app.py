"""Streamlit dashboard — main entry point with sidebar navigation."""
from __future__ import annotations

import sys
import os
# Add the root 'stock-aiml' directory to sys.path so 'dashboard.' and 'backend.' modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock-AIML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's default auto-generated sidebar navigation
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 📈 Stock-AIML")
st.sidebar.markdown("### Deep Learning Backtesting & Strategy Optimization")
st.sidebar.divider()
st.sidebar.caption("v1.0.0")

PAGES = {
    "ℹ️ About Project":    "about_project",
    "📁 Dataset Upload":   "dataset_upload",
    "⚙️ Strategy Config":  "strategy_config",
    "▶️ Run Backtest":      "backtest_run",
    "📊 Results Summary":  "results_summary",
    "⚡ Stress Test":      "stress_test",
    "🔍 Explanation":      "explanation",
    "📤 Export":           "export",
    "📚 API Docs":         "api_docs",
    "🧪 MLflow Tracking":  "mlflow_tracking",
}

# Store selected page
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "ℹ️ About Project"

st.session_state["current_page"] = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    label_visibility="collapsed",
)

st.sidebar.divider()

# ── Load the selected page ───────────────────────────────────────────────────
page_module_name = PAGES[st.session_state["current_page"]]
page_module = __import__(f"dashboard.pages.{page_module_name}", fromlist=["render"])

with st.spinner(f"Loading {st.session_state['current_page']}..."):
    page_module.render()
