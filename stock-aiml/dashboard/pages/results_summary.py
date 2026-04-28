"""Results Summary page."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def _metric_card(label: str, value: str | float, delta: str | None = None, color: str = "#3B82F6"):
    st.metric(label=label, value=value, delta=delta)

def render():
    st.title("📊 Results Summary")

    if "backtest_result" not in st.session_state:
        st.warning("⚠️ No backtest results yet. Run a backtest first.")
        return

    result = st.session_state["backtest_result"]
    m = st.session_state["metrics"]

    # ── Summary cards ────────────────────────────────────────────────────────
    st.markdown("### Performance Overview")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        ar = m.get("annualized_return", 0)
        _metric_card("Annualized Return", f"{ar:.2f}%",
                     color="#22C55E" if ar > 0 else "#EF4444")
    with kpi_cols[1]:
        sr = m.get("sharpe_ratio", 0)
        _metric_card("Sharpe Ratio", f"{sr:.3f}",
                     color="#22C55E" if sr > 1 else "#F59E0B")
    with kpi_cols[2]:
        mdd = m.get("max_drawdown_pct", 0)
        _metric_card("Max Drawdown", f"-{mdd:.2f}%",
                     color="#EF4444")
    with kpi_cols[3]:
        cvar = m.get("cvar_95", 0)
        _metric_card("CVaR (95%)", f"{cvar:.3f}%",
                     color="#EF4444" if cvar < -1 else "#22C55E")

    kpi_cols2 = st.columns(4)
    with kpi_cols2[0]:
        st.metric("Sortino Ratio", f"{m.get('sortino_ratio', 0):.3f}")
    with kpi_cols2[1]:
        st.metric("Win Rate", f"{m.get('win_rate', 0):.1f}%")
    with kpi_cols2[2]:
        st.metric("Profit Factor", f"{m.get('profit_factor', 0):.3f}")
    with kpi_cols2[3]:
        st.metric("Total Trades", m.get("total_trades", 0))

    st.divider()

    # ── Charts ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Equity Curve")
        equity = st.session_state.get("equity_curve", [])
        if equity:
            fig = go.Figure(go.Scatter(
                x=list(range(len(equity))),
                y=equity,
                mode="lines",
                line=dict(color="#3B82F6", width=2),
            ))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("### 📉 Drawdown")
        dd = st.session_state.get("drawdown_curve", [])
        if dd:
            fig = go.Figure(go.Scatter(
                x=list(range(len(dd))),
                y=dd,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#EF4444", width=1.5),
            ))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.markdown("### 📋 Trade Log")
    trades = st.session_state.get("trades", [])
    if trades:
        df_trades = pd.DataFrame(trades)
        st.dataframe(df_trades, width="stretch", height=300)
        st.caption(f"{len(trades)} trades")
    else:
        st.info("No trades recorded in this backtest.")

    # ── Config used ──────────────────────────────────────────────────────────
    with st.expander("🔧 View Backtest Configuration"):
        st.json(result.config)
