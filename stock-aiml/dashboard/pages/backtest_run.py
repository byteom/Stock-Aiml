"""Run Backtest page."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def render():
    st.title("▶️ Run Backtest")

    if "data_ready" not in st.session_state:
        st.warning("⚠️ Please upload data first on the **Dataset Upload** page.")
        return

    if "strategy_params" not in st.session_state:
        st.warning("⚠️ Please configure your strategy on the **Strategy Config** page.")
        return

    col_run = st.columns([1, 2, 1])
    with col_run[1]:
        run_button = st.button("🚀 Run Execution-Aware Backtest", width="stretch", type="primary")

    st.divider()

    if run_button:
        with st.spinner("Running backtest..."):
            from backend.services.data_loader import DataLoader
            from backend.services.feature_engineering import FeatureEngine
            from backend.services.backtester import Backtester
            from backend.services.execution_model import ExecutionSurrogate

            start = time.time()

            # Load data
            data_path = Path(__file__).parents[3] / "data.csv"
            loader = DataLoader(data_path)
            loader.load_csv(data_path)
            splits = loader.create_walk_forward_splits(n_splits=3)
            test_df = splits[0]["test"]

            # Engineer features
            engine = FeatureEngine()
            test_df = engine.transform(test_df)

            # Run backtest
            exec_params = st.session_state.get("exec_params", {})
            initial_cap = exec_params.get("initial_capital", 1_000_000.0)
            commission  = exec_params.get("commission_pct", 0.001)
            slippage    = exec_params.get("slippage_bps", 5.0)
            exec_surrogate = ExecutionSurrogate()
            bt = Backtester(
                initial_capital=initial_cap,
                commission_pct=commission,
                slippage_bps=slippage,
                exec_surrogate=exec_surrogate,
            )

            result = bt.run(
                df=test_df,
                strategy_name=st.session_state.get("strategy_name", "momentum"),
                strategy_params=st.session_state.get("strategy_params", {}),
                stochastic_exec=True,
            )

            elapsed = time.time() - start
            st.session_state["backtest_result"] = result
            st.session_state["equity_curve"] = result.equity_curve
            st.session_state["timestamps"] = result.timestamps
            st.session_state["drawdown_curve"] = result.drawdown_curve
            st.session_state["metrics"] = result.metrics
            st.session_state["trades"] = result.trades

        st.success(f"✅ Backtest completed in {elapsed:.1f}s — {len(result.trades)} trades")

        # ── Equity curve plot ────────────────────────────────────────────────
        st.markdown("### 📈 Equity Curve")
        timestamps = result.timestamps
        equity = result.equity_curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=list(range(len(equity))),
            y=equity,
            mode="lines",
            line=dict(color="#3B82F6", width=2),
            name="Equity",
        ))
        fig_eq.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title="Bar",
            yaxis_title="Portfolio Value ($)",
        )
        st.plotly_chart(fig_eq, width="stretch")

        # ── Drawdown curve ────────────────────────────────────────────────────
        st.markdown("### 📉 Drawdown Curve")
        dd_curve = result.drawdown_curve
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=list(range(len(dd_curve))),
            y=dd_curve,
            mode="lines",
            line=dict(color="#EF4444", width=2),
            fill="tozeroy",
            name="Drawdown (%)",
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title="Bar",
            yaxis_title="Drawdown (%)",
        )
        st.plotly_chart(fig_dd, width="stretch")

        st.info(f"View full results on the **Results Summary** page →")
    else:
        st.info("Configure your data and strategy, then click **Run Execution-Aware Backtest** above.")
