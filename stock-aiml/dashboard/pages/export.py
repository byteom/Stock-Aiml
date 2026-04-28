"""Export page — download reports, CSVs, JSON."""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

def render():
    st.title("📤 Export")

    st.markdown("Download your backtest results in various formats.")

    if "backtest_result" not in st.session_state:
        st.warning("⚠️ Run a backtest first before exporting.")
        return

    result = st.session_state["backtest_result"]
    m      = st.session_state["metrics"]
    trades = st.session_state.get("trades", [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 JSON Report")
        report = {
            "strategy_name": result.strategy_name,
            "split_period":  result.split_period,
            "config":        result.config,
            "metrics":       m,
            "equity_curve":  result.equity_curve,
            "drawdown_curve": result.drawdown_curve,
            "n_trades":      len(trades),
        }
        st.json(report, expanded=False)
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(report, indent=2),
            file_name=f"backtest_{result.strategy_name}.json",
            mime="application/json",
        )

    with col2:
        st.markdown("### 📋 Trade Log (CSV)")
        if trades:
            df = pd.DataFrame(trades)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Trades CSV",
                data=csv,
                file_name=f"trades_{result.strategy_name}.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades to export.")

    st.divider()

    st.markdown("### 📊 Equity & Drawdown Data")
    if "equity_curve" in st.session_state:
        eq_df = pd.DataFrame({
            "bar":      list(range(len(result.equity_curve))),
            "equity":   result.equity_curve,
            "drawdown": result.drawdown_curve,
        })
        st.dataframe(eq_df, width="stretch")
        csv_eq = eq_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Equity CSV",
            data=csv_eq,
            file_name=f"equity_{result.strategy_name}.csv",
            mime="text/csv",
        )

    st.divider()

    st.markdown("### 📈 Quick Stats Copy")
    stats_text = (
        f"Strategy: {result.strategy_name}\n"
        f"Period: {result.split_period}\n"
        f"Annualized Return: {m.get('annualized_return', 0):.2f}%\n"
        f"Sharpe: {m.get('sharpe_ratio', 0):.3f}\n"
        f"Max Drawdown: {m.get('max_drawdown_pct', 0):.2f}%\n"
        f"CVaR (95%): {m.get('cvar_95', 0):.3f}%\n"
        f"Win Rate: {m.get('win_rate', 0):.1f}%\n"
        f"Total Trades: {m.get('total_trades', 0)}\n"
    )
    st.text_area("Stats", value=stats_text, height=180)
    st.caption("Copy these stats for your report or presentation.")
