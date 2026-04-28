"""Strategy Config page."""
from __future__ import annotations

import streamlit as st

def render():
    st.title("⚙️ Strategy Configuration")

    st.markdown("Configure your trading strategy parameters.")

    strategy_type = st.selectbox(
        "Strategy Type",
        ["momentum", "mean_reversion"],
        help="Momentum = trend-following. Mean Reversion = fade the move when price diverges from mean.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Signal Parameters")
        if strategy_type == "momentum":
            rsi_buy  = st.slider("RSI Buy Threshold (oversold)",  10.0, 50.0, 30.0, step=1.0)
            rsi_sell = st.slider("RSI Sell Threshold (overbought)", 50.0, 90.0, 70.0, step=1.0)
            lb_short = st.slider("Lookback Short (bars)", 2, 20, 5)
            lb_long  = st.slider("Lookback Long (bars)",  5, 60, 20)
            min_diff = st.slider("Min Return Spread", 0.0, 0.05, 0.02, step=0.005)
        else:
            bb_window = st.slider("Bollinger Window (bars)", 10, 50, 20)
            bb_std    = st.slider("BB Std Devs", 1.0, 3.0, 2.0, step=0.1)
            z_entry   = st.slider("Z-Score Entry", -3.0, 0.0, -2.0, step=0.1)
            z_exit    = st.slider("Z-Score Exit", 0.0, 3.0, 0.5, step=0.1)

        st.markdown("### Position Sizing")
        max_pos = st.slider("Max Position (%)", 10, 100, 100, step=5) / 100.0

    with col2:
        st.markdown("### Entry / Exit")
        stop_loss    = st.number_input("Stop Loss (%)",    0.0, 20.0, 2.0, step=0.1) / 100
        take_profit  = st.number_input("Take Profit (%)",   0.0, 50.0, 5.0, step=0.1) / 100
        trail_stop   = st.number_input("Trailing Stop (%)", 0.0, 10.0, 0.0, step=0.1) / 100
        time_exit    = st.number_input("Time Exit (bars)", 0, 100, 0, step=5)

        st.markdown("### Execution")
        capital = st.number_input("Initial Capital ($)", 10_000.0, 10_000_000.0, 1_000_000.0, step=10_000.0)
        commission = st.number_input("Commission (%)", 0.0, 0.5, 0.1, step=0.01) / 100
        slippage = st.number_input("Slippage (bps)", 0, 20, 5, step=1)
        latency  = st.number_input("Latency (ms)", 0, 1000, 100, step=10)

    st.divider()

    # Build params dict for session state
    if strategy_type == "momentum":
        params = {
            "rsi_buy_threshold":  rsi_buy,
            "rsi_sell_threshold": rsi_sell,
            "lookback_short":     lb_short,
            "lookback_long":      lb_long,
            "min_return_diff":    min_diff,
            "max_position_pct":   max_pos,
            "stop_loss_pct":      stop_loss,
            "take_profit_pct":    take_profit,
            "trailing_stop_pct":  trail_stop,
            "time_exit_bars":     time_exit,
        }
    else:
        params = {
            "bb_window":          bb_window,
            "bb_std":             bb_std,
            "z_entry_threshold":  z_entry,
            "z_exit_threshold":    z_exit,
            "max_position_pct":   max_pos,
            "stop_loss_pct":      stop_loss,
            "take_profit_pct":     take_profit,
            "trailing_stop_pct":   trail_stop,
            "time_exit_bars":      time_exit,
        }

    st.session_state["strategy_name"]   = strategy_type
    st.session_state["strategy_params"] = params
    st.session_state["exec_params"] = {
        "initial_capital": capital,
        "commission_pct":  commission,
        "slippage_bps":    slippage,
        "latency_ms":      latency,
    }

    st.markdown("### 📋 Configuration Summary")
    st.json({**params, **st.session_state["exec_params"]}, expanded=False)

    st.caption("Configuration saved! Click **Run Backtest** to execute.")
