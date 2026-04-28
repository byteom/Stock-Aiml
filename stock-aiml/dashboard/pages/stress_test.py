"""Stress Test page."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

def render():
    st.title("⚡ Stress Test")

    st.markdown("""
    Run adversarial stress scenarios to expose brittle strategy behaviors.
    The Adversarial Market Generator (AMG) produces worst-case but plausible
    market conditions: **liquidity shocks, correlated selloffs, volatility spikes,
    regime shifts, and fast reversals**.
    """)

    if "backtest_result" not in st.session_state:
        st.warning("⚠️ Run a backtest first to enable stress testing.")
        return

    n_scenarios = st.slider("Number of Scenarios", 10, 100, 50, step=5)

    col_types = st.columns(2)
    with col_types[0]:
        st.markdown("### Scenario Types")
        run_liquidity = st.checkbox("Liquidity Shock",     value=True)
        run_selloff   = st.checkbox("Correlated Selloff", value=True)
        run_volspike  = st.checkbox("Volatility Spike",   value=True)
        run_regime    = st.checkbox("Regime Shift",       value=True)
        run_reversal  = st.checkbox("Fast Reversal",       value=True)
        run_crash     = st.checkbox("Market Crash",        value=True)

    selected_types = []
    if run_liquidity: selected_types.append("liquidity_shock")
    if run_selloff:   selected_types.append("correlated_selloff")
    if run_volspike:  selected_types.append("volatility_spike")
    if run_regime:   selected_types.append("regime_shift")
    if run_reversal:  selected_types.append("fast_reversal")
    if run_crash:    selected_types.append("crash")

    if st.button("🚀 Run Stress Test", type="primary"):
        with st.spinner("Generating adversarial scenarios..."):
            from backend.services.data_loader import DataLoader
            from backend.services.feature_engineering import FeatureEngine
            from backend.services.backtester import Backtester
            from backend.services.execution_model import ExecutionSurrogate
            from backend.services.stress_generator import AdversarialMarketGenerator

            from pathlib import Path as _Path
            # Resolve data.csv from project root
            project_root = _Path(__file__).parents[3]
            data_path = project_root / "data.csv"
            if not data_path.exists():
                data_path = _Path("C:/Users/anwee/Desktop_1/Learning-Season/Stock-Aiml/data.csv")

            loader = DataLoader(data_path)
            loader.load_csv(data_path)
            splits = loader.create_walk_forward_splits(n_splits=3)
            test_df = splits[0]["test"]

            engine = FeatureEngine()
            test_df_fe = engine.transform(test_df)

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

            amg = AdversarialMarketGenerator()
            from backend.services.stress_generator import ScenarioType
            sc_types = [ScenarioType(t) for t in selected_types] if selected_types else []

            scenarios = amg.generate_n_scenarios(test_df_fe, n=n_scenarios, scenario_types=sc_types)

            results = []
            for sc in scenarios:
                df_s = sc.generated_df
                if df_s is None or len(df_s) < 20:
                    continue
                df_fe = engine.transform(df_s)
                try:
                    res = bt.run(
                        df=df_fe,
                        strategy_name=st.session_state.get("strategy_name", "momentum"),
                        strategy_params=st.session_state.get("strategy_params", {}),
                        stochastic_exec=False,
                    )
                    results.append({
                        "scenario":       sc.scenario_type.value,
                        "magnitude":      sc.magnitude,
                        "return":         res.metrics.get("annualized_return", 0),
                        "sharpe":         res.metrics.get("sharpe_ratio", 0),
                        "max_dd":         res.metrics.get("max_drawdown_pct", 0),
                        "cvar":           res.metrics.get("cvar_95", 0),
                        "plausibility":   sc.plausibility_score,
                    })
                except Exception:
                    continue

        if results:
            st.success(f"✅ {len(results)}/{n_scenarios} scenarios completed")
            import pandas as pd
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, width="stretch")

            # Worst-case / Best-case cards
            worst = df_results.loc[df_results["return"].idxmin()]
            best  = df_results.loc[df_results["return"].idxmax()]

            wc, bc, avg = st.columns(3)
            with wc:
                st.error(f"⚠️ Worst Case: {worst['scenario']}")
                st.metric("Return", f"{worst['return']:.2f}%")
                st.metric("Max DD", f"-{abs(worst['max_dd']):.2f}%")
            with bc:
                st.success(f"🏆 Best Case: {best['scenario']}")
                st.metric("Return", f"{best['return']:.2f}%")
                st.metric("Sharpe", f"{best['sharpe']:.3f}")
            with avg:
                st.info("📊 Average Scenario")
                st.metric("Return", f"{df_results['return'].mean():.2f}%")
                st.metric("Sharpe", f"{df_results['sharpe'].mean():.3f}")

            # Return distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_results["return"],
                nbinsx=20,
                marker_color="#F59E0B",
                name="Return Distribution",
            ))
            fig.update_layout(template="plotly_dark", height=300, title="Adversarial Return Distribution")
            st.plotly_chart(fig, width="stretch")

            st.session_state["stress_results"] = results
        else:
            st.error("All scenarios failed. Try reducing the number of scenarios.")
