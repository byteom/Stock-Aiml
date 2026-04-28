"""Explanation page — counterfactual and XAI outputs."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

def render():
    st.title("🔍 Explainability")

    st.markdown("""
    Understand **why** your strategy succeeded or failed, and get
    **actionable counterfactual recommendations**.
    """)

    if "backtest_result" not in st.session_state:
        st.warning("⚠️ Run a backtest first to generate explanations.")
        return

    target_metric = st.selectbox(
        "Target Metric to Optimize",
        ["max_drawdown", "sharpe_ratio", "cvar_95"],
        help="Counterfactuals will optimize improvements for this metric.",
    )

    if st.button("🔍 Generate Counterfactual Analysis"):
        with st.spinner("Running counterfactual engine..."):
            from backend.services.counterfactual import CounterfactualEngine, SurrogateModel
            from backend.services.backtester import BacktestResult

            result = st.session_state["backtest_result"]
            engine = CounterfactualEngine(
                surrogate_model=SurrogateModel(),
                n_counterfactuals=10,
            )
            cf = engine.analyze(result, target_metric=target_metric)
            st.session_state["cf_result"] = cf

        st.success("✅ Counterfactual analysis complete")

        # Summary
        st.markdown("### 📝 Summary")
        st.info(cf.summary)

        # Interventions table
        st.markdown("### 🎯 Recommended Actions")
        interventions = cf.interventions
        if interventions:
            import pandas as pd
            rows = []
            for iv in interventions:
                rows.append({
                    "Rank":         iv.rank,
                    "Variable":     iv.variable,
                    "From":         f"{iv.original_value:.4f}",
                    "To":           f"{iv.counterfactual_value:.4f}",
                    "Change":       f"{iv.delta:+.4f}",
                    "New Drawdown": f"{iv.expected_metric.get('max_drawdown', 0):.2f}%",
                    "Improvement":  f"{iv.improvement * 100:+.2f}%",
                    "Actionable":   "✅" if iv.actionable else "❌",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")

            # Feature importance
            st.markdown("### 📊 Feature Importance (Surrogate Model)")
            fi = cf.feature_importance
            fig = go.Figure(go.Bar(
                x=list(fi.keys()),
                y=list(fi.values()),
                marker_color="#3B82F6",
            ))
            fig.update_layout(
                template="plotly_dark",
                height=300,
                title="Operational Variable Impact on Max Drawdown",
                xaxis_title="Variable",
                yaxis_title="Importance",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No actionable improvements found within the parameter ranges.")

        # Baseline metrics
        st.markdown("### 📋 Baseline Metrics")
        bm = cf.baseline_metrics
        cols = st.columns(4)
        for i, (k, v) in enumerate(bm.items()):
            if isinstance(v, (int, float)):
                with cols[i % 4]:
                    st.metric(k, f"{v:.3f}")
