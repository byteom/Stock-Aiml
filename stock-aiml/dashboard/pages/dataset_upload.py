"""Dataset Upload page."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import time
from pathlib import Path

def render():
    st.title("📁 Dataset Upload")

    st.markdown("""
    Upload your OHLCV market data (CSV format) to use in backtests.

    **Expected columns:**
    - `timestamp` / `date` — datetime
    - `open`, `high`, `low`, `close` — price (required)
    - `volume` — trade volume (optional, synthesized if missing)
    - `spread` — bid-ask spread (optional)
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload OHLCV data. Volume is optional — synthesized if missing.",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Sample Data**")
        sample_path = Path(__file__).parents[3] / "data.csv"
        if sample_path.exists():
            st.success(f"Found: `{sample_path.name}`")
            try:
                sample_df = pd.read_csv(sample_path)
                st.dataframe(sample_df.head(10), width="stretch")
                st.caption(f"Shape: {sample_df.shape}")
            except Exception as e:
                st.error(f"Error reading sample: {e}")
        else:
            st.info("No sample data found in project root.")

    with col2:
        if uploaded_file:
            st.markdown("**Upload Preview**")
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(10), width="stretch")
                st.caption(f"Shape: {df.shape}")

                # Validate columns
                required = {"open", "high", "low", "close"}
                cols_lower = {c.lower().strip() for c in df.columns}
                missing = required - cols_lower
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    st.success("✅ All required columns present")

                # Save to session state
                st.session_state["uploaded_data"] = df
                st.session_state["data_path"] = None

                # Show date range
                date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                if date_cols:
                    st.info(f"Date range: {df[date_cols[0]].min()} → {df[date_cols[0]].max()}")

            except Exception as e:
                st.error(f"Error reading file: {e}")

    st.divider()
    st.markdown("### 📌 Data Schema")
    st.markdown("""
    | Column | Type | Description |
    |---|---|---|
    | `timestamp` | datetime | Bar timestamp (UTC) |
    | `open` | float | Opening price |
    | `high` | float | High price |
    | `low` | float | Low price |
    | `close` | float | Closing price |
    | `volume` | float | Trade volume (optional) |
    | `spread` | float | Bid-ask spread (optional) |
    """)

    st.info("💡 The system automatically synthesizes missing volume from intra-bar range (high−low).")

    if st.button("✅ Use This Dataset", disabled=uploaded_file is None):
        st.success("Dataset loaded! Go to **Strategy Config** to configure your strategy.")
        st.session_state["data_ready"] = True
        st.rerun()
