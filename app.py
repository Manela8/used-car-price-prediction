"""
Streamlit app for Used Car Price Prediction.

Note:
- To avoid "attempted relative import" errors when Streamlit runs the script,
  this file inserts the project root into sys.path at runtime (local dev convenience).
"""

# Shim: make project root importable (helps when Streamlit changes cwd)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.config import DATA_FILE, TARGET_COL
from src.deployment import predict_single, predict_batch
from src.preprocessing import data_load


# ──────────────────────────────────────────────────────────────────
# Cache sample data
# ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return data_load(DATA_FILE)


# ──────────────────────────────────────────────────────────────────
# Input Form Builder
# ──────────────────────────────────────────────────────────────────

def build_input_form(df_sample: pd.DataFrame) -> Dict[str, Any]:
    """
    Dynamically build input widgets based on column types in df_sample.
    Numeric columns → number_input
    Categorical columns → selectbox
    """
    st.subheader("Enter Car Details")
    feature_cols = [c for c in df_sample.columns if c != TARGET_COL]

    input_data: Dict[str, Any] = {}
    for col in feature_cols:
        series = df_sample[col]

        if pd.api.types.is_numeric_dtype(series):
            default = float(series.median()) if not series.isna().all() else 0.0
            input_data[col] = st.number_input(label=col, value=default)
        else:
            options = series.dropna().unique().tolist()
            if not options:
                options = ["Unknown"]
            input_data[col] = st.selectbox(label=col, options=options, index=0)

    return input_data


# ──────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🚗 Used Car Price Prediction")
    st.markdown(
        """
        Enter the car details below to get an estimated market price.
        You can also upload a CSV file for bulk predictions.
        """
    )

    df_sample = load_sample_data()

    # Sidebar snapshot
    st.sidebar.markdown("### 📊 Data Snapshot")
    st.sidebar.dataframe(df_sample.head(5))

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # ── Tab 1: Single Prediction ──────────────────────────────────
    with tab1:
        input_data = build_input_form(df_sample)

        st.write("#### Input Preview")
        st.json(input_data)

        if st.button("Predict Price"):
            try:
                result = predict_single(input_data)
                predicted_price = result["predicted_price"]

                # Main metric
                st.metric(
                    label="Estimated Car Price",
                    value=f"₹ {predicted_price:,.2f}"
                )

                # Friendly message based on price range
                if predicted_price >= 1_000_000:
                    st.success("💎 Premium segment car")
                elif predicted_price >= 500_000:
                    st.info("🚘 Mid-range segment car")
                else:
                    st.warning("🚗 Budget segment car")

            except Exception as exc:
                st.error("Prediction failed — check logs for details.")
                st.write(f"Error: {str(exc)}")

    # ── Tab 2: Batch Prediction ───────────────────────────────────
    with tab2:
        st.subheader("Upload CSV for Batch Predictions")
        st.markdown("CSV must contain the same feature columns used during training.")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)

            st.write("#### Preview of Uploaded Data")
            st.dataframe(uploaded_df.head())

            if st.button("Run Batch Prediction"):
                try:
                    result_df = predict_batch(uploaded_df)

                    st.write("#### Prediction Results (first 10 rows)")
                    st.dataframe(result_df.head(10))

                    # Summary stats on predicted prices
                    st.write("#### Predicted Price Summary")
                    st.dataframe(result_df["predicted_price"].describe().to_frame())

                    # Download button
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv,
                        file_name="car_price_predictions.csv",
                        mime="text/csv",
                    )

                except Exception as exc:
                    st.error("Batch prediction failed — check logs for details.")
                    st.write(f"Error: {str(exc)}")


if __name__ == "__main__":
    main()