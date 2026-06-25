import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Drift & Retraining Dashboard", layout="wide")

LOGO_PATH = Path("../models/retrain_log.csv")
UPLOAD_SAVE_DIR = Path("../data/production")

def load_log():
    if not LOGO_PATH.exists():
        return None
    df = pd.read_csv(LOGO_PATH)
    if "batch" in df.columns:
        df["batch"] = df["batch"].astype(str)
    return df

def main():
    st.title("Data Drift & Model Retraining Dashboard")

    st.subheader("Upload a new production batch (CSV)")

    uploaded_file = st.file_uploader(
        "Upload a CSV file to treat as a new production batch",
        type=["csv"],
        key="batch_uploader",
    )

    if uploaded_file is not None:
        try:
            new_batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded batch:")
            st.dataframe(new_batch_df.head())

            UPLOAD_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = UPLOAD_SAVE_DIR / f"production_batch_uploaded_{ts}.csv"
            new_batch_df.to_csv(save_path, index=False)

            st.success(f"Batch saved to {save_path.name}. It will be included in the next drift check.")

            st.info(
                "To include this batch in the drift detection and retraining pipeline, please run the retraining script (retrain_pipeline.py) from the command line. "
                "The dashboard will automatically update with new metrics after retraining."
            )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


    log_df = load_log()
    if log_df is None or log_df.empty:
        st.warning("No retraining log found. Please run the retraining pipeline first.")
        return
    
    st.sidebar.header("Filters")
    batch_options = ["All"] + sorted(log_df["batch"].unique().tolist())
    selected_batch = st.sidebar.selectbox("Select Batch", batch_options)

    if selected_batch != "All":
        df_view = log_df[log_df["batch"] == selected_batch]
    else:
        df_view = log_df

    st.subheader("Raw Log Data")
    st.dataframe(df_view, use_container_width=True)

    st.markdown("---")

    st.subheader("Drift metrics over batches (PSI & KS)")
    drift_cols = ["mean_psi", "max_psi", "mean_ks_pvalue", "min_ks_pvalue"]
    drift_existing = [c for c in drift_cols if c in log_df.columns]

    if drift_existing:
        st.line_chart(log_df.set_index("batch")[drift_existing])
    else:
        st.info("No drift metrics found in log.")

    st.subheader("Model performance over batches (AUC & F1)")
    perf_cols = ["auc", "f1"]
    perf_existing = [c for c in perf_cols if c in log_df.columns]

    if perf_existing:
        st.line_chart(log_df.set_index("batch")[perf_existing])
    else:
        st.info("No performance metrics found in log.")

    st.markdown("---")

    st.subheader("Retraining decisions")
    if "retrained" in log_df.columns:
        retrain_df = log_df[log_df["retrained"] == 1]
        if not retrain_df.empty:
            st.write("Batches that triggered retraining:")
            st.table(
                retrain_df[["timestamp","batch", "mean_psi", "max_psi", "mean_ks_pvalue", "min_ks_pvalue", "auc", "f1"]])
        else:
            st.info("No retraining events recorded yet.")
    else:
        st.info("Retraining flag not found in log.")

if __name__ == "__main__":
    main()