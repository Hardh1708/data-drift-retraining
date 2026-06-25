from pathlib import Path
from datetime import datetime
import csv 

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from drift_detection import (
    compute_batch_drift,
    REF_PATH,
    batch_list,
)

# Paths
MODEL_DIR = Path("models")
PREV_DIR = MODEL_DIR / "previous_models"
LOG_PATH = MODEL_DIR / "retrain_log.csv"

# Drift thresholds
DRIFT_MEAN_PSI_THRESHOLD = 0.10   # moderate overall drift[web:10][web:120]
DRIFT_MAX_PSI_THRESHOLD = 0.25    # significant drift on any feature[web:10][web:120][web:126]
DRIFT_KS_P_THRESHOLD = 0.05       # KS p-value < 0.05 => reject same-distribution hypothesis[web:125][web:131]


def load_current_model():
    """Load current model + scaler, if exists."""
    path = MODEL_DIR / "current_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def save_new_model(model, scaler):
    """Archive old model (if any) and save new one as current_model.pkl."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREV_DIR.mkdir(parents=True, exist_ok=True)

    current_path = MODEL_DIR / "current_model.pkl"
    if current_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = PREV_DIR / f"model_{ts}.pkl"
        current_path.replace(archived)

    joblib.dump({"model": model, "scaler": scaler}, current_path)


def gather_training_data(up_to_batch: Path) -> pd.DataFrame:
    """
    Combine reference data and all production batches up to and including up_to_batch.
    This implements a simple cumulative retraining strategy.
    """
    ref_df = pd.read_csv(REF_PATH)
    all_batches = [b for b in batch_list() if b <= up_to_batch]
    batch_dfs = [pd.read_csv(b) for b in all_batches]

    df_all = pd.concat([ref_df] + batch_dfs, axis=0)
    return df_all


def train_model_on_dataframe(df_all: pd.DataFrame):
    """Train a RandomForest model with scaling on the given labeled dataframe."""
    X = df_all.drop(columns=["Class"])
    y = df_all["Class"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_val_proba)
    f1 = f1_score(y_val, y_val_pred)

    return model, scaler, auc, f1


def evaluate_model_on_batch(model_bundle, batch_df: pd.DataFrame):
    """Evaluate current model on a single production batch."""
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]

    X = batch_df.drop(columns=["Class"])
    y = batch_df["Class"]

    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, y_proba)
    f1 = f1_score(y, y_pred)

    return auc, f1


def append_log(
    batch_name: str,
    mean_psi: float,
    max_psi: float,
    mean_ks_p: float,
    min_ks_p: float,
    auc: float,
    f1: float,
    retrained: bool,
):
    """Append one row to retrain_log.csv."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "batch",
                    "mean_psi",
                    "max_psi",
                    "mean_ks_pvalue",
                    "min_ks_pvalue",
                    "auc",
                    "f1",
                    "retrained",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                batch_name,
                mean_psi,
                max_psi,
                mean_ks_p,
                min_ks_p,
                auc,
                f1,
                int(retrained),
            ]
        )


def should_trigger_drift_retrain(mean_psi: float, max_psi: float, min_ks_p: float) -> bool:
    """
    Decide whether to trigger retraining based on drift metrics.

    - If overall drift (mean_psi) is moderate or higher, or any feature's PSI is ≥ 0.25, we retrain.[web:10][web:120][web:126]
    - If KS test p-value for any feature is < 0.05, distributions differ significantly.[web:125][web:131]
    """
    if mean_psi >= DRIFT_MEAN_PSI_THRESHOLD:
        return True
    if max_psi >= DRIFT_MAX_PSI_THRESHOLD:
        return True
    if min_ks_p < DRIFT_KS_P_THRESHOLD:
        return True
    return False


def main():
    ref_df = pd.read_csv(REF_PATH)

    # Ensure we have an initial model. If not, train one on reference only.
    model_bundle = load_current_model()
    if model_bundle is None:
        print("[INFO] No existing model found. Training baseline model on reference data.")
        baseline_df = pd.read_csv(REF_PATH)
        model, scaler, auc, f1 = train_model_on_dataframe(baseline_df)
        save_new_model(model, scaler)
        model_bundle = {"model": model, "scaler": scaler}
        append_log(
            batch_name="baseline_reference",
            mean_psi=0.0,
            max_psi=0.0,
            mean_ks_p=1.0,
            min_ks_p=1.0,
            auc=auc,
            f1=f1,
            retrained=True,
        )
        print(f"[INFO] Baseline model trained. AUC={auc:.4f}, F1={f1:.4f}")

    for batch_path in batch_list():
        print(f"\n[INFO] Processing batch: {batch_path.name}")

        batch_df = pd.read_csv(batch_path)

        # Compute drift metrics vs reference
        drift_stats = compute_batch_drift(ref_df, batch_df)
        mean_psi = drift_stats["mean_psi"]
        max_psi = drift_stats["max_psi"]
        mean_ks_p = drift_stats["mean_ks_pvalue"]
        min_ks_p = drift_stats["min_ks_pvalue"]

        print(
            f"[INFO] Drift for {batch_path.name}: "
            f"mean_psi={mean_psi:.4f}, max_psi={max_psi:.4f}, "
            f"mean_ks_p={mean_ks_p:.4e}, min_ks_p={min_ks_p:.4e}"
        )

        # Evaluate current model on this batch
        model_bundle = load_current_model()
        auc_before, f1_before = evaluate_model_on_batch(model_bundle, batch_df)
        print(
            f"[INFO] Current model performance on {batch_path.name}: "
            f"AUC={auc_before:.4f}, F1={f1_before:.4f}"
        )

        # Decide whether to retrain
        if should_trigger_drift_retrain(mean_psi, max_psi, min_ks_p):
            print("[ALERT] Significant drift detected. Retraining model...")
            df_all = gather_training_data(batch_path)
            model_new, scaler_new, auc_new, f1_new = train_model_on_dataframe(df_all)
            save_new_model(model_new, scaler_new)
            print(
                f"[INFO] New model trained on data up to {batch_path.name}. "
                f"AUC={auc_new:.4f}, F1={f1_new:.4f}"
            )
            append_log(
                batch_name=batch_path.name,
                mean_psi=mean_psi,
                max_psi=max_psi,
                mean_ks_p=mean_ks_p,
                min_ks_p=min_ks_p,
                auc=auc_new,
                f1=f1_new,
                retrained=True,
            )
        else:
            print("[INFO] Drift below thresholds. Keeping current model.")
            append_log(
                batch_name=batch_path.name,
                mean_psi=mean_psi,
                max_psi=max_psi,
                mean_ks_p=mean_ks_p,
                min_ks_p=min_ks_p,
                auc=auc_before,
                f1=f1_before,
                retrained=False,
            )


if __name__ == "__main__":
    main()
