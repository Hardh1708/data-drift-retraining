import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import numpy as np

REF_PATH = Path("data/reference/reference.csv")
MODEL_DIR = Path("models")

def train_baseline():
    df = pd.read_csv(REF_PATH)
    X = df.drop("Class", axis=1)
    Y = df["Class"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle= True, stratify=Y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scale_pos_weight = (len(Y_train) - Y_train.sum()) / Y_train.sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        n_jobs = -1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train_scaled, Y_train)

    Y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(Y_test, Y_test_pred_proba)
    print(f"Baseline AUC: {auc:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_DIR / "current_model.pkl")

if __name__ == "__main__":
        train_baseline()
