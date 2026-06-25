import pandas as pd
from sklearn.utils import resample
from pathlib import Path

DATA_RAW = Path("data/raw/creditcard.csv")
REF_DIR = Path("data/reference")
PROD_DIR = Path("data/production")

def prepare_splits(ref_frac = 0.6, batch_size = 20000):
    df = pd.read_csv(DATA_RAW)
    df = df.sort_values("Time")

    n_ref =  int(len(df) * ref_frac)
    df_ref = df.iloc[:n_ref]
    df_prod = df.iloc[n_ref:]

    REF_DIR.mkdir(parents=True, exist_ok=True)
    PROD_DIR.mkdir(parents=True, exist_ok=True)

    df_ref.to_csv(REF_DIR / "reference.csv", index=False)

    for i in range(0, len(df_prod), batch_size):
        batch = df_prod.iloc[i:i+batch_size].copy()
        batch_idx = i // batch_size

    # Example: intensify drift in later batches
    if batch_idx >= 3:  # adjust 3 based on how many total batches you have
        batch["Amount"] = batch["Amount"] * 2.0

         # Class distribution drift: oversample frauds
        fraud = batch[batch["Class"] == 1]
        non_fraud = batch[batch["Class"] == 0]

        if len(fraud) > 0:
            fraud_oversampled = resample(
                fraud,
                replace=True,
                n_samples=min(len(fraud) * 3, len(non_fraud)),  # 3x frauds, capped
                random_state=42,
            )
            batch = pd.concat([non_fraud, fraud_oversampled], axis=0).sample(
                frac=1.0, random_state=42
            )

        batch.to_csv(PROD_DIR / f"production_batch_{i//batch_size}.csv", index=False)

if __name__ == "__main__":
    prepare_splits()