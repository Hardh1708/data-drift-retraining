import pandas as pd
from pathlib import Path
from sklearn.utils import resample

# Path to your original file (adjust if needed)
SRC = Path("data/raw/creditcard.csv")  # or Path("creditcard.csv") if it's in root

OUT = Path("creditcard_drifted.csv")

def main():
    df = pd.read_csv(SRC)

    # Use the last 50k rows as a "future" period to drift
    df_future = df.tail(50000).copy()

    # 1) Strong feature drift: scale Amount up
    df_future["Amount"] = df_future["Amount"] * 5.0

    # 2) Class distribution drift: oversample frauds
    fraud = df_future[df_future["Class"] == 1]
    non_fraud = df_future[df_future["Class"] == 0]

    if len(fraud) > 0:
        fraud_oversampled = resample(
            fraud,
            replace=True,
            n_samples=min(len(fraud) * 10, len(non_fraud)),  # boost fraud rate
            random_state=42,
        )
        df_future = pd.concat([non_fraud, fraud_oversampled], axis=0).sample(
            frac=1.0, random_state=42
        )

    df_future.to_csv(OUT, index=False)
    print(f"Saved drifted dataset to: {OUT.resolve()}")

if __name__ == "__main__":
    main()
