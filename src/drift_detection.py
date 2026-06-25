import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, Any
from scipy.stats import ks_2samp

REF_PATH = Path("data/reference/reference.csv")
PROD_DIR = Path("data/production")

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size == 0 or actual.size == 0:
        return 0.0
    
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(expected, quantiles)

    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_counts / (expected_counts.sum() + 1e-10)
    actual_percents = actual_counts / (actual_counts.sum() + 1e-10)

    psi_vals = []
    for e, a in zip(expected_percents, actual_percents):
        if e == 0 or a == 0:
            continue
        psi_vals.append((a - e) * np.log(a / e))

    return float(np.sum(psi_vals))

def ks_stat_pvalue(expected: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size == 0 or actual.size == 0:
        return {"ks_stat": 0.0, "p_value": 1.0}

    res = ks_2samp(expected, actual, alternative='two-sided', mode='auto')
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}

def compute_batch_drift(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> Dict[str, Any]:
    refx = ref_df.drop(columns=["Class"])
    curx = prod_df.drop(columns=["Class"])

    numeric_cols = refx.select_dtypes(include=[np.number]).columns

    psi_per_feature:Dict[str, float] = {}
    ks_per_feature:Dict[str, Dict[str, float]] = {}

    for col in numeric_cols:
        ref_col = refx[col].values
        cur_col = curx[col].values

        psi_val = psi(ref_col, cur_col)
        psi_per_feature[col] = psi_val

        ks_res = ks_stat_pvalue(ref_col, cur_col)
        ks_per_feature[col] = ks_res

    psi_values = list(psi_per_feature.values())
    ks_pvalues = [v["p_value"] for v in ks_per_feature.values()]

    mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
    max_psi = float(np.max(psi_values)) if psi_values else 0.0
    mean_ks_pvalue = float(np.mean(ks_pvalues)) if ks_pvalues else 1.0
    min_ks_pvalue = float(np.min(ks_pvalues)) if ks_pvalues else 1.0

    return {
        "psi_per_feature": psi_per_feature,
        "ks_per_feature": ks_per_feature,
        "mean_psi": mean_psi,
        "max_psi": max_psi,
        "mean_ks_pvalue": mean_ks_pvalue,
        "min_ks_pvalue": min_ks_pvalue
    }


def batch_list():
    batches = sorted(PROD_DIR.glob("production_batch_*.csv"))
    print(f"[INFO] found {len(batches)} production batches.")
    return batches

if __name__ == "__main__":
    ref_df = pd.read_csv(REF_PATH)

    for batch in batch_list():
        print(f"\n[INFO] Running PSI + KS drift detection for batch: {batch.name}")
        prod_df = pd.read_csv(batch)

        stats = compute_batch_drift(ref_df, prod_df)
        print(
            f"Mean PSI: {stats['mean_psi']:.4f}, "
            f" Max PSI: {stats['max_psi']:.4f}," 
            f" Mean KS p-value: {stats['mean_ks_pvalue']:.4f},"
            f" Min KS p-value: {stats['min_ks_pvalue']:.4f}"
        )
