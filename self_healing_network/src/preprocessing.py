"""
src/preprocessing.py
--------------------
Handles all data loading, cleaning, merging and feature engineering.

Steps:
  1. Load the three raw CSV files from data/
  2. Standardise column names
  3. Handle missing values
  4. Engineer common features (latency, packet_loss, bandwidth, jitter, throughput)
  5. Merge into one unified DataFrame
  6. Scale features with StandardScaler
  7. Return X (features), y (target) and the fitted scaler
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

NSL_PATH    = os.path.join(DATA_DIR, "nsl_kdd_style.csv")
CICIDS_PATH = os.path.join(DATA_DIR, "cicids_style.csv")
SERVER_PATH = os.path.join(DATA_DIR, "server_health.csv")

# Human-readable label for each feature (used in self-healing explanations)
FEATURE_LABELS = {
    "latency":           "High latency",
    "packet_loss":       "High packet loss",
    "bandwidth":         "Low bandwidth",
    "jitter":            "High jitter",
    "throughput":        "Low throughput",
    "cpu_usage":         "High CPU usage",
    "memory_usage":      "High memory usage",
    "disk_io":           "High disk I/O",
    "network_errors":    "High network errors",
    "retransmissions":   "High retransmissions",
    "connection_drops":  "Frequent connection drops",
    "serror_rate":       "High SYN error rate",
    "rerror_rate":       "High REJ error rate",
    "src_bytes":         "Excessive source bytes (flood)",
    "syn_flag_count":    "SYN flood detected",
    "uptime_pct":        "Low uptime percentage",
}


# ── helpers ────────────────────────────────────────────────────────────────

def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median; categorical NaNs with mode."""
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def _ensure_columns(df: pd.DataFrame, cols: list, default=0.0) -> pd.DataFrame:
    """Add missing columns filled with a default value."""
    for c in cols:
        if c not in df.columns:
            df[c] = default
    return df


# ── loaders ────────────────────────────────────────────────────────────────

def load_nsl_kdd() -> pd.DataFrame:
    """
    NSL-KDD-style dataset: intrusion/anomaly records.
    We derive network-quality features that are comparable across datasets.
    """
    df = pd.read_csv(NSL_PATH)
    df = _fill_missing(df)

    # Derive proxy network-quality features from raw KDD columns
    # Higher serror_rate / rerror_rate → higher latency proxy
    df["latency"]      = df["serror_rate"] * 300 + df["rerror_rate"] * 200 + 10
    df["packet_loss"]  = (df["serror_rate"] + df["rerror_rate"]) / 2
    df["bandwidth"]    = np.clip(200 - df["src_bytes"] / 500, 1, 200)
    df["jitter"]       = df["diff_srv_rate"] * 50
    df["throughput"]   = np.clip(df["srv_count"] / 5, 1, 200)

    keep = ["latency", "packet_loss", "bandwidth", "jitter", "throughput",
            "serror_rate", "rerror_rate", "src_bytes", "syn_flag_count",
            "failure"]
    df = _ensure_columns(df, keep)
    return df[keep].copy()


def load_cicids() -> pd.DataFrame:
    """
    CICIDS-style traffic dataset: flow-level features with labelled attacks.
    Already contains latency, packet_loss, bandwidth, jitter.
    """
    df = pd.read_csv(CICIDS_PATH)
    df = _fill_missing(df)

    keep = ["latency", "packet_loss", "bandwidth", "jitter", "throughput",
            "syn_flag_count", "rst_flag_count", "flow_bytes_per_sec", "failure"]
    df = _ensure_columns(df, keep)
    return df[keep].copy()


def load_server_health() -> pd.DataFrame:
    """
    Server / node health logs: CPU, memory, disk metrics alongside
    network-quality measurements.
    """
    df = pd.read_csv(SERVER_PATH)
    df = _fill_missing(df)

    keep = ["latency", "packet_loss", "bandwidth", "jitter", "throughput",
            "cpu_usage", "memory_usage", "disk_io",
            "network_errors", "retransmissions",
            "connection_drops", "uptime_pct", "failure"]
    df = _ensure_columns(df, keep)
    return df[keep].copy()


# ── main entry point ────────────────────────────────────────────────────────

def load_and_preprocess():
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_scaled : np.ndarray  – scaled feature matrix
    y        : np.ndarray  – binary failure labels
    features : list[str]   – feature column names
    scaler   : StandardScaler – fitted scaler (needed for new samples)
    df_raw   : pd.DataFrame  – unscaled merged dataframe (for inspection)
    """
    print("─" * 55)
    print("  Loading datasets …")
    print("─" * 55)

    df1 = load_nsl_kdd()
    df2 = load_cicids()
    df3 = load_server_health()

    print(f"  [1] NSL-KDD-style : {len(df1):,} rows, {df1.shape[1]} cols")
    print(f"  [2] CICIDS-style  : {len(df2):,} rows, {df2.shape[1]} cols")
    print(f"  [3] Server Health : {len(df3):,} rows, {df3.shape[1]} cols")

    # ── merge: outer join; missing cells → 0 ──────────────────────────────
    df = pd.concat([df1, df2, df3], ignore_index=True, sort=False)
    df = df.fillna(0)

    # ── shuffle ───────────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Merged dataset    : {len(df):,} rows, {df.shape[1]} cols")
    print(f"  Failure rate      : {df['failure'].mean()*100:.1f}%")
    print("─" * 55)

    # ── feature / target split ────────────────────────────────────────────
    y = df["failure"].values.astype(int)
    X_df = df.drop(columns=["failure"])
    features = list(X_df.columns)

    # ── scale ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    return X_scaled, y, features, scaler, df
