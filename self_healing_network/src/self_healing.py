"""
src/self_healing.py
-------------------
Implements the failure-reasoning and self-healing simulation layer.

Given a trained model (preferably XGBoost or Random Forest) and a raw
feature vector, this module:
  1. Predicts failure / normal
  2. Identifies the top contributing features
  3. Maps them to human-readable reasons
  4. Selects and executes the correct recovery action
"""

import numpy as np
from src.preprocessing import FEATURE_LABELS

# ── thresholds that define "high" / "low" for each feature ─────────────────
# Values are in the *original* (unscaled) domain.
THRESHOLDS = {
    "latency":          80.0,    # ms   – above = high
    "packet_loss":      0.05,    # ratio – above = high
    "bandwidth":        30.0,    # Mbps – below = low
    "jitter":           15.0,    # ms   – above = high
    "throughput":       30.0,    # Mbps – below = low
    "cpu_usage":        80.0,    # %    – above = high
    "memory_usage":     80.0,    # %    – above = high
    "disk_io":          80.0,    # %    – above = high
    "network_errors":   15.0,    # count – above = high
    "retransmissions":  8.0,     # count – above = high
    "connection_drops": 3.0,     # count – above = high
    "uptime_pct":       95.0,    # %    – below = low
    "serror_rate":      0.3,     # ratio – above = high
    "rerror_rate":      0.3,     # ratio – above = high
    "src_bytes":        50000.0, # bytes – above = high
    "syn_flag_count":   20.0,    # count – above = high
}

# ── healing actions (priority order): ──────────────────────────────────────
# Maps a feature name to a healing action string.
HEALING_ACTIONS = {
    "latency":          "🔀  Rerouting traffic to a lower-latency path",
    "packet_loss":      "🔁  Restarting affected network node",
    "bandwidth":        "🖥️   Switching to backup server / uplink",
    "jitter":           "⚖️   Enabling traffic shaping & QoS policies",
    "throughput":       "🖥️   Switching to backup server / uplink",
    "cpu_usage":        "📦  Migrating workload to spare compute node",
    "memory_usage":     "🧹  Triggering memory reclamation & cache flush",
    "disk_io":          "💾  Redirecting I/O to secondary storage pool",
    "network_errors":   "🔁  Restarting affected network node",
    "retransmissions":  "🔀  Rerouting traffic to a lower-latency path",
    "connection_drops": "🔁  Restarting affected network node",
    "uptime_pct":       "🔄  Initiating automatic failover to standby node",
    "serror_rate":      "🛡️   Activating DDoS-mitigation / rate-limiting",
    "rerror_rate":      "🛡️   Activating firewall rules for rejected traffic",
    "src_bytes":        "🛡️   Enabling traffic scrubbing for anomalous flow",
    "syn_flag_count":   "🛡️   Activating SYN-cookie protection",
}


def _get_top_features(importance_dict: dict,
                      sample_dict: dict,
                      feature_names: list,
                      top_n: int = 3) -> list:
    """
    Rank features by importance score, then keep only those whose
    actual value crosses the threshold (i.e. is genuinely anomalous).

    Returns a list of (feature_name, value, label) tuples.
    """
    # Sort all features by importance descending
    sorted_feats = sorted(importance_dict.items(),
                          key=lambda kv: kv[1], reverse=True)

    results = []
    for feat, _ in sorted_feats:
        if feat not in sample_dict:
            continue
        val = sample_dict[feat]
        thresh = THRESHOLDS.get(feat)
        if thresh is None:
            continue

        # Features where HIGH value is bad (most of them)
        high_bad = feat not in ("bandwidth", "throughput", "uptime_pct")
        anomalous = (high_bad and val > thresh) or \
                    (not high_bad and val < thresh)

        if anomalous:
            label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
            results.append((feat, val, label))
            if len(results) == top_n:
                break

    # Fallback: if no anomalous features found, return top-N by importance
    if not results:
        for feat, _ in sorted_feats[:top_n]:
            val   = sample_dict.get(feat, 0)
            label = FEATURE_LABELS.get(feat, feat)
            results.append((feat, val, label))

    return results


def reason_and_heal(model,
                    X_scaled_sample: np.ndarray,
                    sample_dict: dict,
                    feature_names: list,
                    importance_dict: dict,
                    verbose: bool = True) -> dict:
    """
    Core self-healing function.

    Parameters
    ----------
    model            : fitted sklearn / XGBoost model
    X_scaled_sample  : 1-D scaled feature array (shape: [n_features])
    sample_dict      : original (unscaled) feature values {feat: value}
    feature_names    : list of feature column names
    importance_dict  : {feature: importance_score} from tree model
    verbose          : whether to print to console

    Returns
    -------
    result : dict with keys
        predicted_failure, reasons, healing_actions, confidence
    """
    X = X_scaled_sample.reshape(1, -1)
    prediction = model.predict(X)[0]

    # Confidence / probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[1]) if prediction == 1 else float(proba[0])
    else:
        confidence = None

    if prediction == 1:
        # ── FAILURE PATH ───────────────────────────────────────────────────
        top_feats   = _get_top_features(importance_dict, sample_dict,
                                         feature_names, top_n=3)
        reasons     = [(f, v, l) for f, v, l in top_feats]
        actions     = [HEALING_ACTIONS.get(f, "🔄  Initiating generic recovery")
                       for f, _, _ in top_feats]
        # deduplicate while preserving order
        seen   = set()
        actions = [a for a in actions if not (a in seen or seen.add(a))]

        if verbose:
            print("\n" + "═" * 60)
            print("  ⚠️   NETWORK FAILURE PREDICTED")
            if confidence:
                print(f"       Failure probability : {confidence*100:.1f}%")
            print("═" * 60)
            print("\n  🔍  Root-cause analysis (top contributing factors):\n")
            for i, (feat, val, label) in enumerate(reasons, 1):
                unit = _unit(feat)
                print(f"    {i}. {label}  →  current value: {val:.3f} {unit}")
            print("\n  🛠️   Self-healing actions triggered:\n")
            for a in actions:
                print(f"    {a}")
            print("\n" + "═" * 60)

        return {
            "predicted_failure": True,
            "reasons":           reasons,
            "healing_actions":   actions,
            "confidence":        confidence,
        }
    else:
        # ── NORMAL PATH ────────────────────────────────────────────────────
        if verbose:
            print("\n" + "═" * 60)
            print("  ✅  NETWORK IS STABLE – No action required")
            if confidence:
                print(f"       Normal confidence : {confidence*100:.1f}%")
            print("═" * 60)

        return {
            "predicted_failure": False,
            "reasons":           [],
            "healing_actions":   [],
            "confidence":        confidence,
        }


def simulate_batch(model,
                   X_scaled: np.ndarray,
                   df_raw,
                   feature_names: list,
                   importance_dict: dict,
                   n_samples: int = 5):
    """
    Demonstrate the self-healing system on n_samples random records.
    """
    print("\n" + "─" * 60)
    print(f"  Self-Healing Simulation on {n_samples} random samples")
    print("─" * 60)

    idx = np.random.default_rng(0).choice(len(X_scaled), n_samples, replace=False)
    for sample_i, row_idx in enumerate(idx):
        print(f"\n  ┌── Sample #{sample_i + 1}  (dataset row {row_idx}) ───────────────")
        sample_dict = {col: df_raw.iloc[row_idx][col]
                       for col in feature_names if col in df_raw.columns}
        reason_and_heal(
            model, X_scaled[row_idx], sample_dict,
            feature_names, importance_dict, verbose=True,
        )


# ── formatting helper ───────────────────────────────────────────────────────

def _unit(feat: str) -> str:
    units = {
        "latency": "ms", "jitter": "ms",
        "packet_loss": "(ratio)", "serror_rate": "(ratio)",
        "rerror_rate": "(ratio)",
        "bandwidth": "Mbps", "throughput": "Mbps",
        "cpu_usage": "%", "memory_usage": "%",
        "disk_io": "%", "uptime_pct": "%",
        "src_bytes": "bytes",
    }
    return units.get(feat, "")
