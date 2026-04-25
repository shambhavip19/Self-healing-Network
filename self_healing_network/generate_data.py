"""
generate_data.py
----------------
Generates three realistic synthetic network datasets that mimic publicly
available datasets (NSL-KDD style, CICIDS-style traffic, server health logs).
Run this once before main.py to populate the data/ folder.
"""

import numpy as np
import pandas as pd
import os

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Dataset 1 – NSL-KDD-style intrusion / anomaly
# ──────────────────────────────────────────────
def make_nsl_kdd(n=5000):
    normal = int(n * 0.65)
    attack = n - normal

    df_normal = pd.DataFrame({
        "duration":       rng.integers(0, 60, normal),
        "protocol_type":  rng.choice([0, 1, 2], normal),          # tcp/udp/icmp
        "src_bytes":      rng.integers(100, 5000, normal),
        "dst_bytes":      rng.integers(50, 4000, normal),
        "land":           rng.integers(0, 2, normal),
        "wrong_fragment": rng.integers(0, 3, normal),
        "urgent":         rng.integers(0, 2, normal),
        "hot":            rng.integers(0, 10, normal),
        "num_failed_logins": rng.integers(0, 2, normal),
        "logged_in":      rng.integers(0, 2, normal),
        "num_compromised": rng.integers(0, 5, normal),
        "count":          rng.integers(1, 511, normal),
        "srv_count":      rng.integers(1, 511, normal),
        "serror_rate":    rng.uniform(0, 0.1, normal),
        "rerror_rate":    rng.uniform(0, 0.1, normal),
        "same_srv_rate":  rng.uniform(0.8, 1.0, normal),
        "diff_srv_rate":  rng.uniform(0, 0.1, normal),
        "failure":        0,
    })

    df_attack = pd.DataFrame({
        "duration":       rng.integers(0, 600, attack),
        "protocol_type":  rng.choice([0, 1, 2], attack),
        "src_bytes":      rng.integers(5000, 100000, attack),
        "dst_bytes":      rng.integers(0, 500, attack),
        "land":           rng.integers(0, 2, attack),
        "wrong_fragment": rng.integers(0, 10, attack),
        "urgent":         rng.integers(0, 5, attack),
        "hot":            rng.integers(5, 30, attack),
        "num_failed_logins": rng.integers(2, 10, attack),
        "logged_in":      rng.integers(0, 2, attack),
        "num_compromised": rng.integers(5, 50, attack),
        "count":          rng.integers(200, 511, attack),
        "srv_count":      rng.integers(1, 50, attack),
        "serror_rate":    rng.uniform(0.5, 1.0, attack),
        "rerror_rate":    rng.uniform(0.3, 1.0, attack),
        "same_srv_rate":  rng.uniform(0, 0.4, attack),
        "diff_srv_rate":  rng.uniform(0.5, 1.0, attack),
        "failure":        1,
    })

    df = pd.concat([df_normal, df_attack], ignore_index=True).sample(frac=1, random_state=SEED)
    path = os.path.join(OUT_DIR, "nsl_kdd_style.csv")
    df.to_csv(path, index=False)
    print(f"[Dataset 1] NSL-KDD-style → {path}  ({len(df)} rows)")
    return df


# ──────────────────────────────────────────────
# Dataset 2 – CICIDS-style network traffic
# ──────────────────────────────────────────────
def make_cicids(n=5000):
    normal = int(n * 0.60)
    attack = n - normal

    def traffic(size, is_attack):
        base_latency  = rng.uniform(50, 150, size)   if is_attack else rng.uniform(1, 50, size)
        base_pkt_loss = rng.uniform(0.1, 0.5, size)  if is_attack else rng.uniform(0, 0.05, size)
        base_bw       = rng.uniform(1, 20, size)      if is_attack else rng.uniform(50, 200, size)
        return pd.DataFrame({
            "flow_duration":        rng.integers(100, 100000, size),
            "fwd_packets":          rng.integers(1, 1000, size),
            "bwd_packets":          rng.integers(1, 500, size),
            "fwd_packet_len_mean":  rng.uniform(20, 1500, size),
            "bwd_packet_len_mean":  rng.uniform(20, 1500, size),
            "flow_bytes_per_sec":   rng.uniform(100, 1e6, size),
            "flow_packets_per_sec": rng.uniform(1, 1000, size),
            "latency":              base_latency,
            "packet_loss":          base_pkt_loss,
            "bandwidth":            base_bw,
            "jitter":               rng.uniform(0.5, 30, size) if is_attack else rng.uniform(0, 5, size),
            "fin_flag_count":       rng.integers(0, 5, size),
            "syn_flag_count":       rng.integers(5, 100, size) if is_attack else rng.integers(0, 5, size),
            "rst_flag_count":       rng.integers(0, 20, size) if is_attack else rng.integers(0, 2, size),
            "failure":              int(is_attack),
        })

    df = pd.concat([traffic(normal, False), traffic(attack, True)], ignore_index=True).sample(frac=1, random_state=SEED)
    path = os.path.join(OUT_DIR, "cicids_style.csv")
    df.to_csv(path, index=False)
    print(f"[Dataset 2] CICIDS-style  → {path}  ({len(df)} rows)")
    return df


# ──────────────────────────────────────────────
# Dataset 3 – Server / node health logs
# ──────────────────────────────────────────────
def make_server_health(n=5000):
    normal = int(n * 0.70)
    failure = n - normal

    def health(size, is_fail):
        return pd.DataFrame({
            "cpu_usage":        rng.uniform(60, 100, size) if is_fail else rng.uniform(5, 60, size),
            "memory_usage":     rng.uniform(70, 100, size) if is_fail else rng.uniform(10, 70, size),
            "disk_io":          rng.uniform(80, 100, size) if is_fail else rng.uniform(5, 60, size),
            "network_errors":   rng.integers(20, 200, size) if is_fail else rng.integers(0, 10, size),
            "throughput":       rng.uniform(1, 30, size) if is_fail else rng.uniform(50, 200, size),
            "latency":          rng.uniform(80, 500, size) if is_fail else rng.uniform(1, 80, size),
            "packet_loss":      rng.uniform(0.05, 0.8, size) if is_fail else rng.uniform(0, 0.05, size),
            "bandwidth":        rng.uniform(1, 30, size) if is_fail else rng.uniform(40, 200, size),
            "jitter":           rng.uniform(10, 100, size) if is_fail else rng.uniform(0, 10, size),
            "retransmissions":  rng.integers(10, 100, size) if is_fail else rng.integers(0, 5, size),
            "connection_drops": rng.integers(5, 50, size) if is_fail else rng.integers(0, 2, size),
            "uptime_pct":       rng.uniform(0, 90, size) if is_fail else rng.uniform(95, 100, size),
            "failure":          int(is_fail),
        })

    df = pd.concat([health(normal, False), health(failure, True)], ignore_index=True).sample(frac=1, random_state=SEED)
    path = os.path.join(OUT_DIR, "server_health.csv")
    df.to_csv(path, index=False)
    print(f"[Dataset 3] Server Health → {path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    print("=" * 55)
    print("  Generating synthetic network datasets …")
    print("=" * 55)
    make_nsl_kdd()
    make_cicids()
    make_server_health()
    print("\n✅  All datasets saved to data/")
