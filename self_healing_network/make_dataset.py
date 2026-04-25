import pandas as pd
import numpy as np

np.random.seed(42)
n = 3000

data = pd.DataFrame({
    "latency": np.random.uniform(1, 300, n),
    "packet_loss": np.random.uniform(0, 0.5, n),
    "bandwidth": np.random.uniform(1, 150, n),
    "jitter": np.random.uniform(0, 50, n),
    "cpu_usage": np.random.uniform(10, 100, n),
    "memory_usage": np.random.uniform(10, 100, n),
    "disk_io": np.random.uniform(10, 100, n),
    "network_errors": np.random.randint(0, 100, n),
    "throughput": np.random.uniform(10, 150, n),
    "retransmissions": np.random.randint(0, 50, n),
    "connection_drops": np.random.randint(0, 20, n),
    "uptime_pct": np.random.uniform(80, 100, n),
})

prob = (
    0.3 * (data["latency"] > 150) +
    0.3 * (data["packet_loss"] > 0.2) +
    0.2 * (data["cpu_usage"] > 80) +
    0.2 * np.random.rand(n)
)

data["failure"] = (prob > 0.5).astype(int)

data.to_csv("hard_dataset.csv", index=False)

print("✅ Dataset created: hard_dataset.csv")