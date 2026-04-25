"""
main.py
-------
Orchestrates the full Self-Healing Network pipeline:
  1. Generate / load data
  2. Preprocess & merge datasets
  3. Train 4 ML models
  4. Evaluate & visualise results
  5. Extract feature importances
  6. Run self-healing simulation on sample records

Usage:
    python main.py
"""

import os
import sys

# ── make src/ importable ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# ── step 0: generate raw data if not present ──────────────────────────────
DATA_FILES = [
    os.path.join("data", "nsl_kdd_style.csv"),
    os.path.join("data", "cicids_style.csv"),
    os.path.join("data", "server_health.csv"),
]
if not all(os.path.exists(f) for f in DATA_FILES):
    print("  Data files not found – generating them first …\n")
    import generate_data  # noqa: F401  runs as side-effect

# ── imports ────────────────────────────────────────────────────────────────
from src.preprocessing import load_and_preprocess
from src.train          import split_data, train_all
from src.evaluate       import (
    evaluate_models, plot_confusion_matrices,
    plot_model_comparison, plot_feature_importance,
    print_results_table,
)
from src.self_healing   import simulate_batch

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def main():
    print("\n" + "═" * 60)
    print("   🌐  SELF-HEALING NETWORK – Predictive Failure Analysis")
    print("═" * 60)

    # ── 1. preprocess ──────────────────────────────────────────────────────
    X, y, features, scaler, df_raw = load_and_preprocess()

    # ── 2. split ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\n  Train samples : {len(X_train):,}  |  Test samples : {len(X_test):,}")

    # ── 3. train ───────────────────────────────────────────────────────────
    trained_models = train_all(X_train, y_train)

    # ── 4. evaluate ────────────────────────────────────────────────────────
    results_df, predictions = evaluate_models(trained_models, X_test, y_test)
    print_results_table(results_df)

    # ── 5. plots ───────────────────────────────────────────────────────────
    print("\n  Generating plots …")

    plot_confusion_matrices(
        trained_models, predictions, y_test,
        save_path=f"{PLOT_DIR}/confusion_matrices.png",
    )
    plot_model_comparison(
        results_df,
        save_path=f"{PLOT_DIR}/model_comparison.png",
    )
    importance_data = plot_feature_importance(
        trained_models, features,
        top_n=12,
        save_path=f"{PLOT_DIR}/feature_importance.png",
    )

    # ── 6. pick best tree model for self-healing ───────────────────────────
    # Prefer XGBoost; fall back to Random Forest
    heal_model_name = "XGBoost" if "XGBoost" in trained_models else "Random Forest"
    heal_model      = trained_models[heal_model_name]
    imp_dict        = importance_data.get(heal_model_name, {})

    print(f"\n  Using '{heal_model_name}' for self-healing reasoning.")

    # ── 7. self-healing simulation ─────────────────────────────────────────
    simulate_batch(
        heal_model, X_test, df_raw,
        feature_names=features,
        importance_dict=imp_dict,
        n_samples=6,
    )

    print("\n" + "═" * 60)
    print("  ✅  Pipeline complete.  Plots saved to plots/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
