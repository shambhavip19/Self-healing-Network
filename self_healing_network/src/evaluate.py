"""
src/evaluate.py
---------------
Evaluates all trained models and produces:
  • Per-model metrics table (accuracy, precision, recall, F1)
  • Confusion matrix heatmaps
  • Side-by-side bar chart comparing all models
  • Feature importance plots for Random Forest and XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── colour palette ─────────────────────────────────────────────────────────
PALETTE = {
    "Random Forest":      "#2ecc71",
    "SVM":                "#3498db",
    "XGBoost":            "#e74c3c",
    "Logistic Regression":"#f39c12",
}
PLOT_DIR = "plots"


def evaluate_models(trained_models: dict,
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> pd.DataFrame:
    """
    Run predictions for every model and collect metrics.

    Returns
    -------
    results_df : pd.DataFrame with columns
        [Model, Accuracy, Precision, Recall, F1-Score]
    predictions : dict[str, np.ndarray] – raw predictions per model
    """
    rows = []
    predictions = {}

    print("\n  Evaluating models …")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        rows.append({
            "Model":     name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1-Score":  round(f1,   4),
        })

        print(f"\n  ── {name} ──")
        print(classification_report(y_test, y_pred,
                                    target_names=["Normal", "Failure"]))

    return pd.DataFrame(rows), predictions


# ── confusion matrices ──────────────────────────────────────────────────────

def plot_confusion_matrices(trained_models: dict,
                             predictions: dict,
                             y_test: np.ndarray,
                             save_path: str = None):
    """Plot a 2×2 grid of confusion-matrix heatmaps."""
    n = len(trained_models)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Failure"],
            yticklabels=["Normal", "Failure"],
            ax=ax,
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title(f"{name}", fontsize=13, fontweight="bold",
                     color=PALETTE.get(name, "#333"))
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)

    # hide any empty subplot if odd number of models
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices – All Models",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path or f"{PLOT_DIR}/confusion_matrices.png")


# ── model comparison bar chart ──────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame,
                           save_path: str = None):
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1."""
    metrics  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models   = results_df["Model"].tolist()
    x        = np.arange(len(metrics))
    width    = 0.18
    n        = len(models)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(models):
        vals   = results_df.loc[results_df["Model"] == model, metrics].values[0]
        offset = (i - n / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=model, color=PALETTE.get(model, "#888"),
                        edgecolor="white", linewidth=0.6, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_or_show(fig, save_path or f"{PLOT_DIR}/model_comparison.png")


# ── feature importance ──────────────────────────────────────────────────────

def plot_feature_importance(trained_models: dict,
                             feature_names: list,
                             top_n: int = 15,
                             save_path: str = None):
    """
    Horizontal bar charts for Random Forest and XGBoost feature importances.
    Also returns a dict of {feature: importance} for the best tree model.
    """
    tree_models = {k: v for k, v in trained_models.items()
                   if k in ("Random Forest", "XGBoost")}

    fig, axes = plt.subplots(1, len(tree_models),
                              figsize=(7 * len(tree_models), 8))
    if len(tree_models) == 1:
        axes = [axes]

    importance_data = {}

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1][:top_n]
        feat_names  = [feature_names[i] for i in indices]
        feat_vals   = importances[indices]

        importance_data[name] = dict(zip(feature_names,
                                         model.feature_importances_))

        color = PALETTE.get(name, "#888")
        ax.barh(feat_names[::-1], feat_vals[::-1],
                color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(f"{name}\nFeature Importance (top {top_n})",
                     fontsize=13, fontweight="bold", color=color)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Feature Importance – Tree-Based Models",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path or f"{PLOT_DIR}/feature_importance.png")

    return importance_data


# ── util ────────────────────────────────────────────────────────────────────

def print_results_table(results_df: pd.DataFrame):
    """Pretty-print the results table to console."""
    separator = "─" * 70
    print(f"\n{separator}")
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1-Score':>9}")
    print(separator)
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<22} {row['Accuracy']:>9.4f} "
              f"{row['Precision']:>10.4f} {row['Recall']:>8.4f} "
              f"{row['F1-Score']:>9.4f}")
    print(separator)
    best = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
    print(f"\n  🏆  Best model by F1-Score : {best}")
    print(separator)


def _save_or_show(fig, path: str):
    """Save figure to disk; create parent dirs if needed."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Plot saved → {path}")
