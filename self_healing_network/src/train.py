"""
src/train.py
------------
Trains four ML models on the preprocessed dataset:
  1. Random Forest
  2. Support Vector Machine (SVM)
  3. XGBoost
  4. Logistic Regression

Returns a dict of fitted model objects and the train/test splits.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_models() -> dict:
    """
    Returns a dictionary of model name → unfitted model instance.
    All models are configured with sensible defaults for network failure data.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced",   # handles class imbalance
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,          # needed for predict_proba
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    }


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.20):
    """
    Stratified train/test split preserving the failure ratio in both sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )


def train_all(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Fits every model on the training data.

    Returns
    -------
    trained : dict[str, fitted model]
    """
    models = get_models()
    trained = {}

    print("\n  Training models …")
    for name, model in models.items():
        print(f"    ► {name:<22}", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("done ✓")
        trained[name] = model

    return trained
