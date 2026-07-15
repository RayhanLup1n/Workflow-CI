"""
modelling.py — Modelling script for MLflow Project (Workflow CI).

Trains a RandomForest classifier on preprocessed Telco Churn data
and logs metrics + model to MLflow. Designed to run inside a
GitHub Actions CI pipeline.

Usage (via MLflow Project):
    mlflow run MLProject/ -P n_estimators=200 -P max_depth=15
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR  # preprocessed CSVs are in the same folder as this script
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "telco-churn-ci"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(data_dir: str) -> tuple:
    """Load preprocessed train/test data from CSV files."""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
) -> RandomForestClassifier:
    """Train a RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluate & log
# ---------------------------------------------------------------------------
def evaluate_and_log(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluate model and manually log all metrics to MLflow."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "train_roc_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
        "test_roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Print results
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score:  {metrics['test_f1']:.4f}")
    print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run training pipeline with CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=15)
    args = parser.parse_args()

    print(f"Training with n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data(DATA_DIR)

    with mlflow.start_run(run_name="ci-training"):
        model = train_model(X_train, y_train, args.n_estimators, args.max_depth)
        evaluate_and_log(model, X_train, X_test, y_train, y_test)

    print("[DONE] CI training completed.")


if __name__ == "__main__":
    main()
