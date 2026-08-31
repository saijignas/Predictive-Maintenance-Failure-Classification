"""Model comparison under a train/held-out split, with repeated stratified
CV on the train portion so metrics come with a confidence interval instead
of a single lucky number.

Split discipline:
- 20% held out untouched until the very end (final unbiased read).
- All model selection, and the cost-sensitive threshold in
  cost_sensitive.py, use only the 80% training portion via CV.

Metric: PR-AUC, not ROC-AUC or accuracy. With ~3.4% positive class,
accuracy is meaningless (predicting "no failure" always scores ~96.6%)
and ROC-AUC is optimistic under heavy imbalance; PR-AUC reflects the
precision/recall trade-off that actually matters operationally.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, load_raw
from utils import fmt_ci, mean_ci

ROOT = Path(__file__).parent.parent
TABLE_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def build_models():
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=None,  # set per-fold from the training fold's class ratio
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def cv_compare(X_train, y_train):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    scores = {name: [] for name in build_models()}

    for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X_train, y_train)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
        X_va_s = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns)

        models = build_models()
        pos_ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        models["xgboost"].set_params(scale_pos_weight=pos_ratio)

        for name, model in models.items():
            model.fit(X_tr_s, y_tr)
            proba = model.predict_proba(X_va_s)[:, 1]
            scores[name].append(average_precision_score(y_va, proba))

    return scores


def main():
    df = load_raw(ROOT / "data" / "ai4i2020.csv")
    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} ({y_train.sum()} failures), "
          f"held-out test: {len(X_test)} ({y_test.sum()} failures)")

    cv_scores = cv_compare(X_train, y_train)

    rows = []
    for name, vals in cv_scores.items():
        mean, half_width = mean_ci(vals)
        rows.append({"model": name, "pr_auc_cv": fmt_ci(mean, half_width), "mean": mean})
        print(f"{name}: PR-AUC = {fmt_ci(mean, half_width)} (5x5 repeated stratified CV)")
    cv_table = pd.DataFrame(rows).sort_values("mean", ascending=False)
    cv_table.to_csv(TABLE_DIR / "cv_model_comparison.csv", index=False)

    best_name = cv_table.iloc[0]["model"]
    print(f"\nBest by CV PR-AUC: {best_name}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    best_model = build_models()[best_name]
    if best_name == "xgboost":
        pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        best_model.set_params(scale_pos_weight=pos_ratio)
    best_model.fit(X_train_s, y_train)

    test_proba = best_model.predict_proba(X_test_s)[:, 1]
    test_pr_auc = average_precision_score(y_test, test_proba)
    print(f"Held-out test PR-AUC ({best_name}): {test_pr_auc:.3f}")

    with open(TABLE_DIR / "held_out_result.txt", "w") as f:
        f.write(f"best_model={best_name}\nheld_out_pr_auc={test_pr_auc:.4f}\n")

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    X_test.assign(machine_failure=y_test.values, proba=test_proba).to_csv(
        TABLE_DIR / "test_set_predictions.csv", index=False
    )
    print(f"Saved best model ({best_name}) and test-set predictions.")


if __name__ == "__main__":
    main()
