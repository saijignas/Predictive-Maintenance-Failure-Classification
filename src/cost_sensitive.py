"""Cost-sensitive threshold selection.

A missed failure (false negative) means unplanned downtime and possible
secondary damage; a false alarm (false positive) means an unnecessary
inspection stop. These are not equally bad, so scoring the model at the
default 0.5 cutoff -- or even the F1-optimal cutoff, which still treats
both error types as equally costly -- is the wrong question.

Costs below are illustrative business assumptions (disclosed as such,
not measured data): a missed failure costs 10,000 (downtime + damage), a
false alarm costs 500 (a technician checks a healthy machine).

Threshold-selection discipline: the threshold is picked from out-of-fold
predictions on the training set only (5-fold CV), then applied once,
unchanged, to the held-out test set. The test set is never used to pick
the threshold -- only to report how the already-chosen threshold does.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, load_raw

ROOT = Path(__file__).parent.parent
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"

COST_FALSE_NEGATIVE = 10_000
COST_FALSE_POSITIVE = 500
RANDOM_STATE = 42


def fresh_model(name, pos_ratio):
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_ratio, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    raise ValueError(name)


def expected_cost(y_true, y_pred):
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE


def sweep_thresholds(y_true, proba, thresholds):
    return [expected_cost(y_true, (proba >= t).astype(int)) for t in thresholds]


def main():
    best_name = (TABLE_DIR / "held_out_result.txt").read_text().splitlines()[0].split("=")[1]
    print(f"Selecting cost-optimal threshold for: {best_name}")

    df = load_raw(ROOT / "data" / "ai4i2020.csv")
    X, y = engineer_features(df)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", fresh_model(best_name, pos_ratio)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    oof_costs = sweep_thresholds(y_train.values, oof_proba, thresholds)
    best_idx = int(np.argmin(oof_costs))
    best_threshold = thresholds[best_idx]
    default_cost = expected_cost(y_train.values, (oof_proba >= 0.5).astype(int))
    print(f"Cost-optimal threshold (from training-set CV only): {best_threshold:.2f}")
    print(f"  OOF expected cost @ 0.50: {default_cost:,}")
    print(f"  OOF expected cost @ {best_threshold:.2f}: {oof_costs[best_idx]:,}")

    best_model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    test_proba = best_model.predict_proba(X_test_s)[:, 1]

    test_cost_default = expected_cost(y_test.values, (test_proba >= 0.5).astype(int))
    test_cost_optimal = expected_cost(y_test.values, (test_proba >= best_threshold).astype(int))

    result = pd.DataFrame([
        {"threshold": 0.5, "label": "default", "held_out_expected_cost": test_cost_default},
        {"threshold": best_threshold, "label": "cost-optimal (chosen on train CV)",
         "held_out_expected_cost": test_cost_optimal},
    ])
    result.to_csv(TABLE_DIR / "cost_sensitive_threshold.csv", index=False)
    print(f"\nHeld-out test set, never used for threshold selection:")
    print(result.to_string(index=False))
    savings = test_cost_default - test_cost_optimal
    pct = savings / test_cost_default * 100 if test_cost_default else 0
    print(f"\nSavings from cost-aware thresholding on held-out set: {savings:,.0f} ({pct:.1f}%)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, oof_costs, label="expected cost (train CV, out-of-fold)")
    ax.axvline(best_threshold, color="green", linestyle="--", label=f"chosen threshold = {best_threshold:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="default = 0.50")
    ax.set_xlabel("classification threshold")
    ax.set_ylabel("expected cost ($)")
    ax.set_title("Cost-sensitive threshold selection (training CV only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_sensitive_threshold_curve.png", dpi=150)
    print(f"Saved {FIG_DIR / 'cost_sensitive_threshold_curve.png'}")


if __name__ == "__main__":
    main()
