"""SHAP explainability on the held-out test set, plus a sanity check:
does the model actually lean on the engineered features that mirror the
dataset's real failure mechanisms (temp_diff, power_w, wear_torque), or
did it find some other, less physically meaningful shortcut?
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, load_raw

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"

RANDOM_STATE = 42
MECHANISM_FEATURES = {"temp_diff", "power_w", "wear_torque"}


def main():
    df = load_raw(ROOT / "data" / "ai4i2020.csv")
    X, y = engineer_features(df)

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    explainer = shap.TreeExplainer(model) if hasattr(model, "get_booster") or hasattr(model, "estimators_") else shap.Explainer(model, X_test_s)
    shap_values = explainer(X_test_s)

    # Binary classifiers return one SHAP value per class; keep the
    # positive ("failure") class for both the summary plot and the table.
    if shap_values.values.ndim == 3:
        shap_values = shap_values[..., 1]

    mean_abs = pd.Series(
        abs(shap_values.values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)
    mean_abs.to_csv(TABLE_DIR / "shap_feature_importance.csv", header=["mean_abs_shap"])
    print("Mean |SHAP| by feature:")
    print(mean_abs)

    top5 = set(mean_abs.head(5).index)
    overlap = top5 & MECHANISM_FEATURES
    print(f"\nMechanism features {MECHANISM_FEATURES} in top-5 by importance: {overlap or 'none'}")
    with open(TABLE_DIR / "shap_sanity_check.txt", "w") as f:
        f.write(f"top5_features={sorted(top5)}\n")
        f.write(f"mechanism_features_in_top5={sorted(overlap)}\n")

    fig = plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_values, show=False, max_display=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_summary.png", dpi=150)
    print(f"\nSaved {FIG_DIR / 'shap_summary.png'}")


if __name__ == "__main__":
    main()
