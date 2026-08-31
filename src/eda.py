"""Exploratory analysis: class balance, failure-mode breakdown, feature
distributions split by outcome. Writes tables/figures to results/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import LEAKAGE_COLUMNS, RAW_NUMERIC, load_raw

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = load_raw(ROOT / "data" / "ai4i2020.csv")

    n = len(df)
    n_fail = df["Machine failure"].sum()
    print(f"Rows: {n}, failures: {n_fail} ({n_fail / n:.2%})")

    by_type = df.groupby("Type")["Machine failure"].agg(["count", "sum", "mean"])
    by_type.columns = ["n", "failures", "failure_rate"]
    by_type.to_csv(TABLE_DIR / "failure_rate_by_type.csv")
    print("\nFailure rate by product quality variant:")
    print(by_type)

    mode_counts = df[LEAKAGE_COLUMNS].sum().sort_values(ascending=False)
    mode_counts.to_csv(TABLE_DIR / "failure_mode_counts.csv", header=["count"])
    print("\nFailure mode counts (non-exclusive, a row can trigger more than one):")
    print(mode_counts)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, RAW_NUMERIC):
        df[df["Machine failure"] == 0][col].plot(
            kind="density", ax=ax, label="no failure", color="#4c72b0"
        )
        df[df["Machine failure"] == 1][col].plot(
            kind="density", ax=ax, label="failure", color="#c44e52"
        )
        ax.set_title(col)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_distributions_by_outcome.png", dpi=150)
    print(f"\nSaved {FIG_DIR / 'feature_distributions_by_outcome.png'}")

    corr = df[RAW_NUMERIC + ["Machine failure"]].corr()
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(corr.columns, fontsize=7)
    fig2.colorbar(im)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "correlation_matrix.png", dpi=150)
    print(f"Saved {FIG_DIR / 'correlation_matrix.png'}")


if __name__ == "__main__":
    main()
