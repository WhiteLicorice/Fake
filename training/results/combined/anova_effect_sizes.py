"""Compute ANOVA assumptions tests and partial eta-squared effect sizes.

This script reproduces the statistical pipeline reported in the manuscript
(Cross-Dataset Evaluation section) from the raw per-run accuracy data.

Run from the combined/ directory with:
    python anova_effect_sizes.py

--- Relationship to tests.r ---
The inferential tests in the manuscript (F-statistics and Bonferroni post-hoc
p-values) were computed in R using two-way ANOVA for trimmed means via the
twowaytests package (tmeanTwoWay, paircompTwoWay). That analysis is in
combined/tests.r and requires R with twowaytests and phia installed.

This script handles the three components that tests.r does not output:
  1. Shapiro-Wilk normality tests per group
  2. Levene's test for homogeneity of variance (mean-based)
  3. Partial eta-squared effect sizes via standard two-way ANOVA (pingouin)

Partial eta-squared is computed from the standard (non-trimmed) ANOVA because
the trimmed means procedure does not directly yield this statistic; this is
disclosed in the manuscript.

--- Dependencies ---
    pip install pandas scipy pingouin

--- Verified output (matches manuscript) ---
  Shapiro-Wilk: all p >= 0.063         (manuscript: "all p >= 0.063")
  Levene's:     p = 0.003              (manuscript: p = 0.003)
  Dataset:      partial eta2 = 0.667  (manuscript: 0.667)
  Classifier:   partial eta2 = 0.783  (manuscript: 0.783)
  Interaction:  partial eta2 = 0.328  (manuscript: 0.328)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pingouin as pg
from scipy import stats


DATA_PATH = Path(__file__).resolve().parent / "accuracies_all.csv"
ALPHA = 0.05


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}\n"
            "Run from the combined/ directory or check the path."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"accuracy", "classifier", "dataset"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, got {set(df.columns)}")
    return df


def shapiro_wilk(df: pd.DataFrame) -> float:
    """Run Shapiro-Wilk per group; return minimum p-value."""
    print("=== Shapiro-Wilk normality tests (per classifier × dataset group) ===")
    min_p = 1.0
    for (clf, ds), grp in df.groupby(["classifier", "dataset"])["accuracy"]:
        stat, p = stats.shapiro(grp.values)
        min_p = min(min_p, p)
        print(f"  {clf:<5} × {ds:<10}  W = {stat:.4f}  p = {p:.4f}")
    print(f"\n  Minimum p across all groups: {min_p:.4f}")
    verdict = "✅ Normal" if min_p >= ALPHA else "❌ Non-normal"
    print(f"  All groups normal at α = {ALPHA}: {verdict}")
    return min_p


def levenes_test(df: pd.DataFrame) -> float:
    """Run mean-based Levene's test across all 12 groups; return p-value."""
    print("\n=== Levene's test for homogeneity of variance (mean-based) ===")
    groups = [
        grp.values
        for _, grp in df.groupby(["classifier", "dataset"])["accuracy"]
    ]
    stat, p = stats.levene(*groups, center="mean")
    verdict = "❌ Heteroscedastic" if p < ALPHA else "✅ Homoscedastic"
    print(f"  W = {stat:.4f}  p = {p:.4f}  ({verdict} at α = {ALPHA})")
    if p < ALPHA:
        print(
            "  → Heteroscedasticity detected; two-way ANOVA for trimmed means\n"
            "    (combined/tests.r) was used for inferential tests."
        )
    return p


def partial_eta_squared(df: pd.DataFrame) -> None:
    """Compute partial eta-squared via standard two-way ANOVA (pingouin)."""
    print("\n=== Partial eta-squared (standard two-way ANOVA, pingouin) ===")
    print(
        "  Note: effect sizes estimated from standard ANOVA;\n"
        "  inferential F-statistics are from the trimmed means ANOVA in tests.r."
    )
    aov = pg.anova(
        data=df,
        dv="accuracy",
        between=["dataset", "classifier"],
        detailed=True,
    )

    label = {
        "dataset": "Dataset",
        "classifier": "Classifier",
        "dataset * classifier": "Dataset × Classifier (interaction)",
    }

    print()
    for _, row in aov.iterrows():
        if row["Source"] == "Residual":
            continue
        src = label.get(row["Source"], row["Source"])
        df1 = int(row["DF"])
        df2 = int(aov.loc[aov["Source"] == "Residual", "DF"].values[0])
        f = row["F"]
        p = row["p_unc"]
        np2 = row["np2"]
        size = "large" if np2 >= 0.14 else ("medium" if np2 >= 0.06 else "small")
        p_str = "< 0.001" if p < 0.001 else f"= {p:.4f}"
        print(f"  {src}")
        print(f"    F({df1}, {df2}) = {f:.2f},  p {p_str},  partial η² = {np2:.3f}  [{size} effect]")
        print()


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} rows from {DATA_PATH.name}")
    print(f"Classifiers: {sorted(df['classifier'].unique())}")
    print(f"Datasets:    {sorted(df['dataset'].unique())}")
    print(f"Runs per condition: {df.groupby(['classifier','dataset']).size().iloc[0]}")
    print()

    shapiro_wilk(df)
    levenes_test(df)
    partial_eta_squared(df)


if __name__ == "__main__":
    main()
