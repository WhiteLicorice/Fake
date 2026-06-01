"""Compare Cruz and Lupac dataset characteristics with two-sample tests.

Run from the repository root with:
    python training/dataset_characteristics_test.py

The script reads the precomputed per-article feature CSVs under
training/root/datasets/{Cruz,Lupac}, checks Shapiro-Wilk normality for each
dataset group, selects a t-test only when both groups are normal, otherwise
uses Mann-Whitney U, and prints Bonferroni-corrected results for the three
planned comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05
NORMALITY_ALPHA = 0.05
BONFERRONI_TESTS = 3
EXPECTED_N = 3206
DATA_ROOT = Path(__file__).resolve().parent / "root" / "datasets"


@dataclass(frozen=True)
class FeatureSpec:
    title: str
    file_name: str
    column: str


@dataclass(frozen=True)
class Summary:
    mean: float
    sd: float
    n: int


FEATURES = (
    FeatureSpec("OOV word count", "OovFeatures.csv", "count_oov_words"),
    FeatureSpec("Readability index", "ReadFeatures.csv", "readability_score"),
    FeatureSpec("Stop word count", "SwFeatures.csv", "count_stopwords"),
)


def load_values(dataset: str, spec: FeatureSpec) -> np.ndarray:
    path = DATA_ROOT / dataset / spec.file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    data = pd.read_csv(path)
    if spec.column not in data.columns:
        raise ValueError(f"Missing column {spec.column!r} in {path}")

    values = pd.to_numeric(data[spec.column], errors="coerce").dropna().to_numpy(float)
    if len(values) != EXPECTED_N:
        raise ValueError(
            f"{path} has {len(values)} usable rows for {spec.column!r}; "
            f"expected {EXPECTED_N}"
        )
    return values


def summarize(values: np.ndarray) -> Summary:
    return Summary(
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)),
        n=len(values),
    )


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    pooled_variance = (
        ((len(group_a) - 1) * np.var(group_a, ddof=1))
        + ((len(group_b) - 1) * np.var(group_b, ddof=1))
    ) / (len(group_a) + len(group_b) - 2)
    return float((np.mean(group_a) - np.mean(group_b)) / np.sqrt(pooled_variance))


def rank_biserial_from_u(u_statistic: float, n_a: int, n_b: int) -> float:
    return float((2 * u_statistic / (n_a * n_b)) - 1)


def classify_effect(effect: float, effect_type: str) -> str:
    magnitude = abs(effect)
    if effect_type == "cohen_d":
        if magnitude < 0.5:
            return "small"
        if magnitude < 0.8:
            return "medium"
        return "large"

    if magnitude < 0.3:
        return "small"
    if magnitude < 0.5:
        return "medium"
    return "large"


def run_test(cruz: np.ndarray, lupac: np.ndarray) -> tuple[str, float, float, str, float]:
    cruz_normal = stats.shapiro(cruz).pvalue > NORMALITY_ALPHA
    lupac_normal = stats.shapiro(lupac).pvalue > NORMALITY_ALPHA

    # Effects are oriented as Lupac minus Cruz, so positive values mean Lupac
    # tends to have higher values than Cruz.
    if cruz_normal and lupac_normal:
        result = stats.ttest_ind(lupac, cruz, equal_var=False)
        return (
            "t-test",
            float(result.statistic),
            float(result.pvalue),
            "Cohen's d",
            cohen_d(lupac, cruz),
        )

    result = stats.mannwhitneyu(lupac, cruz, alternative="two-sided", method="asymptotic")
    return (
        "Mann-Whitney U",
        float(result.statistic),
        float(result.pvalue),
        "rank-biserial r",
        rank_biserial_from_u(float(result.statistic), len(lupac), len(cruz)),
    )


def print_result(spec: FeatureSpec) -> None:
    cruz = load_values("Cruz", spec)
    lupac = load_values("Lupac", spec)
    cruz_summary = summarize(cruz)
    lupac_summary = summarize(lupac)

    test_name, statistic, raw_p, effect_name, effect = run_test(cruz, lupac)
    corrected_p = min(raw_p * BONFERRONI_TESTS, 1.0)
    effect_type = "cohen_d" if effect_name == "Cohen's d" else "rank_biserial"
    significance = "significant" if corrected_p < ALPHA else "not significant"

    print(spec.title)
    print(
        f"  Cruz (FNF):     mean = {cruz_summary.mean:.1f}, "
        f"SD = {cruz_summary.sd:.1f}, n = {cruz_summary.n}"
    )
    print(
        f"  Lupac (FNF2024): mean = {lupac_summary.mean:.1f}, "
        f"SD = {lupac_summary.sd:.1f}, n = {lupac_summary.n}"
    )
    print(f"  Test: {test_name}")
    print(f"  Statistic: {statistic:.2f}, p = {corrected_p:.4f} (Bonferroni-corrected)")
    print(
        f"  Effect size: {effect_name} = {effect:.2f} "
        f"[{classify_effect(effect, effect_type)}]"
    )
    print(f"  Result: {significance} at α = {ALPHA:.2f}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for index, spec in enumerate(FEATURES):
        if index:
            print()
        print_result(spec)


if __name__ == "__main__":
    main()
