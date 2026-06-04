"""Steps 3 and 4: descriptive statistics and Mann-Whitney U tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from rerun_common import BASE_DIR, RESULTS_DIR, configure_stdout, write_json


FEATURES = [
    ("OOV count", "OovFeatures.csv", "count_oov_words", "mean_oov"),
    ("Readability index", "ReadFeatures.csv", "readability_score", "mean_readability"),
    ("Stop word count", "SwFeatures.csv", "count_stopwords", "mean_stopwords"),
]


def load_feature(dataset_key: str, file_name: str, column: str) -> pd.Series:
    path = BASE_DIR / "root" / "datasets" / dataset_key / file_name
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise ValueError(f"Missing {column} in {path}")
    return pd.to_numeric(frame[column], errors="raise")


def rank_biserial_from_u(u_statistic: float, n_a: int, n_b: int) -> float:
    return (2 * u_statistic / (n_a * n_b)) - 1


def main() -> None:
    configure_stdout()
    output_path = RESULTS_DIR / "descriptives_mannwhitney.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("# Steps 3-4: Descriptive Statistics and Mann-Whitney U Tests")
    print("Dataset comparison orientation: Fake News Filipino 2024 versus Fake News Filipino 2020")
    print("Bonferroni alpha for three planned tests: 0.0167")

    descriptives = []
    tests = []
    for feature_name, file_name, column, mean_key in FEATURES:
        cruz = load_feature("Cruz", file_name, column)
        lupac = load_feature("Lupac", file_name, column)

        for dataset_name, values in [
            ("Fake News Filipino 2020", cruz),
            ("Fake News Filipino 2024", lupac),
        ]:
            descriptives.append(
                {
                    "dataset": dataset_name,
                    "feature": feature_name,
                    "mean_key": mean_key,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "n": int(values.shape[0]),
                }
            )

        result = stats.mannwhitneyu(
            lupac,
            cruz,
            alternative="two-sided",
            method="asymptotic",
        )
        raw_p = float(result.pvalue)
        corrected_p = min(raw_p * len(FEATURES), 1.0)
        r = rank_biserial_from_u(float(result.statistic), len(lupac), len(cruz))
        tests.append(
            {
                "feature": feature_name,
                "U": float(result.statistic),
                "p": raw_p,
                "bonferroni_corrected_p": corrected_p,
                "rank_biserial_r": float(r),
                "significant_at_0.0167": bool(raw_p < (0.05 / len(FEATURES))),
            }
        )

    print("\nDescriptive statistics:")
    desc_frame = pd.DataFrame(descriptives)
    print(
        desc_frame.pivot(index="dataset", columns="mean_key", values="mean")
        .reset_index()
        .to_string(index=False)
    )

    print("\nMann-Whitney U tests:")
    test_frame = pd.DataFrame(tests)
    print(
        test_frame[
            ["feature", "U", "p", "bonferroni_corrected_p", "rank_biserial_r", "significant_at_0.0167"]
        ].to_string(index=False)
    )

    write_json(output_path, {"descriptives": descriptives, "mann_whitney": tests})
    print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
    main()
