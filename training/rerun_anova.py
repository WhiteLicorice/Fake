"""Step 8: ANOVA pipeline from the 30-run accuracy values."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from rerun_common import CLASSIFIER_ORDER, RESULTS_DIR, configure_stdout, write_json


DATASET_ORDER = [
    "Fake News Filipino 2020",
    "Fake News Filipino 2024",
    "Joint corpus",
]
TRIM_PROPORTION = 0.1


def winsorvar(values: np.ndarray, tr: float = TRIM_PROPORTION) -> float:
    data = np.sort(np.asarray(values, dtype=float))
    xbottom = int(np.floor(tr * len(data)))
    xtop = len(data) - xbottom - 1
    winsorized = data.copy()
    winsorized[winsorized <= data[xbottom]] = data[xbottom]
    winsorized[winsorized >= data[xtop]] = data[xtop]
    return float(np.var(winsorized, ddof=1))


def trim_mean(values: np.ndarray, tr: float = TRIM_PROPORTION) -> float:
    data = np.sort(np.asarray(values, dtype=float))
    trim_n = int(np.floor(tr * len(data)))
    if trim_n == 0:
        return float(np.mean(data))
    return float(np.mean(data[trim_n:-trim_n]))


def p_value_fon(kron_matrix: np.ndarray, v: np.ndarray, h: np.ndarray, alpha: float) -> float:
    inv_term = np.linalg.pinv(kron_matrix @ v @ kron_matrix.T)
    r_matrix = v @ kron_matrix.T @ inv_term @ kron_matrix
    diag_r = np.diag(r_matrix)
    a_value = float(np.sum((diag_r**2) / (h.flatten() - 1)))
    df = kron_matrix.shape[0]
    crit_value = stats.chi2.ppf(1 - alpha, df)
    crit_value = crit_value + (crit_value / (2 * df)) * a_value * (
        1 + 3 * crit_value / (df + 2)
    )
    return float(crit_value)


def trimmed_anova(df: pd.DataFrame) -> list[dict]:
    j = len(DATASET_ORDER)
    k = len(CLASSIFIER_ORDER)
    p = j * k
    n = len(df) / p
    if int(n) != n:
        raise ValueError("Trimmed-means ANOVA expects balanced cells")
    n = int(n)
    h = np.full((p, 1), n - 2 * int(np.floor(TRIM_PROPORTION * n)), dtype=float)

    grouped_values = []
    for dataset in DATASET_ORDER:
        for classifier in CLASSIFIER_ORDER:
            values = df[
                (df["dataset"] == dataset) & (df["classifier"] == classifier)
            ]["accuracy"].to_numpy(float)
            if len(values) != n:
                raise ValueError(
                    f"Expected {n} rows for {dataset}/{classifier}, got {len(values)}"
                )
            grouped_values.append(values)

    variances = [
        (n - 1) * winsorvar(values) / ((h[index, 0] - 1) * h[index, 0])
        for index, values in enumerate(grouped_values)
    ]
    v = np.diag(variances)
    trim_means = np.array([trim_mean(values) for values in grouped_values]).reshape(-1, 1)

    ij = np.ones((1, j))
    ik = np.ones((1, k))
    cj = np.eye(j - 1, j)
    for idx in range(j - 1):
        cj[idx, idx + 1] = -1
    ck = np.eye(k - 1, k)
    for idx in range(k - 1):
        ck[idx, idx + 1] = -1

    matrices = {
        "dataset": np.kron(cj, ik),
        "classifier": np.kron(ij, ck),
        "dataset * classifier": np.kron(cj, ck),
    }

    rows = []
    for factor, matrix in matrices.items():
        inv_term = np.linalg.pinv(matrix @ v @ matrix.T)
        statistic = float((trim_means.T @ matrix.T @ inv_term @ matrix @ trim_means).item())
        p_value = 0.999
        for i in range(1, 1000):
            alpha = i / 1000
            if statistic > p_value_fon(matrix, v, h, alpha):
                p_value = alpha
                break
        rows.append(
            {
                "factor": factor,
                "statistic": statistic,
                "p": p_value,
            }
        )
    return rows


def standard_two_way_anova(df: pd.DataFrame) -> list[dict]:
    grand = df["accuracy"].mean()
    a = len(DATASET_ORDER)
    b = len(CLASSIFIER_ORDER)
    n = int(len(df) / (a * b))

    dataset_means = df.groupby("dataset")["accuracy"].mean()
    classifier_means = df.groupby("classifier")["accuracy"].mean()
    cell_means = df.groupby(["dataset", "classifier"])["accuracy"].mean()

    ss_dataset = b * n * sum((dataset_means[dataset] - grand) ** 2 for dataset in DATASET_ORDER)
    ss_classifier = n * a * sum((classifier_means[classifier] - grand) ** 2 for classifier in CLASSIFIER_ORDER)
    ss_interaction = 0.0
    for dataset in DATASET_ORDER:
        for classifier in CLASSIFIER_ORDER:
            ss_interaction += (
                cell_means[(dataset, classifier)]
                - dataset_means[dataset]
                - classifier_means[classifier]
                + grand
            ) ** 2
    ss_interaction *= n

    ss_error = 0.0
    for (dataset, classifier), cell in df.groupby(["dataset", "classifier"]):
        ss_error += float(((cell["accuracy"] - cell["accuracy"].mean()) ** 2).sum())

    df_dataset = a - 1
    df_classifier = b - 1
    df_interaction = (a - 1) * (b - 1)
    df_error = a * b * (n - 1)
    ms_error = ss_error / df_error

    effects = [
        ("dataset", ss_dataset, df_dataset),
        ("classifier", ss_classifier, df_classifier),
        ("dataset * classifier", ss_interaction, df_interaction),
    ]
    rows = []
    for effect, ss_effect, df_effect in effects:
        ms_effect = ss_effect / df_effect
        f_stat = ms_effect / ms_error
        p_value = float(stats.f.sf(f_stat, df_effect, df_error))
        partial_eta = ss_effect / (ss_effect + ss_error)
        rows.append(
            {
                "effect": effect,
                "df_effect": df_effect,
                "df_error": df_error,
                "F": float(f_stat),
                "p": p_value,
                "partial_eta_squared": float(partial_eta),
            }
        )
    return rows


def shapiro_tests(df: pd.DataFrame) -> list[dict]:
    rows = []
    for dataset in DATASET_ORDER:
        for classifier in CLASSIFIER_ORDER:
            values = df[
                (df["dataset"] == dataset) & (df["classifier"] == classifier)
            ]["accuracy"]
            stat, p_value = stats.shapiro(values.to_numpy(float))
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "W": float(stat),
                    "p": float(p_value),
                }
            )
    return rows


def levene_test(df: pd.DataFrame) -> dict:
    groups = [
        df[(df["dataset"] == dataset) & (df["classifier"] == classifier)]["accuracy"].to_numpy(float)
        for dataset in DATASET_ORDER
        for classifier in CLASSIFIER_ORDER
    ]
    stat, p_value = stats.levene(*groups, center="mean")
    return {"W": float(stat), "p": float(p_value)}


def pairwise_tests(df: pd.DataFrame) -> dict[str, list[dict]]:
    by_dataset = []
    for dataset in DATASET_ORDER:
        pairs = list(combinations(CLASSIFIER_ORDER, 2))
        for classifier_a, classifier_b in pairs:
            values_a = df[
                (df["dataset"] == dataset) & (df["classifier"] == classifier_a)
            ]["accuracy"]
            values_b = df[
                (df["dataset"] == dataset) & (df["classifier"] == classifier_b)
            ]["accuracy"]
            result = stats.ttest_ind(values_a, values_b, equal_var=False)
            by_dataset.append(
                {
                    "dataset": dataset,
                    "classifier_a": classifier_a,
                    "classifier_b": classifier_b,
                    "raw_p": float(result.pvalue),
                    "bonferroni_p": float(min(result.pvalue * len(pairs), 1.0)),
                }
            )

    by_classifier = []
    for classifier in CLASSIFIER_ORDER:
        pairs = list(combinations(DATASET_ORDER, 2))
        for dataset_a, dataset_b in pairs:
            values_a = df[
                (df["dataset"] == dataset_a) & (df["classifier"] == classifier)
            ]["accuracy"]
            values_b = df[
                (df["dataset"] == dataset_b) & (df["classifier"] == classifier)
            ]["accuracy"]
            result = stats.ttest_ind(values_a, values_b, equal_var=False)
            by_classifier.append(
                {
                    "classifier": classifier,
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "raw_p": float(result.pvalue),
                    "bonferroni_p": float(min(result.pvalue * len(pairs), 1.0)),
                }
            )
    return {
        "classifier_pairs_within_dataset": by_dataset,
        "dataset_pairs_within_classifier": by_classifier,
    }


def main() -> None:
    configure_stdout()
    input_csv = RESULTS_DIR / "cross_dataset_30run_raw.csv"
    output_json = RESULTS_DIR / "anova_pipeline.json"
    if not input_csv.exists():
        raise FileNotFoundError(f"Run rerun_cross_dataset_eval.py first: {input_csv}")

    print("# Step 8: ANOVA Pipeline")
    print(f"Input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df = df[df["feature_set"] == "full"].copy()

    shapiro = shapiro_tests(df)
    levene = levene_test(df)
    trimmed = trimmed_anova(df)
    standard = standard_two_way_anova(df)
    pairwise = pairwise_tests(df)

    print("\nShapiro-Wilk tests:")
    print(pd.DataFrame(shapiro).to_string(index=False))
    print("\nLevene's test:")
    print(pd.DataFrame([levene]).to_string(index=False))
    print("\nTwo-way ANOVA for trimmed means:")
    print(pd.DataFrame(trimmed).to_string(index=False))
    print("\nStandard two-way ANOVA partial eta-squared:")
    print(pd.DataFrame(standard).to_string(index=False))
    print("\nBonferroni post-hoc classifier pairs within each dataset:")
    print(pd.DataFrame(pairwise["classifier_pairs_within_dataset"]).to_string(index=False))
    print("\nBonferroni post-hoc dataset pairs within each classifier:")
    print(pd.DataFrame(pairwise["dataset_pairs_within_classifier"]).to_string(index=False))

    write_json(
        output_json,
        {
            "shapiro_wilk": shapiro,
            "levene": levene,
            "trimmed_means_anova": trimmed,
            "standard_anova_partial_eta_squared": standard,
            "posthoc": pairwise,
        },
    )
    print(f"\nSaved JSON: {output_json}")


if __name__ == "__main__":
    main()
