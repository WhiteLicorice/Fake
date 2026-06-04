"""Build the consolidated markdown report for the stop-word-fix rerun."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
REPO = BASE.parent
RESULTS = BASE / "results" / "stopwords_fix_rerun"
SW_RESULTS = BASE / "results" / "stopwords_rerun"
REPORT = REPO / "rerun_results.md"
REPORT_COPY = RESULTS / "rerun_results.md"


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def records_table(records: list[dict], headers: list[str], digits: int = 6) -> str:
    return table(
        headers,
        [
            [
                fmt(record.get(header), digits)
                if isinstance(record.get(header), (int, float))
                else record.get(header, "")
                for header in headers
            ]
            for record in records
        ],
    )


def classification_rows(payload: dict, tuned: bool = False) -> list[list[object]]:
    rows: list[list[object]] = []
    for result in payload["results"]:
        report = result["classification_report"]
        params = json.dumps(result.get("best_params", {}), sort_keys=True) if tuned else ""
        accuracy = result.get("holdout_accuracy", result.get("accuracy"))
        for label in ["Fake", "Real"]:
            values = report[label]
            row = [
                result["classifier"],
                params,
                label,
                fmt(values["precision"], 3),
                fmt(values["recall"], 3),
                fmt(values["f1-score"], 3),
                fmt(accuracy, 3),
            ]
            rows.append(row)
    return rows


def confusion_rows(payload: dict, tuned: bool = False) -> list[list[object]]:
    rows: list[list[object]] = []
    for result in payload["results"]:
        cm = result["confusion_matrix"]
        rows.append(
            [
                result["classifier"],
                json.dumps(result.get("best_params", {}), sort_keys=True) if tuned else "",
                cm[0][0],
                cm[0][1],
                cm[1][0],
                cm[1][1],
            ]
        )
    return rows


def params_rows(payload: dict) -> list[list[object]]:
    rows = []
    for result in payload["results"]:
        rows.append(
            [
                result["classifier"],
                json.dumps(result["best_params"], sort_keys=True),
                fmt(result["best_cv_accuracy"], 6),
                "No" if not result.get("differs_from_previous") else "Yes",
            ]
        )
    return rows


def ablation_tables() -> str:
    raw = pd.read_csv(RESULTS / "13_tuned_feature_ablation" / "tuned_feature_ablation_raw.csv")
    summary = (
        raw.groupby(["dataset", "feature_set", "classifier"], as_index=False)["accuracy"]
        .mean()
    )
    feature_order = [
        "Vectorizers (TF-IDF + BOW)",
        "+ Readability (READ)",
        "+ Out-of-vocabulary (OOV)",
        "+ Stop words (SW)",
        "+ Traditional features (TRAD)",
        "+ Syllabic features (SYLL)",
        "+ Lexical features (LEX)",
        "+ Morphological features (MORPH) [full set]",
    ]
    dataset_order = ["Fake News Filipino 2020", "Fake News Filipino 2024", "Joint corpus"]
    blocks = []
    for dataset in dataset_order:
        rows = []
        for feature_set in feature_order:
            cells = []
            for clf in ["MNB", "LR", "RF", "SVC"]:
                value = summary[
                    (summary["dataset"] == dataset)
                    & (summary["feature_set"] == feature_set)
                    & (summary["classifier"] == clf)
                ]["accuracy"].iloc[0]
                cells.append(fmt(value, 4))
            rows.append([feature_set, *cells])
        blocks.append(f"### {dataset}\n\n" + table(["Feature Set", "MNB", "LR", "RF", "SVC"], rows))
    return "\n\n".join(blocks)


def raw_console_output() -> str:
    ordered_logs = [
        SW_RESULTS / "step0_recache_sw.log",
        SW_RESULTS / "step1_verify_sw_extractor.log",
        *sorted(RESULTS.glob("*.log")),
        *sorted(RESULTS.glob("*.err.log")),
    ]
    seen: set[Path] = set()
    parts = []
    for path in ordered_logs:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        rel = path.relative_to(REPO)
        content = path.read_text(encoding="utf-8", errors="replace").rstrip()
        parts.append(f"### {rel.as_posix()}\n\n```text\n{content}\n```")
    return "\n\n".join(parts)


def main() -> None:
    descriptives = load_json("descriptives_mannwhitney.json")
    default = load_json("train_default_joint.json")
    tuned = load_json("train_tuned_joint.json")
    cross = load_json("cross_dataset_30run_summary.json")
    anova = load_json("anova_pipeline.json")
    roc = load_json("roc_auc_summary.json")
    coeff = load_json("lr_coefficients_full.json")
    misc = load_json("misclassification_analysis.json")
    deploy = load_json("deployment_model.json")

    lines: list[str] = ["# Full Re-run Results After Stopwords Fix", ""]

    lines.extend(
        [
            "## Fix Verification",
            "",
            "Step 0 regenerated only `SwFeatures.csv` for Cruz/FNF2020 and Lupac/FNF2024. Both regenerated files matched the original CSVs exactly: `Changed SW rows after regeneration: 0`; `Unified diff contains changes: False`; SW was distinct from OOV in both datasets.",
            "",
            "### Step 0 Side-by-side Samples",
            "",
            "Cruz/FNF2020:",
            table(
                ["row_index", "count_stopwords", "count_oov_words"],
                [[0,117,20],[1,60,7],[2,11,2],[3,28,2],[4,9,2],[5,17,7],[6,14,4],[7,30,1],[8,20,2],[9,85,5]],
            ),
            "",
            "Lupac/FNF2024:",
            table(
                ["row_index", "count_stopwords", "count_oov_words"],
                [[0,132,17],[1,259,43],[2,130,16],[3,154,29],[4,209,29],[5,46,12],[6,138,30],[7,151,21],[8,93,24],[9,46,11]],
            ),
            "",
            "Step 1 verified `StopWordsExtractor(from_csv=True)` matches `csv_count_stopwords` and not `csv_count_oov_words` for the same samples in both datasets.",
            "",
        ]
    )

    desc_by_dataset: dict[str, dict[str, float]] = {}
    for row in descriptives["descriptives"]:
        desc_by_dataset.setdefault(row["dataset"], {})[row["mean_key"]] = row["mean"]
    desc_rows = [
        [
            dataset,
            fmt(values["mean_oov"], 6),
            fmt(values["mean_readability"], 6),
            fmt(values["mean_stopwords"], 6),
        ]
        for dataset, values in desc_by_dataset.items()
    ]
    lines.extend(["## Descriptive Statistics (replaces P10, L202-L208)", "", table(["Dataset", "mean_oov", "mean_readability", "mean_stopwords"], desc_rows), ""])

    mw_rows = [
        [
            row["feature"],
            fmt(row["U"], 2),
            f"{row['p']:.6e}",
            f"{row['bonferroni_corrected_p']:.6e}",
            fmt(row["rank_biserial_r"], 6),
        ]
        for row in descriptives["mann_whitney"]
    ]
    lines.extend(["## Mann-Whitney U Tests (replaces P13, L259-L263)", "", table(["Feature", "U", "p", "Bonferroni p", "rank-biserial r"], mw_rows), ""])

    lines.extend(
        [
            "## Table 2 replacement (no tuning, joint corpus)",
            "",
            table(["Classifier", "Params", "Class", "Precision", "Recall", "F1", "Accuracy"], classification_rows(default)),
            "",
            "## Table 3 replacement (tuned, joint corpus)",
            "",
            table(["Classifier", "Best params", "Best CV accuracy", "Differs from previous prompt value"], params_rows(tuned)),
            "",
            table(["Classifier", "Params", "Class", "Precision", "Recall", "F1", "Accuracy"], classification_rows(tuned, tuned=True)),
            "",
            "## Table 4 replacement (confusion matrices, no tuning)",
            "",
            table(["Classifier", "Params", "Fake->Fake", "Fake->Real", "Real->Fake", "Real->Real"], confusion_rows(default)),
            "",
            "## Table 5 replacement (confusion matrices, tuned)",
            "",
            table(["Classifier", "Params", "Fake->Fake", "Fake->Real", "Real->Fake", "Real->Real"], confusion_rows(tuned, tuned=True)),
            "",
        ]
    )

    cross_rows = [
        [row["dataset"], row["classifier"], fmt(row["mean_accuracy"], 6), fmt(row["sd_accuracy"], 6)]
        for row in cross["summary"]
    ]
    lines.extend(["## Table 6 replacement (30-run accuracies across datasets)", "", table(["Dataset", "Classifier", "Mean accuracy", "SD"], cross_rows), ""])

    lines.extend(["## ANOVA replacement (replaces P12-P13, L246-L271)", ""])
    lines.append("### Shapiro-Wilk")
    lines.append(records_table(anova["shapiro_wilk"], ["dataset", "classifier", "W", "p"], 6))
    lines.extend(["", "### Levene", table(["W", "p"], [[fmt(anova["levene"]["W"], 6), fmt(anova["levene"]["p"], 6)]]), ""])
    lines.append("### Two-way ANOVA for Trimmed Means")
    trimmed_rows = [
        {"effect": row["factor"], "statistic": row["statistic"], "p_value": row["p"]}
        for row in anova["trimmed_means_anova"]
    ]
    lines.append(records_table(trimmed_rows, ["effect", "statistic", "p_value"], 6))
    lines.extend(["", "### Standard Two-way ANOVA Partial Eta Squared"])
    standard_rows = [
        {
            "effect": row["effect"],
            "F": row["F"],
            "p_value": row["p"],
            "partial_eta_squared": row["partial_eta_squared"],
        }
        for row in anova["standard_anova_partial_eta_squared"]
    ]
    lines.append(records_table(standard_rows, ["effect", "F", "p_value", "partial_eta_squared"], 6))
    lines.extend(["", "### Bonferroni: Classifier Pairs Within Dataset"])
    lines.append(records_table(anova["posthoc"]["classifier_pairs_within_dataset"], ["dataset", "group_a", "group_b", "p_raw", "p_bonferroni"], 6))
    lines.extend(["", "### Bonferroni: Dataset Pairs Within Classifier"])
    lines.append(records_table(anova["posthoc"]["dataset_pairs_within_classifier"], ["classifier", "group_a", "group_b", "p_raw", "p_bonferroni"], 6))
    lines.append("")

    roc_rows = [
        [row["classifier"], fmt(row["mean_auc"], 6), fmt(row["sd_auc"], 6)]
        for row in roc["summary"]
    ]
    lines.extend(["## Table 12 replacement (ROC-AUC)", "", table(["Classifier", "Mean AUC", "SD"], roc_rows), ""])

    linguistic_rows = [[row["feature"], fmt(row["coefficient"], 6)] for row in coeff["linguistic"]]
    vector_rows = [[row["feature"], fmt(row["coefficient"], 6)] for row in coeff["top_fake_vectorizers"][:10]]
    vector_rows.extend([[row["feature"], fmt(row["coefficient"], 6)] for row in coeff["top_real_vectorizers"][:10]])
    lines.extend(
        [
            "## Table 9 replacement (linguistic feature coefficients)",
            "",
            table(["Feature", "Coefficient"], linguistic_rows),
            "",
            "## Table 10 replacement (top vectorizer predictors)",
            "",
            table(["Feature", "Coefficient"], vector_rows),
            "",
        ]
    )

    lines.extend(
        [
            "## Tables 13-14 replacement (misclassification analysis)",
            "",
            "Classification outcomes:",
            table(
                ["article_id", "gold_label", "predicted_label", "previous_outcome", "new_outcome", "changed"],
                [
                    [
                        row["article_id"],
                        row["gold_label"],
                        row["predicted_label"],
                        row["previous_outcome"],
                        row["new_outcome"],
                        row["classification_changed"],
                    ]
                    for row in misc["classification_outcomes"]
                ],
            ),
            "",
            "Linguistic feature values:",
            records_table(misc["linguistic_features"], list(misc["linguistic_features"][0].keys()), 6),
            "",
            "Vectorizer predictors:",
            records_table(misc["vectorizer_predictors"], ["article_id", "direction", "feature", "coefficient"], 6),
            "",
        ]
    )

    bench_rows = [
        [row["article_length"], fmt(row["mean_ms"], 3), fmt(row["median_ms"], 3), fmt(row["sd_ms"], 3)]
        for row in deploy["benchmark"]
    ]
    lines.extend(
        [
            "## Table 8 replacement (inference benchmark)",
            "",
            table(["Article length", "Mean ms", "Median ms", "SD ms"], bench_rows),
            "",
            "## Deployment model",
            "",
            table(
                ["CV mean accuracy", "CV SD", "Model file size bytes", "Model file size MB", "Parameter count"],
                [[fmt(deploy["cv_summary"]["mean_accuracy"], 6), fmt(deploy["cv_summary"]["sd_accuracy"], 6), deploy["model_size_bytes"], fmt(deploy["model_size_mb"], 3), deploy["parameter_count"]]],
            ),
            "",
            "## Table 7 replacement (tuned ablation, ALL three datasets)",
            "",
            ablation_tables(),
            "",
        ]
    )

    changes = [
        "Step 0, Cruz SwFeatures.csv: diff present -> no diff; changed rows 0.",
        "Step 0, Lupac SwFeatures.csv: diff present -> no diff; changed rows 0.",
        "Server `StopWordsExtractor.from_csv`: `count_oov_words` source -> `count_stopwords` source.",
        "Table 2, MNB, confusion matrix Fake->Real: 537 -> 538.",
        "Table 2, LR, confusion matrix Real->Fake: 51 -> 50.",
        "Table 2, RF, confusion matrix Fake->Real: 66 -> 62; Real->Fake: 59 -> 49; accuracy approximately 0.90 -> 0.913.",
        "Table 2, SVC, confusion matrix Fake->Real: 169 -> 158; Real->Fake: 70 -> 71; accuracy approximately 0.81 -> 0.822.",
        "Table 3, MNB, best alpha in manuscript text: 0.01 -> 0.1; holdout accuracy 0.865 -> 0.861.",
        "Table 3, RF, best params: max_depth=20 -> max_depth=20, min_samples_split=2, n_estimators=100; holdout accuracy 0.889 -> 0.892.",
        "Table 3, SVC, confusion matrix Fake->Real: 42 -> 41; Real->Fake: 51 -> 53; holdout accuracy 0.928 -> 0.927.",
        "Table 6, LR, FNF2020 accuracy: 0.951 -> 0.952288.",
        "Table 6, LR, FNF2024 accuracy: 0.947 -> 0.945462.",
        "Table 6, LR, Combined accuracy: 0.924 -> 0.923084.",
        "Table 6, MNB, FNF2020 accuracy: 0.923 -> 0.920504.",
        "Table 6, MNB, FNF2024 accuracy: 0.885 -> 0.881371.",
        "Table 6, MNB, Combined accuracy: 0.860 -> 0.856337.",
        "Table 6, RF, Combined accuracy: 0.888 -> 0.890523.",
        "Table 6, SVC, Combined accuracy: 0.922 -> 0.922726.",
        "Table 12, MNB ROC-AUC: 0.925 +/- 0.010 -> 0.921877 +/- 0.009800.",
        "Table 12, LR ROC-AUC: 0.976 +/- 0.005 -> 0.975503 +/- 0.004597.",
        "Table 12, RF ROC-AUC: 0.960 +/- 0.006 -> 0.960274 +/- 0.005898.",
        "Table 12, SVC ROC-AUC: 0.973 +/- 0.005 -> 0.973595 +/- 0.005171.",
        "Table 9, count-stopwords coefficient: 0.025210022216103887 -> -0.063949.",
        "Table 9, count-oov-words coefficient: 0.025210022216103887 -> 0.045380.",
        "Table 13, false positive article outcome: FP -> TP.",
        "Table 13, false positive readability score: 14.673 -> 14.028.",
        "Table 13, false negative word_count: 109 -> 110; count_oov_words 10 -> 8.",
        "Table 8/Deployment model, CV accuracy: 0.92 -> 0.923668 +/- 0.008531.",
        "Deployment model size: 55.64 MB -> 66.036 MB.",
        "Deployment parameter count: 1,200,000 -> 1,451,808.",
        "Table 7, ablation protocol: default-hyperparameter joint-only table -> tuned-hyperparameter tables for FNF2020, FNF2024, and joint corpus.",
    ]
    lines.extend(["## Changes summary", "", *[f"- {item}" for item in changes], ""])

    lines.extend(["## Raw console output", "", raw_console_output(), ""])

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    REPORT_COPY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {REPORT_COPY}")


if __name__ == "__main__":
    main()
