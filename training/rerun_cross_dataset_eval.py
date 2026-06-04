"""Step 7: 30-run cross-dataset evaluation for tuned classifiers."""

from __future__ import annotations

import json

import pandas as pd

from rerun_common import (
    CLASSIFIER_ORDER,
    RESULTS_DIR,
    configure_stdout,
    repeated_cv_scores,
    summarize_scores,
    tuned_classifiers,
    write_json,
)


DATASET_ORDER = ["Cruz", "Lupac", "Joint"]


def load_best_params() -> dict | None:
    path = RESULTS_DIR / "tuned_best_params.json"
    if not path.exists():
        print(f"Best-params file not found, using manuscript tuned defaults: {path}")
        return None
    print(f"Loading tuned best params from: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    configure_stdout()
    output_csv = RESULTS_DIR / "cross_dataset_30run_raw.csv"
    output_json = RESULTS_DIR / "cross_dataset_30run_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("# Step 7: Cross-Dataset Evaluation, 30 Runs")
    print(
        "Protocol: stratified 80% train partition, then "
        "RepeatedKFold(n_splits=5, n_repeats=6, random_state=42) on the train partition"
    )

    classifiers = tuned_classifiers(load_best_params())
    rows = []
    for dataset_key in DATASET_ORDER:
        for classifier_id in CLASSIFIER_ORDER:
            rows.extend(
                repeated_cv_scores(
                    classifier_id=classifier_id,
                    classifier=classifiers[classifier_id],
                    dataset_key=dataset_key,
                    include_lex_morph=True,
                    feature_label="full",
                )
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_csv, index=False)
    summary = summarize_scores(rows)
    print("\nMean accuracy summary:")
    print(summary.to_string(index=False))

    write_json(
        output_json,
        {
            "raw_csv": output_csv,
            "summary": summary.to_dict(orient="records"),
        },
    )
    print(f"\nSaved raw CSV: {output_csv}")
    print(f"Saved summary JSON: {output_json}")


if __name__ == "__main__":
    main()
