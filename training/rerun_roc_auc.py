"""Step 9: ROC-AUC on the joint corpus with tuned hyperparameters."""

from __future__ import annotations

import json

import pandas as pd

from rerun_common import (
    CLASSIFIER_ORDER,
    RESULTS_DIR,
    configure_stdout,
    repeated_cv_auc,
    tuned_classifiers,
    write_json,
)


def load_best_params() -> dict | None:
    path = RESULTS_DIR / "tuned_best_params.json"
    if not path.exists():
        print(f"Best-params file not found, using manuscript tuned defaults: {path}")
        return None
    print(f"Loading tuned best params from: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    configure_stdout()
    output_csv = RESULTS_DIR / "roc_auc_30run_raw.csv"
    output_json = RESULTS_DIR / "roc_auc_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("# Step 9: ROC-AUC on Joint Corpus")
    print(
        "Protocol: stratified 80% train partition, then "
        "RepeatedKFold(n_splits=5, n_repeats=6, random_state=42) on the train partition"
    )

    classifiers = tuned_classifiers(load_best_params(), probability_for_svc=True)
    rows = []
    for classifier_id in CLASSIFIER_ORDER:
        rows.extend(repeated_cv_auc(classifier_id, classifiers[classifier_id]))

    raw = pd.DataFrame(rows)
    raw.to_csv(output_csv, index=False)
    summary = (
        raw.groupby(["classifier", "classifier_name"], as_index=False)
        .agg(mean_auc=("auc", "mean"), sd_auc=("auc", lambda x: x.std(ddof=1)))
        .sort_values("classifier")
    )

    print("\nROC-AUC summary:")
    print(summary.to_string(index=False))

    write_json(
        output_json,
        {"raw_csv": output_csv, "summary": summary.to_dict(orient="records")},
    )
    print(f"\nSaved raw CSV: {output_csv}")
    print(f"Saved summary JSON: {output_json}")


if __name__ == "__main__":
    main()
