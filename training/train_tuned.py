"""Step 6: train joint-corpus classifiers with five-fold grid search."""

from __future__ import annotations

from rerun_common import (
    CLASSIFIER_ORDER,
    RESULTS_DIR,
    configure_stdout,
    grid_search_specs,
    load_dataset,
    run_grid_search,
    train_partition,
    write_json,
)


PREVIOUS_BEST = {
    "MNB": {"alpha": 0.1},
    "LR": {"C": 1.0},
    "RF": {"max_depth": 20},
    "SVC": {"C": 0.1, "kernel": "linear"},
}


def main() -> None:
    configure_stdout()
    output_path = RESULTS_DIR / "train_tuned_joint.json"
    best_params_path = RESULTS_DIR / "tuned_best_params.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("# Step 6: Joint-Corpus Training With Hyperparameter Tuning")
    print("Feature set: vectorizers + READ + OOV + SW + TRAD + SYLL + LEX + MORPH")
    print("Grid search: five-fold CV, scoring=accuracy")
    print("Split: train_test_split(test_size=0.2, stratify=y, random_state=42)")

    X, y, dataset_name = load_dataset("Joint", include_lex_morph=True)
    X_train, X_test, y_train, y_test = train_partition(X, y)
    print(f"Dataset: {dataset_name}")
    print(f"Rows: total={len(X)}, train={len(X_train)}, test={len(X_test)}")

    specs = grid_search_specs()
    results = []
    best_params = {}
    for classifier_id in CLASSIFIER_ORDER:
        classifier, params = specs[classifier_id]
        result = run_grid_search(
            classifier_id=classifier_id,
            classifier=classifier,
            params=params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            include_lex_morph=True,
        )
        previous = PREVIOUS_BEST[classifier_id]
        result["previous_best"] = previous
        result["differs_from_previous"] = result["best_params"] != previous
        results.append(result)
        best_params[classifier_id] = result["best_params"]
        print(f"Differs from previous best {previous}: {result['differs_from_previous']}")

    write_json(output_path, {"dataset": dataset_name, "results": results})
    write_json(best_params_path, best_params)
    print(f"\nSaved JSON: {output_path}")
    print(f"Saved best params JSON: {best_params_path}")


if __name__ == "__main__":
    main()
