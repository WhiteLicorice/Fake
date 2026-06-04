"""Step 5: train joint-corpus classifiers without hyperparameter tuning."""

from __future__ import annotations

from rerun_common import (
    CLASSIFIER_ORDER,
    RESULTS_DIR,
    configure_stdout,
    default_classifiers,
    evaluate_holdout,
    load_dataset,
    train_partition,
    write_json,
)


def main() -> None:
    configure_stdout()
    output_path = RESULTS_DIR / "train_default_joint.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("# Step 5: Joint-Corpus Training Without Hyperparameter Tuning")
    print("Feature set: vectorizers + READ + OOV + SW + TRAD + SYLL + LEX + MORPH")
    print("Split: train_test_split(test_size=0.2, stratify=y, random_state=42)")

    X, y, dataset_name = load_dataset("Joint", include_lex_morph=True)
    X_train, X_test, y_train, y_test = train_partition(X, y)
    print(f"Dataset: {dataset_name}")
    print(f"Rows: total={len(X)}, train={len(X_train)}, test={len(X_test)}")

    classifiers = default_classifiers()
    results = []
    for classifier_id in CLASSIFIER_ORDER:
        results.append(
            evaluate_holdout(
                classifier_id=classifier_id,
                classifier=classifiers[classifier_id],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                include_lex_morph=True,
            )
        )

    write_json(output_path, {"dataset": dataset_name, "results": results})
    print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
    main()
