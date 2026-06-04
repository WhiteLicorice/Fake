"""Step 11: deployment test-article misclassification analysis."""

from __future__ import annotations

import json
import pickle

import pandas as pd

from rerun_common import (
    BASE_DIR,
    RESULTS_DIR,
    TEST_ARTICLES,
    configure_stdout,
    extract_deployment_features_for_text,
    fit_lr_pipeline,
    nonzero_vectorizer_coefficients,
    outcome_from_labels,
    write_json,
)


def load_lr_c() -> float:
    path = RESULTS_DIR / "tuned_best_params.json"
    if not path.exists():
        return 1.0
    params = json.loads(path.read_text(encoding="utf-8"))
    return float(params.get("LR", {}).get("C", 1.0))


def load_or_train_model():
    model_path = BASE_DIR / "models" / "LogisticRegression_stopwords_fix.pkl"
    if model_path.exists():
        print(f"Loading deployment model: {model_path}")
        with model_path.open("rb") as file:
            return pickle.load(file)
    print("Deployment model not found; training raw-text deployment LR for analysis.")
    pipeline, _, _ = fit_lr_pipeline(include_lex_morph=False, C=load_lr_c(), raw_text=True)
    return pipeline


def main() -> None:
    configure_stdout()
    output_json = RESULTS_DIR / "misclassification_analysis.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("# Step 11: Misclassification Analysis")
    print("Model: retrained deployment LR without LEX and MORPH")
    pipeline = load_or_train_model()

    article_rows = []
    linguistic_rows = []
    vectorizer_rows = []
    for article in TEST_ARTICLES:
        predicted = int(pipeline.predict([article["text"]])[0])
        outcome = outcome_from_labels(article["gold_label"], predicted)
        changed = outcome != article["previous_outcome"]
        features = extract_deployment_features_for_text(article["text"])
        vectors = nonzero_vectorizer_coefficients(pipeline, article["text"], limit=4)

        article_rows.append(
            {
                "article_id": article["id"],
                "gold_label": article["gold_label"],
                "predicted_label": predicted,
                "previous_outcome": article["previous_outcome"],
                "new_outcome": outcome,
                "classification_changed": changed,
            }
        )

        feature_row = {"article_id": article["id"], **features}
        linguistic_rows.append(feature_row)

        for direction, rows in vectors.items():
            for row in rows:
                vectorizer_rows.append(
                    {
                        "article_id": article["id"],
                        "direction": direction,
                        **row,
                    }
                )

    article_frame = pd.DataFrame(article_rows)
    linguistic_frame = pd.DataFrame(linguistic_rows)
    vectorizer_frame = pd.DataFrame(vectorizer_rows)

    print("\nClassification outcomes:")
    print(article_frame.to_string(index=False))
    print("\nLinguistic feature values:")
    print(linguistic_frame.to_string(index=False))
    print("\nActive vectorizer predictors, four lowest and four highest coefficients per article:")
    print(vectorizer_frame.to_string(index=False))

    write_json(
        output_json,
        {
            "classification_outcomes": article_rows,
            "linguistic_features": linguistic_rows,
            "vectorizer_predictors": vectorizer_rows,
        },
    )
    print(f"\nSaved JSON: {output_json}")


if __name__ == "__main__":
    main()
