"""Step 10: Logistic Regression coefficients for linguistic/vectorizer features."""

from __future__ import annotations

import json

import pandas as pd

from rerun_common import (
    RESULTS_DIR,
    coefficients_for_pipeline,
    configure_stdout,
    feature_names_by_kind,
    fit_lr_pipeline,
    write_json,
)


def load_lr_c() -> float:
    path = RESULTS_DIR / "tuned_best_params.json"
    if not path.exists():
        return 1.0
    params = json.loads(path.read_text(encoding="utf-8"))
    return float(params.get("LR", {}).get("C", 1.0))


def main() -> None:
    configure_stdout()
    output_json = RESULTS_DIR / "lr_coefficients_full.json"
    linguistic_csv = RESULTS_DIR / "lr_linguistic_coefficients_full.csv"
    vectorizer_csv = RESULTS_DIR / "lr_vectorizer_coefficients_full.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    c_value = load_lr_c()
    print("# Step 10: Logistic Regression Coefficients")
    print("Training LR once on the full joint corpus.")
    print("Feature set: vectorizers + READ + OOV + SW + TRAD + SYLL + LEX + MORPH")
    print(f"LR C: {c_value}, max_iter=2000")

    pipeline, X, y = fit_lr_pipeline(include_lex_morph=True, C=c_value, raw_text=False)
    coef_frame = coefficients_for_pipeline(pipeline)
    all_names, vectorizer_names, linguistic_names = feature_names_by_kind(pipeline)
    vectorizer = coef_frame[coef_frame["feature"].isin(vectorizer_names)].copy()
    linguistic = coef_frame[coef_frame["feature"].isin(linguistic_names)].copy()

    linguistic = linguistic.sort_values("coefficient")
    vectorizer = vectorizer.sort_values("coefficient")
    linguistic.to_csv(linguistic_csv, index=False)
    vectorizer.to_csv(vectorizer_csv, index=False)

    top_fake = vectorizer.head(20)
    top_real = vectorizer.tail(20).sort_values("coefficient", ascending=False)

    print(f"Rows trained on: {len(X)}")
    print(f"Total coefficients: {len(all_names)}")
    print(f"Vectorizer coefficients: {len(vectorizer_names)}")
    print(f"Linguistic coefficients: {len(linguistic_names)}")
    print("\nLinguistic coefficients:")
    print(linguistic.to_string(index=False))
    print("\nTop 20 vectorizer predictors for Fake (most negative):")
    print(top_fake.to_string(index=False))
    print("\nTop 20 vectorizer predictors for Real (most positive):")
    print(top_real.to_string(index=False))

    write_json(
        output_json,
        {
            "lr_C": c_value,
            "rows": len(X),
            "total_coefficients": len(all_names),
            "linguistic_csv": linguistic_csv,
            "vectorizer_csv": vectorizer_csv,
            "linguistic": linguistic.to_dict(orient="records"),
            "top_fake_vectorizers": top_fake.to_dict(orient="records"),
            "top_real_vectorizers": top_real.to_dict(orient="records"),
        },
    )
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved linguistic CSV: {linguistic_csv}")
    print(f"Saved vectorizer CSV: {vectorizer_csv}")


if __name__ == "__main__":
    main()
