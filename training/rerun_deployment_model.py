"""Step 12: retrain and benchmark the deployment Logistic Regression model."""

from __future__ import annotations

import json
import shutil
import time

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from rerun_common import (
    BASE_DIR,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
    REPO_DIR,
    RESULTS_DIR,
    TEST_SIZE,
    all_feature_steps,
    configure_stdout,
    fit_lr_pipeline,
    local_inference_benchmark,
    parameter_count,
    save_pickle,
    write_json,
)


def load_lr_c() -> float:
    path = RESULTS_DIR / "tuned_best_params.json"
    if not path.exists():
        return 1.0
    params = json.loads(path.read_text(encoding="utf-8"))
    return float(params.get("LR", {}).get("C", 1.0))


def load_raw_joint() -> tuple[pd.Series, pd.Series]:
    cruz = pd.read_csv(BASE_DIR / "root" / "datasets" / "Cruz" / "FakeNewsFilipino_Cruz2020.csv")
    lupac = pd.read_csv(BASE_DIR / "root" / "datasets" / "Lupac" / "FakeNewsPhilippines2024_Lupac.csv")
    data = pd.concat([cruz, lupac], ignore_index=True)
    return data["article"], data["label"]


def deployment_cv(c_value: float) -> list[dict]:
    X, y = load_raw_joint()
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    base_classifier = LogisticRegression(
        C=c_value,
        max_iter=2000,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    print("\nDeployment LR 30-run CV")
    splits = list(enumerate(cv.split(X_train), start=1))

    def fit_fold(run_index, train_index, val_index) -> dict:
        repeat = ((run_index - 1) // N_SPLITS) + 1
        fold = ((run_index - 1) % N_SPLITS) + 1
        pipeline = Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(
                        all_feature_steps(
                            include_lex_morph=False,
                            from_csv=False,
                            raw_text=True,
                        )
                    ),
                ),
                ("classifier", clone(base_classifier)),
            ]
        )
        started = time.perf_counter()
        pipeline.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        pred = pipeline.predict(X_train.iloc[val_index])
        accuracy = accuracy_score(y_train.iloc[val_index], pred)
        elapsed = time.perf_counter() - started
        message = (
            f"Fold metric | repeat={repeat} | fold={fold} | "
            f"accuracy={accuracy:.9f} | elapsed_seconds={elapsed:.2f}"
        )
        return {
            "repeat": repeat,
            "fold": fold,
            "accuracy": float(accuracy),
            "elapsed_seconds": elapsed,
            "log": message,
        }

    rows = Parallel(n_jobs=4, pre_dispatch="2*n_jobs")(
        delayed(fit_fold)(run_index, train_index, val_index)
        for run_index, (train_index, val_index) in splits
    )
    for row in rows:
        print(row.pop("log"))
    return rows


def main() -> None:
    configure_stdout()
    output_json = RESULTS_DIR / "deployment_model.json"
    raw_cv_csv = RESULTS_DIR / "deployment_lr_30run_raw.csv"
    local_model_path = BASE_DIR / "models" / "LogisticRegression_stopwords_fix.pkl"
    server_model_path = REPO_DIR / "server" / "root" / "models" / "LogisticRegression.pkl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    c_value = load_lr_c()
    print("# Step 12: Deployment Logistic Regression Model")
    print("Feature set: raw text vectorizers + READ + OOV + SW + TRAD + SYLL")
    print(f"LR C: {c_value}, max_iter=2000")

    cv_rows = deployment_cv(c_value)
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(raw_cv_csv, index=False)
    cv_summary = {
        "mean_accuracy": float(cv_frame["accuracy"].mean()),
        "sd_accuracy": float(cv_frame["accuracy"].std(ddof=1)),
    }
    print("\nDeployment CV summary:")
    print(cv_frame["accuracy"].describe().to_string())

    pipeline, X, y = fit_lr_pipeline(include_lex_morph=False, C=c_value, raw_text=True)
    save_pickle(local_model_path, pipeline)
    shutil.copy2(local_model_path, server_model_path)
    model_size_bytes = server_model_path.stat().st_size
    params = parameter_count(pipeline)

    print(f"\nSaved local deployment model: {local_model_path}")
    print(f"Copied deployment model to server path: {server_model_path}")
    print(f"Model file size bytes: {model_size_bytes}")
    print(f"Model file size MB: {model_size_bytes / (1024 * 1024):.3f}")
    print(f"Parameter count: {params}")

    print("\nLocal inference benchmark:")
    benchmark = local_inference_benchmark(pipeline)
    print(pd.DataFrame(benchmark)[["article_length", "mean_ms", "median_ms", "sd_ms"]].to_string(index=False))

    write_json(
        output_json,
        {
            "lr_C": c_value,
            "feature_set": "raw text vectorizers + READ + OOV + SW + TRAD + SYLL",
            "cv_raw_csv": raw_cv_csv,
            "cv_summary": cv_summary,
            "local_model_path": local_model_path,
            "server_model_path": server_model_path,
            "model_size_bytes": model_size_bytes,
            "model_size_mb": model_size_bytes / (1024 * 1024),
            "parameter_count": params,
            "benchmark": benchmark,
        },
    )
    print(f"\nSaved JSON: {output_json}")


if __name__ == "__main__":
    main()
