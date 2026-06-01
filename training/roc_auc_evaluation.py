"""Evaluate ROC-AUC for the four classifiers on the combined corpus.

This script retrains MNB, Logistic Regression, Random Forest, and SVC with
the tuned hyperparameters used in the original combined-dataset evaluation.
It loads only the cached feature CSVs under root/datasets/Cruz and
root/datasets/Lupac, runs six repetitions of five-fold cross-validation, and
saves a representative ROC curve plot for fold 1, run 1.

Dependencies are listed in training/requirements.txt. Run from the repository
root with:

    training\\venv\\Scripts\\python.exe training\\roc_auc_evaluation.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from root.scripts.BPE import BPETokenizer  # noqa: E402
from root.scripts.FILTRANS import (  # noqa: E402
    LEXExtractor,
    MORPHExtractor,
    OOVExtractor,
    READExtractor,
    SYLLExtractor,
    TRADExtractor,
    StopWordsExtractor,
)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.feature_extraction.text"
)

LOAD_FROM_CSV = True
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 6
PLOT_PATH = Path("results") / "combined" / "roc_auc_curves.png"

DATASETS = {
    "Cruz": {
        "articles": "FakeNewsFilipino_Cruz2020.csv",
        "features": [
            "TradFeatures.csv",
            "SyllFeatures.csv",
            "OovFeatures.csv",
            "SwFeatures.csv",
            "ReadFeatures.csv",
            "LexFeatures.csv",
            "MorphFeatures.csv",
        ],
    },
    "Lupac": {
        "articles": "FakeNewsPhilippines2024_Lupac.csv",
        "features": [
            "TradFeatures.csv",
            "SyllFeatures.csv",
            "OovFeatures.csv",
            "SwFeatures.csv",
            "ReadFeatures.csv",
            "LexFeatures.csv",
            "MorphFeatures.csv",
        ],
    },
}


def fail_if_missing_or_empty() -> None:
    missing_or_empty: list[Path] = []
    for dataset, files in DATASETS.items():
        dataset_dir = Path("root") / "datasets" / dataset
        paths = [dataset_dir / files["articles"]]
        paths.extend(dataset_dir / feature_file for feature_file in files["features"])
        for path in paths:
            if not path.exists() or path.stat().st_size == 0:
                missing_or_empty.append(path)

    if missing_or_empty:
        print("Missing or empty required feature inputs:", file=sys.stderr)
        for path in missing_or_empty:
            print(f"  {path}", file=sys.stderr)
        raise SystemExit(1)


def read_non_empty_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV has no rows: {path}")
    return frame


def load_dataset(dataset: str) -> pd.DataFrame:
    dataset_dir = Path("root") / "datasets" / dataset
    files = DATASETS[dataset]

    data = read_non_empty_csv(dataset_dir / files["articles"])
    feature_frames = [
        read_non_empty_csv(dataset_dir / feature_file)
        for feature_file in files["features"]
    ]

    expected_rows = len(data)
    mismatched = [
        feature_file
        for feature_file, frame in zip(files["features"], feature_frames)
        if len(frame) != expected_rows
    ]
    if mismatched:
        details = ", ".join(mismatched)
        raise ValueError(
            f"{dataset} feature row count mismatch against articles: {details}"
        )

    return pd.concat([data, *feature_frames], axis=1)


def load_combined_data() -> tuple[pd.DataFrame, pd.Series]:
    fail_if_missing_or_empty()
    data_cruz = load_dataset("Cruz")
    data_lupac = load_dataset("Lupac")
    data = pd.concat([data_cruz, data_lupac], ignore_index=True)

    y = data["label"]
    X = data.drop("label", axis=1)
    return X, y


def make_feature_space() -> list[tuple[str, object]]:
    return [
        (
            "vectorizers",
            ColumnTransformer(
                transformers=[
                    ("bow", CountVectorizer(), "article"),
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 3), tokenizer=BPETokenizer().tokenize
                        ),
                        "article",
                    ),
                ]
            ),
        ),
        ("read", READExtractor(from_csv=LOAD_FROM_CSV)),
        ("oov", OOVExtractor(from_csv=LOAD_FROM_CSV)),
        ("sw", StopWordsExtractor(from_csv=LOAD_FROM_CSV)),
        ("trad", TRADExtractor(from_csv=LOAD_FROM_CSV)),
        ("syll", SYLLExtractor(from_csv=LOAD_FROM_CSV)),
        ("lex", LEXExtractor(from_csv=LOAD_FROM_CSV)),
        ("morph", MORPHExtractor(from_csv=LOAD_FROM_CSV)),
    ]


def make_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FeatureUnion(make_feature_space())),
            ("classifier", classifier),
        ]
    )


def classifiers() -> list[tuple[str, object]]:
    return [
        ("MNB", MultinomialNB(alpha=0.1)),
        (
            "LR",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "RF",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "SVC",
            SVC(
                C=0.1,
                kernel="linear",
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def positive_class_scores(pipeline: Pipeline, X_val: pd.DataFrame) -> np.ndarray:
    classes = pipeline.named_steps["classifier"].classes_
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return pipeline.predict_proba(X_val)[:, positive_index]


def evaluate_classifier(
    name: str,
    classifier: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[list[float], tuple[pd.Series, np.ndarray, float]]:
    pipeline = make_pipeline(classifier)
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    aucs: list[float] = []
    representative_curve: tuple[pd.Series, np.ndarray, float] | None = None

    for fold_number, (train_index, test_index) in enumerate(cv.split(X_train), start=1):
        started = time.perf_counter()
        X_train_fold = X_train.iloc[train_index]
        X_val_fold = X_train.iloc[test_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[test_index]

        print(f"{name}: fitting fold {fold_number}/30", file=sys.stderr, flush=True)
        pipeline.fit(X_train_fold, y_train_fold)
        y_score = positive_class_scores(pipeline, X_val_fold)
        auc = roc_auc_score(y_val_fold, y_score)
        aucs.append(auc)
        elapsed = time.perf_counter() - started
        print(
            f"{name}: fold {fold_number}/30 AUC={auc:.4f} ({elapsed:.1f}s)",
            file=sys.stderr,
            flush=True,
        )

        if fold_number == 1:
            representative_curve = (y_val_fold.copy(), y_score.copy(), auc)

    if representative_curve is None:
        raise RuntimeError(f"No ROC data collected for {name}")
    return aucs, representative_curve


def save_plot(
    representative_curves: dict[str, tuple[pd.Series, np.ndarray, float]]
) -> None:
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_score, auc) in representative_curves.items():
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            name=f"{name} (AUC={auc:.3f})",
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1)
    ax.set_title("ROC Curves on Joint Corpus")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    X, y = load_combined_data()
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: dict[str, list[float]] = {}
    representative_curves: dict[str, tuple[pd.Series, np.ndarray, float]] = {}

    for name, classifier in classifiers():
        aucs, representative_curve = evaluate_classifier(name, classifier, X_train, y_train)
        results[name] = aucs
        representative_curves[name] = representative_curve

    save_plot(representative_curves)

    print(
        "ROC-AUC on joint corpus (30-run average, tuned hyperparameters, "
        "full feature set):"
    )
    print()
    for name in ["MNB", "LR", "RF", "SVC"]:
        aucs = np.array(results[name], dtype=float)
        print(f"  {name}:  AUC = {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
