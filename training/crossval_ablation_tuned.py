"""
Progressive feature ablation under the tuned classical ML hyperparameters.

This script follows the existing crossval.py/crossval_combined_ds.py protocol:
an 80/20 stratified split is made first, then six repetitions of five-fold
cross-validation are run on the 80% training partition. The feature order and
extractors match the existing progressive ablation logic.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from root.scripts.BPE import BPETokenizer
from root.scripts.FILTRANS import (
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
BASE_RANDOM_STATE = 42
REPETITIONS = 6
N_FOLDS = 5

CLASSIFIERS = [
    ("MNB", MultinomialNB(alpha=0.1)),
    ("LR", LogisticRegression(C=1.0, max_iter=2000, n_jobs=1)),
    (
        "RF",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            n_jobs=1,
            random_state=BASE_RANDOM_STATE,
        ),
    ),
    ("SVC", SVC(C=0.1, kernel="linear")),
]

FEATURE_SET_LABELS = [
    "Vectorizers (TF-IDF + BOW)",
    "+ Readability (READ)",
    "+ Out-of-vocabulary (OOV)",
    "+ Stop words (SW)",
    "+ Traditional features (TRAD)",
    "+ Syllabic features (SYLL)",
    "+ Lexical features (LEX)",
    "+ Morphological features (MORPH) [full set]",
]

DEFAULT_JOINT_ABLATION = {
    "Vectorizers (TF-IDF + BOW)": {
        "MNB": 0.6297,
        "LR": 0.9475,
        "RF": 0.9358,
        "SVC": 0.9212,
    },
    "+ Readability (READ)": {
        "MNB": 0.6079,
        "LR": 0.9469,
        "RF": 0.9355,
        "SVC": 0.9113,
    },
    "+ Out-of-vocabulary (OOV)": {
        "MNB": 0.5872,
        "LR": 0.9495,
        "RF": 0.9346,
        "SVC": 0.9059,
    },
    "+ Stop words (SW)": {
        "MNB": 0.5714,
        "LR": 0.9495,
        "RF": 0.9335,
        "SVC": 0.9012,
    },
    "+ Traditional features (TRAD)": {
        "MNB": 0.5920,
        "LR": 0.9510,
        "RF": 0.9314,
        "SVC": 0.7825,
    },
    "+ Syllabic features (SYLL)": {
        "MNB": 0.6080,
        "LR": 0.9511,
        "RF": 0.9334,
        "SVC": 0.7825,
    },
    "+ Lexical features (LEX)": {
        "MNB": 0.6254,
        "LR": 0.9514,
        "RF": 0.9338,
        "SVC": 0.7824,
    },
    "+ Morphological features (MORPH) [full set]": {
        "MNB": 0.6251,
        "LR": 0.9513,
        "RF": 0.9308,
        "SVC": 0.7824,
    },
}


class TeeStream:
    """Write stdout/stderr to both the console and a persistent log file."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self.stream = stream
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tuned progressive feature ablation for all datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the markdown, CSV, and console log outputs.",
    )
    return parser.parse_args()


def load_dataset_frame(dataset_key: str) -> pd.DataFrame:
    if dataset_key == "Cruz":
        return load_single_dataset("Cruz", "FakeNewsFilipino_Cruz2020.csv")
    if dataset_key == "Lupac":
        return load_single_dataset("Lupac", "FakeNewsPhilippines2024_Lupac.csv")
    if dataset_key == "Joint":
        cruz = load_single_dataset("Cruz", "FakeNewsFilipino_Cruz2020.csv")
        lupac = load_single_dataset("Lupac", "FakeNewsPhilippines2024_Lupac.csv")
        return pd.concat([cruz, lupac], ignore_index=True)
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def load_single_dataset(dataset_dir: str, dataset_filename: str) -> pd.DataFrame:
    base_path = BASE_DIR / "root" / "datasets" / dataset_dir
    data = pd.read_csv(base_path / dataset_filename)
    feature_frames = [
        pd.read_csv(base_path / "TradFeatures.csv"),
        pd.read_csv(base_path / "SyllFeatures.csv"),
        pd.read_csv(base_path / "OovFeatures.csv"),
        pd.read_csv(base_path / "SwFeatures.csv"),
        pd.read_csv(base_path / "ReadFeatures.csv"),
        pd.read_csv(base_path / "LexFeatures.csv"),
        pd.read_csv(base_path / "MorphFeatures.csv"),
    ]
    frames = [data.reset_index(drop=True)]
    frames.extend(frame.reset_index(drop=True) for frame in feature_frames)
    return pd.concat(frames, axis=1)


def refresh_stopword_feature_csvs() -> None:
    targets = [
        ("Cruz", "FakeNewsFilipino_Cruz2020.csv"),
        ("Lupac", "FakeNewsPhilippines2024_Lupac.csv"),
    ]
    extractor = StopWordsExtractor(from_csv=False)

    print("")
    print("Refreshing cached stop-word feature CSVs with StopWordsExtractor")
    for dataset_dir, dataset_filename in targets:
        base_path = BASE_DIR / "root" / "datasets" / dataset_dir
        data = pd.read_csv(base_path / dataset_filename)
        output_path = base_path / "SwFeatures.csv"
        print(
            f"Stop-word refresh started | dataset={dataset_dir} | "
            f"rows={len(data)} | output={output_path}"
        )
        values = extractor.transform(data["article"])
        sw_features = pd.DataFrame(values, columns=["count_stopwords"])
        if len(sw_features) != len(data):
            raise ValueError(
                f"Stop-word feature row mismatch for {dataset_dir}: "
                f"{len(sw_features)} features for {len(data)} articles"
            )
        sw_features.to_csv(output_path, index=False)
        print(
            f"Stop-word refresh completed | dataset={dataset_dir} | "
            f"cached_rows={len(sw_features)}"
        )


def all_feature_steps():
    tokenizer = BPETokenizer()
    return [
        (
            "vectorizers",
            ColumnTransformer(
                transformers=[
                    ("bow", CountVectorizer(), "article"),
                    (
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 3), tokenizer=tokenizer.tokenize
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


def append_raw_rows(raw_csv_path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(raw_csv_path, mode="a", header=not raw_csv_path.exists(), index=False)


def run_condition(
    dataset_key: str,
    dataset_name: str,
    feature_label: str,
    feature_steps: list,
    classifier_id: str,
    classifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[dict]:
    cv = RepeatedKFold(
        n_splits=N_FOLDS, n_repeats=REPETITIONS, random_state=BASE_RANDOM_STATE
    )
    print("")
    print(
        "Condition | "
        f"dataset={dataset_name} | feature_set={feature_label} | "
        f"classifier={classifier_id}"
    )

    splits = list(enumerate(cv.split(X_train), start=1))

    def fit_fold(run_index, train_index, test_index) -> dict:
        repeat = ((run_index - 1) // N_FOLDS) + 1
        fold = ((run_index - 1) % N_FOLDS) + 1
        pipeline = Pipeline(
            steps=[
                ("features", FeatureUnion(feature_steps)),
                ("classifier", clone(classifier)),
            ]
        )
        X_train_fold = X_train.iloc[train_index]
        X_val_fold = X_train.iloc[test_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[test_index]

        pipeline.fit(X_train_fold, y_train_fold)
        y_val_pred = pipeline.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_val_pred)

        message = (
            "Fold metric | "
            f"dataset={dataset_name} | feature_set={feature_label} | "
            f"classifier={classifier_id} | repeat={repeat} | fold={fold} | "
            f"accuracy={accuracy:.9f}"
        )

        return {
            "dataset_key": dataset_key,
            "dataset": dataset_name,
            "feature_set": feature_label,
            "features": ",".join(step_name for step_name, _ in feature_steps),
            "classifier": classifier_id,
            "repeat": repeat,
            "fold": fold,
            "accuracy": accuracy,
            "log": message,
        }

    rows = Parallel(n_jobs=4, pre_dispatch="2*n_jobs")(
        delayed(fit_fold)(run_index, train_index, test_index)
        for run_index, (train_index, test_index) in splits
    )
    for row in rows:
        print(row.pop("log"))

    accuracies = [row["accuracy"] for row in rows]
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f"Accuracies: {accuracies}")
    print(
        "Mean accuracy for "
        f"{REPETITIONS} repetitions of {N_FOLDS}-fold cross-validation "
        f"with {classifier_id}: {mean_accuracy:.9f}"
    )
    return rows


def run_feature_set_condition(
    dataset_key: str,
    dataset_name: str,
    feature_label: str,
    feature_steps: list,
    classifiers: list[tuple[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[dict]:
    cv = RepeatedKFold(
        n_splits=N_FOLDS, n_repeats=REPETITIONS, random_state=BASE_RANDOM_STATE
    )

    print("")
    print(
        "Feature-set condition | "
        f"dataset={dataset_name} | feature_set={feature_label} | "
        f"classifiers={','.join(classifier_id for classifier_id, _ in classifiers)}"
    )

    splits = list(enumerate(cv.split(X_train), start=1))
    feature_names = ",".join(step_name for step_name, _ in feature_steps)

    def fit_fold(run_index, train_index, test_index) -> list[dict]:
        repeat = ((run_index - 1) // N_FOLDS) + 1
        fold = ((run_index - 1) % N_FOLDS) + 1
        feature_union = FeatureUnion(feature_steps)
        X_train_fold = X_train.iloc[train_index]
        X_val_fold = X_train.iloc[test_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[test_index]

        X_train_features = feature_union.fit_transform(X_train_fold, y_train_fold)
        X_val_features = feature_union.transform(X_val_fold)

        fold_rows = []
        for classifier_id, classifier in classifiers:
            fitted_classifier = clone(classifier)
            fitted_classifier.fit(X_train_features, y_train_fold)
            y_val_pred = fitted_classifier.predict(X_val_features)
            accuracy = accuracy_score(y_val_fold, y_val_pred)

            message = (
                "Fold metric | "
                f"dataset={dataset_name} | feature_set={feature_label} | "
                f"classifier={classifier_id} | repeat={repeat} | fold={fold} | "
                f"accuracy={accuracy:.9f}"
            )
            fold_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": dataset_name,
                    "feature_set": feature_label,
                    "features": feature_names,
                    "classifier": classifier_id,
                    "repeat": repeat,
                    "fold": fold,
                    "accuracy": accuracy,
                    "log": message,
                }
            )
        return fold_rows

    nested_rows = Parallel(n_jobs=4, pre_dispatch="2*n_jobs")(
        delayed(fit_fold)(run_index, train_index, test_index)
        for run_index, (train_index, test_index) in splits
    )
    rows = [row for fold_rows in nested_rows for row in fold_rows]
    for row in rows:
        print(row.pop("log"))

    for classifier_id, _ in classifiers:
        accuracies = [
            row["accuracy"] for row in rows if row["classifier"] == classifier_id
        ]
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(
            "Mean accuracy for "
            f"{REPETITIONS} repetitions of {N_FOLDS}-fold cross-validation "
            f"with {classifier_id}: {mean_accuracy:.9f}"
        )
    return rows


def markdown_table(summary: pd.DataFrame, dataset_name: str) -> str:
    lines = [
        f"### {dataset_name}",
        "",
        "| Feature Set | MNB | LR | RF | SVC |",
        "|---|---:|---:|---:|---:|",
    ]
    dataset_summary = summary[summary["dataset"] == dataset_name]
    for feature_label in FEATURE_SET_LABELS:
        row = dataset_summary[dataset_summary["feature_set"] == feature_label]
        values = {}
        for classifier_id in ["MNB", "LR", "RF", "SVC"]:
            cell = row[row["classifier"] == classifier_id]["accuracy"]
            values[classifier_id] = f"{float(cell.iloc[0]):.4f}" if not cell.empty else ""
        lines.append(
            f"| {feature_label} | {values['MNB']} | {values['LR']} | "
            f"{values['RF']} | {values['SVC']} |"
        )
    return "\n".join(lines)


def comparison_text(summary: pd.DataFrame) -> str:
    joint = summary[summary["dataset_key"] == "Joint"]

    def tuned(feature_label: str, classifier_id: str) -> float:
        value = joint[
            (joint["feature_set"] == feature_label)
            & (joint["classifier"] == classifier_id)
        ]["accuracy"]
        return float(value.iloc[0])

    baseline = "Vectorizers (TF-IDF + BOW)"
    sw = "+ Stop words (SW)"
    trad = "+ Traditional features (TRAD)"
    full = "+ Morphological features (MORPH) [full set]"

    default_baseline_to_trad = (
        DEFAULT_JOINT_ABLATION[trad]["SVC"] - DEFAULT_JOINT_ABLATION[baseline]["SVC"]
    )
    default_sw_to_trad = DEFAULT_JOINT_ABLATION[trad]["SVC"] - DEFAULT_JOINT_ABLATION[
        sw
    ]["SVC"]
    tuned_baseline_to_trad = tuned(trad, "SVC") - tuned(baseline, "SVC")
    tuned_sw_to_trad = tuned(trad, "SVC") - tuned(sw, "SVC")

    if abs(tuned_baseline_to_trad) < abs(default_baseline_to_trad):
        svc_interpretation = (
            "The tuned linear SVC reduces the default SVC drop when TRAD enters "
            "the feature set."
        )
    else:
        svc_interpretation = (
            "The tuned linear SVC does not reduce the default SVC drop when TRAD "
            "enters the feature set."
        )

    full_set_lines = []
    for classifier_id in ["MNB", "LR", "RF", "SVC"]:
        delta = tuned(full, classifier_id) - DEFAULT_JOINT_ABLATION[full][classifier_id]
        full_set_lines.append(f"{classifier_id}: {delta:+.4f}")

    return "\n".join(
        [
            "## Comparison With Default-Parameter Joint Ablation",
            "",
            "Default-reference values are from Manuscript.docx Table 7, which "
            "reports the joint-corpus progressive ablation under default "
            "classifier parameters.",
            "",
            f"- Default SVC changed from {DEFAULT_JOINT_ABLATION[baseline]['SVC']:.4f} "
            f"at vectorizers only to {DEFAULT_JOINT_ABLATION[trad]['SVC']:.4f} "
            f"when TRAD entered ({default_baseline_to_trad:+.4f}); from +SW "
            f"to +TRAD, it changed {default_sw_to_trad:+.4f}.",
            f"- Tuned SVC changed from {tuned(baseline, 'SVC'):.4f} at "
            f"vectorizers only to {tuned(trad, 'SVC'):.4f} when TRAD entered "
            f"({tuned_baseline_to_trad:+.4f}); from +SW to +TRAD, it changed "
            f"{tuned_sw_to_trad:+.4f}. {svc_interpretation}",
            "- Full-set tuned minus default joint-corpus deltas: "
            + "; ".join(full_set_lines)
            + ".",
        ]
    )


def write_summary_markdown(
    summary_path: Path,
    raw_csv_path: Path,
    log_path: Path,
    rows: list[dict],
    started_at: str,
    finished_at: str,
) -> None:
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["dataset_key", "dataset", "feature_set", "classifier"], as_index=False)[
            "accuracy"
        ]
        .mean()
        .sort_values(["dataset_key", "feature_set", "classifier"])
    )

    dataset_names = [
        "Fake News Filipino 2020",
        "Fake News Filipino 2024",
        "Joint corpus",
    ]

    lines = [
        "# Tuned Hyperparameter Progressive Feature Ablation",
        "",
        f"Started: {started_at}",
        f"Finished: {finished_at}",
        "",
        "Protocol: stratified 80% training partition, followed by "
        f"{REPETITIONS} repetitions of {N_FOLDS}-fold cross-validation on the "
        f"training partition, base random_state={BASE_RANDOM_STATE}.",
        "",
        "Tuned hyperparameters: MNB alpha=0.1; LR C=1.0, max_iter=2000; "
        "RF n_estimators=100, max_depth=20, min_samples_split=2; "
        "SVC C=0.1, kernel=linear.",
        "",
        f"Raw fold-level CSV: `{raw_csv_path.as_posix()}`",
        f"Console log: `{log_path.as_posix()}`",
        "",
        "## Mean Accuracy Tables",
        "",
    ]

    for dataset_name in dataset_names:
        lines.append(markdown_table(summary, dataset_name))
        lines.append("")

    lines.append(comparison_text(summary))
    lines.append("")
    lines.append("## Raw Console Output")
    lines.append("")
    lines.append("```text")
    lines.append(log_path.read_text(encoding="utf-8", errors="replace").rstrip())
    lines.append("```")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path, log_path: Path) -> None:
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    raw_csv_path = output_dir / "tuned_feature_ablation_raw.csv"
    summary_path = output_dir / "tuned_feature_ablation_summary.md"

    if summary_path.exists():
        summary_path.unlink()
    if raw_csv_path.exists():
        existing_frame = pd.read_csv(raw_csv_path)
    else:
        existing_frame = pd.DataFrame()

    print(f"Tuned Feature Ablation Results ({started_at})")
    print(f"Output directory: {output_dir}")
    print(f"Raw CSV: {raw_csv_path}")
    print(f"Console log: {log_path}")
    print(
        "Protocol: train_test_split(test_size=0.2, stratify=y, random_state=42), "
        "then RepeatedKFold(n_splits=5, n_repeats=6, random_state=42)."
    )
    print(
        "Tuned classifiers: MNB(alpha=0.1); LR(C=1.0, max_iter=2000); "
        "RF(n_estimators=100, max_depth=20, min_samples_split=2); "
        "SVC(C=0.1, kernel=linear)."
    )
    print("Using existing Step 0 stop-word caches; no feature CSVs are regenerated here.")

    if existing_frame.empty:
        print("No existing raw CSV found; starting from the first condition.")
    else:
        print(
            "Resuming from existing raw CSV | "
            f"rows={len(existing_frame)} | path={raw_csv_path}"
        )

    all_rows: list[dict] = existing_frame.to_dict(orient="records")
    expected_condition_rows = REPETITIONS * N_FOLDS * len(CLASSIFIERS)

    def condition_complete(dataset_key: str, feature_label: str) -> bool:
        if existing_frame.empty:
            return False
        subset = existing_frame[
            (existing_frame["dataset_key"] == dataset_key)
            & (existing_frame["feature_set"] == feature_label)
        ]
        return (
            len(subset) == expected_condition_rows
            and set(subset["classifier"]) == {key for key, _ in CLASSIFIERS}
        )

    datasets = [
        ("Cruz", "Fake News Filipino 2020"),
        ("Lupac", "Fake News Filipino 2024"),
        ("Joint", "Joint corpus"),
    ]

    for dataset_key, dataset_name in datasets:
        data = load_dataset_frame(dataset_key)
        y = data["label"]
        X = data.drop("label", axis=1)
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=BASE_RANDOM_STATE,
            stratify=y,
        )
        print("")
        print(
            f"Dataset loaded | key={dataset_key} | name={dataset_name} | "
            f"rows={len(data)} | training_rows={len(X_train)}"
        )

        feature_steps = []
        for feature_label, feature_step in zip(FEATURE_SET_LABELS, all_feature_steps()):
            feature_steps.append(feature_step)
            active_steps = list(feature_steps)
            if condition_complete(dataset_key, feature_label):
                print(
                    "Skipping completed condition | "
                    f"dataset={dataset_name} | feature_set={feature_label} | "
                    f"rows={expected_condition_rows}"
                )
                continue
            rows = run_feature_set_condition(
                dataset_key=dataset_key,
                dataset_name=dataset_name,
                feature_label=feature_label,
                feature_steps=active_steps,
                classifiers=CLASSIFIERS,
                X_train=X_train,
                y_train=y_train,
            )
            all_rows.extend(rows)
            append_raw_rows(raw_csv_path, rows)

    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("")
    print(f"Completed tuned feature ablation at {finished_at}")
    print(f"Total fold metrics: {len(all_rows)}")

    sys.stdout.flush()
    sys.stderr.flush()
    write_summary_markdown(
        summary_path=summary_path,
        raw_csv_path=raw_csv_path,
        log_path=log_path,
        rows=all_rows,
        started_at=started_at,
        finished_at=finished_at,
    )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or (
        BASE_DIR / "results" / "tuned_feature_ablation" / timestamp
    )
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tuned_feature_ablation_console.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(sys.__stdout__, log_file)
        tee_stderr = TeeStream(sys.__stderr__, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            run(output_dir=output_dir, log_path=log_path)


if __name__ == "__main__":
    main()
