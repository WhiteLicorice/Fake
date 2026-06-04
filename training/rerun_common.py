"""Shared utilities for the stop-word-fix rerun scripts."""

from __future__ import annotations

import json
import os
import pickle
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "stopwords_fix_rerun"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_REPEATS = 6
LOAD_FROM_CSV = True

os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

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
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    article_file: str


DATASETS = {
    "Cruz": DatasetSpec(
        key="Cruz",
        display_name="Fake News Filipino 2020",
        article_file="FakeNewsFilipino_Cruz2020.csv",
    ),
    "Lupac": DatasetSpec(
        key="Lupac",
        display_name="Fake News Filipino 2024",
        article_file="FakeNewsPhilippines2024_Lupac.csv",
    ),
}

FEATURE_FILES = [
    "TradFeatures.csv",
    "SyllFeatures.csv",
    "OovFeatures.csv",
    "SwFeatures.csv",
    "ReadFeatures.csv",
    "LexFeatures.csv",
    "MorphFeatures.csv",
]

FULL_FEATURE_NAMES = [
    "vectorizers",
    "read",
    "oov",
    "sw",
    "trad",
    "syll",
    "lex",
    "morph",
]

DEPLOYMENT_FEATURE_NAMES = [
    "vectorizers",
    "read",
    "oov",
    "sw",
    "trad",
    "syll",
]

FEATURE_SET_LABELS = [
    "Vectorizers only (TF-IDF + BOW)",
    "+ READ",
    "+ OOV",
    "+ SW",
    "+ TRAD",
    "+ SYLL",
    "+ LEX",
    "+ MORPH",
]

CLASSIFIER_ORDER = ["MNB", "LR", "RF", "SVC"]
CLASSIFIER_DISPLAY = {
    "MNB": "Multinomial Naive Bayes",
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "SVC": "Support Vector Classifier",
}

DEFAULT_REFERENCE = {
    "Table 6": {
        ("LR", "Fake News Filipino 2020"): 0.951,
        ("LR", "Fake News Filipino 2024"): 0.947,
        ("LR", "Joint corpus"): 0.924,
        ("MNB", "Fake News Filipino 2020"): 0.923,
        ("MNB", "Fake News Filipino 2024"): 0.885,
        ("MNB", "Joint corpus"): 0.860,
        ("RF", "Fake News Filipino 2020"): 0.919,
        ("RF", "Fake News Filipino 2024"): 0.926,
        ("RF", "Joint corpus"): 0.888,
        ("SVC", "Fake News Filipino 2020"): 0.951,
        ("SVC", "Fake News Filipino 2024"): 0.947,
        ("SVC", "Joint corpus"): 0.922,
    },
    "Descriptives": {
        ("Fake News Filipino 2024", "mean_oov"): 22.2,
        ("Fake News Filipino 2024", "mean_readability"): 19.7,
        ("Fake News Filipino 2024", "mean_stopwords"): 74.5,
        ("Fake News Filipino 2020", "mean_oov"): 17.6,
        ("Fake News Filipino 2020", "mean_readability"): 20.5,
        ("Fake News Filipino 2020", "mean_stopwords"): 77.8,
    },
    "Mann-Whitney": {
        ("OOV count", "U"): 5814710.50,
        ("OOV count", "r"): 0.13,
        ("Readability index", "U"): 4745072.00,
        ("Readability index", "r"): -0.08,
        ("Stop word count", "U"): 4759872.00,
        ("Stop word count", "r"): -0.07,
    },
    "Table 9 count-stopwords coefficient": 0.025210022216103887,
}

TEST_ARTICLES = [
    {
        "id": "false_positive_article",
        "previous_outcome": "FP",
        "gold_label": 0,
        "text": (
            "Mahaharap sa kasong administratibo ang isang opisyal ng pulisya "
            "matapos magwala sa mismong himpilan, pinasok sa opisina ang kanyang "
            "hepe at pinagsasalitaan umano ng masama, Lunes ng gabi, sa Bacoor "
            "City, Cavite. Kasong grave misconduct ang kakaharapin ni Chief Insp. "
            "Virgilio Rubio, deputy chief ng Bacoor City Police, batay sa reklamo "
            "ni Supt. Rommel Estolano. Sa ulat sa tanggapan ni Cavite Police "
            "Provincial Office director Senior Supt. Joselito Esquivel, bandang "
            "7:30 ng gabi nang magtungo sa istasyon ng pulisya si Rubio na lasing "
            "na lasing, biglang pinaghahagis ang mga upuan at iba pang gamit sa "
            "opisina hanggang pumasok sa tanggapan ni Rubio habang may hawak na "
            "baril at nagbitiw ng kung anu-anong masasamang salita. Gayunman, "
            "naawat ni Senior Insp. Chey Chey Saulog si Rubio at pinalabas sa "
            "himpilan ng pulisya."
        ),
    },
    {
        "id": "true_positive_article",
        "previous_outcome": "TP",
        "gold_label": 0,
        "text": (
            "SAN CARLOS CITY, Pangasinan - Dalawang hinihinalang carnapper na "
            "nagpapanggap na miyembro ng Criminal Investigation and Detection "
            "Group (CIDG) ang naaresto sa San Carlos City, Pangasinan. Sa kanyang "
            "report kay Pangasinan Police Provincial Office director Senior Supt. "
            "Reynaldo Biay, kinilala ni San Carlos City Police chief Supt. Charlie "
            "Umayam ang mga nadakip na sina Michael Edades, 34, may asawa, "
            "negosyante, at residente ng Barangay Mangin, Dagupan City; at Daniel "
            "Salopagio Jr., 29, binata, bus driver, ng Bgy. Nalsian Norte, "
            "Bayambang. Nabawi mula sa dalawa ang isang motorized tricycle, isang "
            "Sony Ericsson cell phone ng kanilang nabiktima at iba't ibang ID ng "
            "CIDG. Ini-report sa pulisya ang pagtangay ng mga suspek sa isang "
            "Honda TMX 155 motorized tricycle (8150-ZA) na minamaneho ni Edwin "
            "Arzadon y Balat, 36, biyudo, ng lungsod, madaling araw noong Linggo "
            "sa Sta. Isabel Subdivision."
        ),
    },
    {
        "id": "false_negative_article",
        "previous_outcome": "FN",
        "gold_label": 1,
        "text": (
            "Binigyan ng tatlong taong extension para sa kanyang panunungkulan "
            "bilang Commissioner ng Philippine Basketball Association si Willie "
            "Marcial. Sa kanilang annual planning session sa Star Hotels sa "
            "bansang Italya, gaya ng naging pagkakatalaga sa kanya bilang "
            "Commissioner ng liga, naging unanimous ang pagbibigay ng board of "
            "governors ng extension sa termino ni Marcial kahapon (Huwebes). May "
            "nalalabi pang isang taon sa naunang tatlong taong kontrata na "
            "nilagdaan ni Marcial noong 2018 pero binigyan sya ng PBA board ng "
            "bagong vote of confidence. Ito'y bunga na rin ng magandang "
            "performance nito na nagustuhan ng board. We're open and very "
            "transparent about his performance, wika ni PBA Chairman Ricky Vargas "
            "tungkol kay Marcial."
        ),
    },
    {
        "id": "true_negative_article",
        "previous_outcome": "TN",
        "gold_label": 1,
        "text": (
            "Patay ang isang 5-anyos na lalaki sa Lapu-Lapu City, Cebu matapos "
            "umano siyang pukpukin sa ulo at ihagis sa dagat ng kaniyang 14-anyos "
            "na kapatid. Sa ulat ng ABS-CBN News, nangyari ang insidente sa Sitio "
            "Lawis, Barangay Suba-Basbas nitong Biyernes ng umaga, Mayo 10. "
            "Naglalaro lamang daw ang dalawa nang bigla umanong kumuha ng bato "
            "ang 14-anyos at ipinukpok ito sa 5-anyos niyang kapatid. Nang "
            "mawalan ng malay ang biktima ay doon na siya tinulak sa dagat ng "
            "hanggang sa malunod, ayon kay Police Lt. Col. Christian Torres. "
            "Dagdag pa ng ulat, napag-alaman umano sa imbestigasyon ng pulisya na "
            "magkaiba ang ama ng magkapatid at galit umano ang 14-anyos sa tatay "
            "ng 5-anyos na biktima. Dahil menor de edad ang suspek, isinailalim "
            "umano siya sa isang home care facility. Patuloy pa rin namang "
            "nagsasagawa ang mga awtoridad ng imbestigasyon hinggil sa naturang "
            "insidente."
        ),
    },
]

BENCHMARK_ARTICLES = [
    (
        "Short article (~50 words)",
        TEST_ARTICLES[3]["text"].split(".")[0] + ". " + TEST_ARTICLES[3]["text"].split(".")[1] + ".",
    ),
    ("Medium article (~100 words)", TEST_ARTICLES[2]["text"]),
    ("Long article (~200 words)", TEST_ARTICLES[0]["text"]),
]


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_single_dataset(key: str, include_lex_morph: bool = True) -> pd.DataFrame:
    spec = DATASETS[key]
    dataset_dir = BASE_DIR / "root" / "datasets" / key
    data = pd.read_csv(dataset_dir / spec.article_file)
    feature_files = FEATURE_FILES if include_lex_morph else FEATURE_FILES[:5]
    frames = [data.reset_index(drop=True)]
    for feature_file in feature_files:
        feature_frame = pd.read_csv(dataset_dir / feature_file)
        if len(feature_frame) != len(data):
            raise ValueError(
                f"Row mismatch for {key}/{feature_file}: "
                f"{len(feature_frame)} features for {len(data)} articles"
            )
        frames.append(feature_frame.reset_index(drop=True))
    return pd.concat(frames, axis=1)


def load_dataset(dataset_key: str, include_lex_morph: bool = True) -> tuple[pd.DataFrame, pd.Series, str]:
    if dataset_key == "Joint":
        cruz = load_single_dataset("Cruz", include_lex_morph=include_lex_morph)
        lupac = load_single_dataset("Lupac", include_lex_morph=include_lex_morph)
        data = pd.concat([cruz, lupac], ignore_index=True)
        display_name = "Joint corpus"
    else:
        data = load_single_dataset(dataset_key, include_lex_morph=include_lex_morph)
        display_name = DATASETS[dataset_key].display_name

    y = data["label"]
    X = data.drop("label", axis=1)
    return X, y, display_name


def train_partition(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def all_feature_steps(
    include_lex_morph: bool = True,
    from_csv: bool = LOAD_FROM_CSV,
    raw_text: bool = False,
) -> list[tuple[str, object]]:
    tokenizer = BPETokenizer()
    if raw_text:
        steps: list[tuple[str, object]] = [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    tokenizer=tokenizer.tokenize,
                ),
            ),
            ("bow", CountVectorizer()),
            ("read", READExtractor(from_csv=False)),
            ("oov", OOVExtractor(from_csv=False)),
            ("sw", StopWordsExtractor(from_csv=False)),
            ("trad", TRADExtractor(from_csv=False)),
            ("syll", SYLLExtractor(from_csv=False)),
        ]
    else:
        steps = [
            (
                "vectorizers",
                ColumnTransformer(
                    transformers=[
                        ("bow", CountVectorizer(), "article"),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 3),
                                tokenizer=tokenizer.tokenize,
                            ),
                            "article",
                        ),
                    ]
                ),
            ),
            ("read", READExtractor(from_csv=from_csv)),
            ("oov", OOVExtractor(from_csv=from_csv)),
            ("sw", StopWordsExtractor(from_csv=from_csv)),
            ("trad", TRADExtractor(from_csv=from_csv)),
            ("syll", SYLLExtractor(from_csv=from_csv)),
        ]
    if include_lex_morph:
        extra_from_csv = False if raw_text else from_csv
        steps.extend(
            [
                ("lex", LEXExtractor(from_csv=extra_from_csv)),
                ("morph", MORPHExtractor(from_csv=extra_from_csv)),
            ]
        )
    return steps


def make_pipeline(
    classifier,
    include_lex_morph: bool = True,
    from_csv: bool = LOAD_FROM_CSV,
    raw_text: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    all_feature_steps(
                        include_lex_morph=include_lex_morph,
                        from_csv=from_csv,
                        raw_text=raw_text,
                    )
                ),
            ),
            ("classifier", classifier),
        ]
    )


def default_classifiers() -> dict[str, object]:
    return {
        "MNB": MultinomialNB(),
        "LR": LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        "SVC": SVC(),
    }


def tuned_classifiers(best_params: dict | None = None, probability_for_svc: bool = False) -> dict[str, object]:
    params = best_params or {}
    rf_params = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
    }
    rf_params.update(params.get("RF", {}))
    return {
        "MNB": MultinomialNB(alpha=params.get("MNB", {}).get("alpha", 0.1)),
        "LR": LogisticRegression(
            C=params.get("LR", {}).get("C", 1.0),
            max_iter=2000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "RF": RandomForestClassifier(
            n_jobs=1,
            random_state=RANDOM_STATE,
            **rf_params,
        ),
        "SVC": SVC(
            C=params.get("SVC", {}).get("C", 0.1),
            kernel=params.get("SVC", {}).get("kernel", "linear"),
            probability=probability_for_svc,
            random_state=RANDOM_STATE,
        ),
    }


def grid_search_specs() -> dict[str, tuple[object, dict]]:
    return {
        "MNB": (
            MultinomialNB(),
            {"classifier__alpha": [0.1, 1.0, 10.0]},
        ),
        "LR": (
            LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "RF": (
            RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [10, 20],
                "classifier__min_samples_split": [2, 5, 10],
            },
        ),
        "SVC": (
            SVC(random_state=RANDOM_STATE),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"],
            },
        ),
    }


def strip_classifier_prefix(best_params: dict[str, object]) -> dict[str, object]:
    return {
        key.replace("classifier__", ""): value
        for key, value in best_params.items()
    }


def evaluate_holdout(
    classifier_id: str,
    classifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    include_lex_morph: bool = True,
) -> dict:
    pipeline = make_pipeline(clone(classifier), include_lex_morph=include_lex_morph)
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    elapsed = time.perf_counter() - started
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTraining Model: {CLASSIFIER_DISPLAY[classifier_id]} ({classifier_id})")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Accuracy: {accuracy:.9f}")
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0,
        )
    )
    print("Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:")
    print(cm)
    return {
        "classifier": classifier_id,
        "classifier_name": CLASSIFIER_DISPLAY[classifier_id],
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "elapsed_seconds": elapsed,
    }


def run_grid_search(
    classifier_id: str,
    classifier,
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    include_lex_morph: bool = True,
) -> dict:
    pipeline = make_pipeline(classifier, include_lex_morph=include_lex_morph)
    grid = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=4,
        pre_dispatch="2*n_jobs",
        verbose=2,
    )
    started = time.perf_counter()
    print(f"\nGrid search: {CLASSIFIER_DISPLAY[classifier_id]} ({classifier_id})")
    print(f"Search space: {params}")
    grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    y_pred = grid.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_test, y_pred)
    best_params = strip_classifier_prefix(grid.best_params_)
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Best params: {best_params}")
    print(f"Best CV accuracy: {grid.best_score_:.9f}")
    print(f"Holdout accuracy: {accuracy:.9f}")
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0,
        )
    )
    print("Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:")
    print(cm)
    return {
        "classifier": classifier_id,
        "classifier_name": CLASSIFIER_DISPLAY[classifier_id],
        "best_params": best_params,
        "best_cv_accuracy": float(grid.best_score_),
        "holdout_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "elapsed_seconds": elapsed,
    }


def repeated_cv_scores(
    classifier_id: str,
    classifier,
    dataset_key: str,
    include_lex_morph: bool = True,
    feature_steps: list[tuple[str, object]] | None = None,
    feature_label: str = "full",
) -> list[dict]:
    X, y, dataset_name = load_dataset(dataset_key, include_lex_morph=include_lex_morph)
    X_train, _, y_train, _ = train_partition(X, y)
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    steps = feature_steps or all_feature_steps(include_lex_morph=include_lex_morph)
    rows = []
    print(
        f"\nCV condition: dataset={dataset_name}, classifier={classifier_id}, "
        f"feature_set={feature_label}"
    )

    splits = list(enumerate(cv.split(X_train), start=1))

    def fit_fold(run_index: int, train_index: np.ndarray, val_index: np.ndarray) -> dict:
        repeat = ((run_index - 1) // N_SPLITS) + 1
        fold = ((run_index - 1) % N_SPLITS) + 1
        pipeline = Pipeline(
            steps=[
                ("features", FeatureUnion(steps)),
                ("classifier", clone(classifier)),
            ]
        )
        X_fold_train = X_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]
        X_val = X_train.iloc[val_index]
        y_val = y_train.iloc[val_index]
        fold_started = time.perf_counter()
        pipeline.fit(X_fold_train, y_fold_train)
        pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, pred)
        elapsed = time.perf_counter() - fold_started
        message = (
            f"Fold metric | dataset={dataset_name} | classifier={classifier_id} | "
            f"feature_set={feature_label} | repeat={repeat} | fold={fold} | "
            f"accuracy={accuracy:.9f} | elapsed_seconds={elapsed:.2f}"
        )
        return {
            "dataset_key": dataset_key,
            "dataset": dataset_name,
            "classifier": classifier_id,
            "classifier_name": CLASSIFIER_DISPLAY[classifier_id],
            "feature_set": feature_label,
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


def summarize_scores(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return (
        df.groupby(["dataset", "classifier", "classifier_name", "feature_set"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", lambda x: x.std(ddof=1)))
        .sort_values(["dataset", "classifier"])
    )


def positive_scores(pipeline: Pipeline, X_val: pd.DataFrame) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    classes = classifier.classes_
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return classifier.predict_proba(pipeline.named_steps["features"].transform(X_val))[:, positive_index]


def repeated_cv_auc(classifier_id: str, classifier) -> list[dict]:
    X, y, dataset_name = load_dataset("Joint", include_lex_morph=True)
    X_train, _, y_train, _ = train_partition(X, y)
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    rows = []
    print(f"\nROC-AUC condition: dataset={dataset_name}, classifier={classifier_id}")

    splits = list(enumerate(cv.split(X_train), start=1))

    def fit_auc_fold(run_index: int, train_index: np.ndarray, val_index: np.ndarray) -> dict:
        repeat = ((run_index - 1) // N_SPLITS) + 1
        fold = ((run_index - 1) % N_SPLITS) + 1
        pipeline = make_pipeline(clone(classifier), include_lex_morph=True)
        X_fold_train = X_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]
        X_val = X_train.iloc[val_index]
        y_val = y_train.iloc[val_index]
        fold_started = time.perf_counter()
        pipeline.fit(X_fold_train, y_fold_train)
        y_score = positive_scores(pipeline, X_val)
        auc = roc_auc_score(y_val, y_score)
        elapsed = time.perf_counter() - fold_started
        message = (
            f"AUC metric | classifier={classifier_id} | repeat={repeat} | "
            f"fold={fold} | auc={auc:.9f} | elapsed_seconds={elapsed:.2f}"
        )
        return {
            "classifier": classifier_id,
            "classifier_name": CLASSIFIER_DISPLAY[classifier_id],
            "repeat": repeat,
            "fold": fold,
            "auc": float(auc),
            "elapsed_seconds": elapsed,
            "log": message,
        }

    rows = Parallel(n_jobs=4, pre_dispatch="2*n_jobs")(
        delayed(fit_auc_fold)(run_index, train_index, val_index)
        for run_index, (train_index, val_index) in splits
    )
    for row in rows:
        print(row.pop("log"))
    return rows


def is_vectorizer_feature(name: object) -> bool:
    value = str(name)
    return (
        value.startswith("vectorizers__")
        or value.startswith("tfidf__")
        or value.startswith("bow__")
    )


def feature_names_by_kind(pipeline: Pipeline) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = np.asarray(pipeline.named_steps["features"].get_feature_names_out())
    is_vectorizer = np.asarray([is_vectorizer_feature(name) for name in names])
    return names, names[is_vectorizer], names[~is_vectorizer]


def fit_lr_pipeline(
    include_lex_morph: bool = True,
    C: float = 1.0,
    raw_text: bool = False,
) -> tuple[Pipeline, pd.DataFrame | pd.Series, pd.Series]:
    if raw_text:
        cruz = pd.read_csv(BASE_DIR / "root" / "datasets" / "Cruz" / DATASETS["Cruz"].article_file)
        lupac = pd.read_csv(BASE_DIR / "root" / "datasets" / "Lupac" / DATASETS["Lupac"].article_file)
        data = pd.concat([cruz, lupac], ignore_index=True)
        X = data["article"]
        y = data["label"]
    else:
        X, y, _ = load_dataset("Joint", include_lex_morph=include_lex_morph)
    classifier = LogisticRegression(
        C=C,
        max_iter=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    pipeline = make_pipeline(
        classifier,
        include_lex_morph=include_lex_morph,
        from_csv=not raw_text,
        raw_text=raw_text,
    )
    pipeline.fit(X, y)
    return pipeline, X, y


def coefficients_for_pipeline(pipeline: Pipeline) -> pd.DataFrame:
    names, _, _ = feature_names_by_kind(pipeline)
    coef = pipeline.named_steps["classifier"].coef_[0]
    return pd.DataFrame({"feature": names, "coefficient": coef})


def parameter_count(pipeline: Pipeline) -> int:
    classifier = pipeline.named_steps["classifier"]
    return int(classifier.coef_.size + classifier.intercept_.size)


def local_inference_benchmark(pipeline: Pipeline) -> list[dict]:
    rows = []
    for article_length, text in BENCHMARK_ARTICLES:
        times = []
        for request_number in range(1, 11):
            started = time.perf_counter()
            prediction = int(pipeline.predict([text])[0])
            elapsed_ms = (time.perf_counter() - started) * 1000
            times.append(elapsed_ms)
            print(
                f"Benchmark request | length={article_length} | request={request_number} | "
                f"prediction={prediction} | elapsed_ms={elapsed_ms:.3f}"
            )
        rows.append(
            {
                "article_length": article_length,
                "times_ms": times,
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "sd_ms": statistics.stdev(times),
            }
        )
    all_times = [time_ms for row in rows for time_ms in row["times_ms"]]
    rows.append(
        {
            "article_length": "Overall",
            "times_ms": all_times,
            "mean_ms": statistics.mean(all_times),
            "median_ms": statistics.median(all_times),
            "sd_ms": statistics.stdev(all_times),
        }
    )
    return rows


def outcome_from_labels(gold_label: int, predicted_label: int) -> str:
    if gold_label == 0 and predicted_label == 0:
        return "TP"
    if gold_label == 0 and predicted_label == 1:
        return "FP"
    if gold_label == 1 and predicted_label == 0:
        return "FN"
    return "TN"


def extract_deployment_features_for_text(text: str) -> dict[str, float]:
    from root.scripts import OOV, READ, SW, SYLL, TRAD  # noqa: E402

    return {
        "word_count": TRAD.word_count_per_doc(text),
        "sentence_count": TRAD.sentence_count_per_doc(text),
        "polysyll_count": TRAD.polysyll_count_per_doc(text),
        "ave_word_length": TRAD.ave_word_length(text),
        "ave_phrase_count": TRAD.ave_phrase_count_per_doc(text),
        "ave_syllable_count_of_word": TRAD.ave_syllable_count_of_word(text),
        "word_count_per_sentence": TRAD.word_count_per_sentence(text),
        "consonant_cluster": SYLL.get_consonant_cluster(text),
        "v_density": SYLL.get_v(text),
        "cv_density": SYLL.get_cv(text),
        "vc_density": SYLL.get_vc(text),
        "cvc_density": SYLL.get_cvc(text),
        "vcc_density": SYLL.get_vcc(text),
        "cvcc_density": SYLL.get_cvcc(text),
        "ccvcc_density": SYLL.get_ccvcc(text),
        "ccvccc_density": SYLL.get_ccvccc(text),
        "count_oov_words": OOV.count_oov_words(text),
        "count_stopwords": SW.count_stopwords(text),
        "readability_score": READ.compute_readability_score(text),
    }


def nonzero_vectorizer_coefficients(pipeline: Pipeline, text: str, limit: int = 4) -> dict[str, list[dict]]:
    feature_union = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    names = np.asarray(feature_union.get_feature_names_out())
    coef = classifier.coef_[0]
    try:
        transformed = feature_union.transform([text])
    except Exception:
        transformed = feature_union.transform(pd.DataFrame({"article": [text]}))
    if hasattr(transformed, "tocoo"):
        row = transformed.tocoo()
        active = set(row.col.tolist())
    else:
        active = set(np.flatnonzero(np.asarray(transformed)[0]).tolist())
    vectorizer_indices = [
        idx
        for idx in active
        if is_vectorizer_feature(names[idx])
    ]
    rows = [
        {"feature": str(names[idx]), "coefficient": float(coef[idx])}
        for idx in vectorizer_indices
    ]
    rows = sorted(rows, key=lambda row: row["coefficient"])
    return {
        "lowest": rows[:limit],
        "highest": rows[-limit:],
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])
