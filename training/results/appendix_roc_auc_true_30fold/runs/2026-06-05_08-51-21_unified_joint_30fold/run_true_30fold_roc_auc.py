"""Run true 30-fold ROC-AUC evaluation for the joint FaKe corpus.

The pipeline uses one unified stratified 80/20 training partition and one
RepeatedStratifiedKFold split plan for all requested models. It saves per-fold
probabilities/logits, raw ROC arrays, fold metrics, summary tables, figures,
and a reproducibility manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


INVOCATION_CWD = Path.cwd().resolve()
SCRIPT_PATH = Path(__file__).resolve()


def find_training_dir() -> Path:
    for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
        if (candidate / "rerun_common.py").exists():
            return candidate
    raise RuntimeError("Could not locate training directory containing rerun_common.py")


TRAINING_DIR = find_training_dir()
REPO_DIR = TRAINING_DIR.parent
OUTPUT_ROOT = TRAINING_DIR / "results" / "appendix_roc_auc_true_30fold"
RUNS_DIR = OUTPUT_ROOT / "runs"

os.chdir(TRAINING_DIR)
sys.path.insert(0, str(TRAINING_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from rerun_common import (
    FEATURE_FILES,
    make_pipeline,
    positive_scores,
    tuned_classifiers,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_REPEATS = 6
MEAN_FPR = np.linspace(0.0, 1.0, 200)

MODEL_ORDER = ["MNB", "RF", "LR", "SVC", "DistilBERT", "RoBERTa"]
CLASSICAL_MODELS = ["MNB", "RF", "LR", "SVC"]
TRANSFORMER_MODELS = ["DistilBERT", "RoBERTa"]
TRANSFORMER_MODEL_NAMES = {
    "DistilBERT": "jcblaise/distilbert-tagalog-base-cased",
    "RoBERTa": "jcblaise/roberta-tagalog-base",
}
MODEL_COLORS = {
    "LR": "#0173B2",
    "MNB": "#DE8F05",
    "RF": "#029E73",
    "SVC": "#D55E00",
    "DistilBERT": "#CC78BC",
    "RoBERTa": "#CA9161",
}
PM = "+/-"
TITLE_DASH = "-"

ARTICLE_PATHS = {
    "Cruz": TRAINING_DIR / "root" / "datasets" / "Cruz" / "FakeNewsFilipino_Cruz2020.csv",
    "Lupac": TRAINING_DIR
    / "root"
    / "datasets"
    / "Lupac"
    / "FakeNewsPhilippines2024_Lupac.csv",
}

PREDICTION_COLUMNS = [
    "model",
    "repeat",
    "fold",
    "split_number",
    "seed",
    "combined_row_id",
    "y_true",
    "y_score",
    "y_pred",
    "logit_0",
    "logit_1",
    "elapsed_seconds",
    "batch_size",
    "eval_batch_size",
]

FOLD_METRIC_COLUMNS = [
    "model",
    "repeat",
    "fold",
    "split_number",
    "seed",
    "n_validation",
    "roc_auc",
    "accuracy",
    "f1_macro",
    "f1_fake",
    "f1_real",
    "elapsed_seconds",
    "batch_size",
    "eval_batch_size",
    "prediction_path",
    "roc_path",
]


class Tee:
    """Write console output to terminal and run.log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def __getattr__(self, name: str):
        return getattr(self.streams[0], name)


@dataclass(frozen=True)
class FoldSpec:
    split_number: int
    repeat: int
    fold: int
    train_index: np.ndarray
    val_index: np.ndarray

    @property
    def seed(self) -> int:
        return RANDOM_STATE + self.split_number - 1


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *MODEL_ORDER],
        default=["all"],
        help="Models to run. Default: all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible run and skip completed folds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute requested folds even when completed artifacts exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, splits, and output paths without training.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate summaries, plots, report, and manifest from saved predictions.",
    )
    parser.add_argument(
        "--fold-limit",
        type=positive_int,
        default=None,
        help="Optional smoke-test limit per model.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name under runs/.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory. Overrides --run-name.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--folds", type=positive_int, default=N_SPLITS)
    parser.add_argument("--repeats", type=positive_int, default=N_REPEATS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--eval-batch-size", type=positive_int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 mixed precision. Default uses fp16 when CUDA is available.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for transformer training.",
    )
    return parser.parse_args()


def selected_models(raw_models: Iterable[str]) -> list[str]:
    models = list(raw_models)
    if "all" in models:
        return MODEL_ORDER.copy()
    requested = set(models)
    return [model for model in MODEL_ORDER if model in requested]


def model_config_signature(args: argparse.Namespace, models: list[str]) -> dict[str, object]:
    return {
        "models": models,
        "seed": args.seed,
        "folds": args.folds,
        "repeats": args.repeats,
        "test_size": args.test_size,
        "fold_limit": args.fold_limit,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "no_fp16": args.no_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }


def config_matches(manifest: dict, signature: dict[str, object]) -> bool:
    saved = manifest.get("config_signature")
    if not isinstance(saved, dict):
        return False
    comparable_keys = ["seed", "folds", "repeats", "test_size", "fold_limit"]
    if any(saved.get(key) != signature.get(key) for key in comparable_keys):
        return False
    saved_models = saved.get("models")
    requested_models = signature.get("models")
    return saved_models == requested_models


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(args: argparse.Namespace, models: list[str]) -> Path:
    if args.run_dir is not None:
        if args.run_dir.is_absolute():
            return args.run_dir.resolve()
        return (INVOCATION_CWD / args.run_dir).resolve()
    if args.run_name:
        return (RUNS_DIR / args.run_name).resolve()

    signature = model_config_signature(args, models)
    if args.resume and RUNS_DIR.exists():
        candidates = sorted(
            [path for path in RUNS_DIR.glob("*_unified_joint_30fold") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            manifest_path = candidate / "run_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = read_json(manifest_path)
            except json.JSONDecodeError:
                continue
            if config_matches(manifest, signature):
                return candidate.resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (RUNS_DIR / f"{timestamp}_unified_joint_30fold").resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
        output = completed.stdout.strip()
        if completed.stderr.strip():
            output = (output + "\n" + completed.stderr.strip()).strip()
        return output
    except Exception as exc:
        return f"unavailable: {exc}"


def pip_freeze() -> list[str]:
    output = run_command([sys.executable, "-m", "pip", "freeze"], REPO_DIR)
    if output.startswith("unavailable:"):
        return [output]
    return [line for line in output.splitlines() if line.strip()]


def torch_environment() -> dict[str, object]:
    try:
        import torch

        payload: dict[str, object] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
        }
        if torch.cuda.is_available():
            payload["cuda_device"] = torch.cuda.get_device_name(0)
            payload["cuda_device_count"] = torch.cuda.device_count()
        return payload
    except Exception as exc:
        return {"torch_unavailable": str(exc), "cuda_available": False}


def data_source_paths() -> dict[str, object]:
    feature_paths = {}
    for dataset in ["Cruz", "Lupac"]:
        dataset_dir = TRAINING_DIR / "root" / "datasets" / dataset
        feature_paths[dataset] = [str(dataset_dir / filename) for filename in FEATURE_FILES]
    return {
        "article_paths": {key: str(path) for key, path in ARTICLE_PATHS.items()},
        "feature_paths": feature_paths,
        "tuned_best_params": str(
            TRAINING_DIR / "results" / "stopwords_fix_rerun" / "tuned_best_params.json"
        ),
    }


def list_output_files(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    return sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file()
    )


def write_manifest(run_dir: Path, manifest: dict) -> None:
    manifest["output_file_list"] = list_output_files(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


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


def configure_logging(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    warnings.simplefilter("default")
    return log_path


def initialize_manifest(
    args: argparse.Namespace,
    models: list[str],
    run_dir: Path,
    log_path: Path,
    start_time: datetime,
) -> dict:
    prior_manifest = None
    prior_manifest_path = run_dir / "run_manifest.json"
    if args.aggregate_only and prior_manifest_path.exists():
        try:
            prior_manifest = read_json(prior_manifest_path)
        except json.JSONDecodeError:
            prior_manifest = None

    copied_script = run_dir / SCRIPT_PATH.name
    shutil.copy2(SCRIPT_PATH, copied_script)
    manifest = {
        "status": "running",
        "command_used": " ".join([sys.executable, *sys.argv]),
        "script_path": str(SCRIPT_PATH),
        "copied_script_path": str(copied_script),
        "copied_script_sha256": sha256_file(copied_script),
        "git_commit_sha": run_command(["git", "rev-parse", "HEAD"], REPO_DIR),
        "git_status_summary": run_command(["git", "status", "--short"], REPO_DIR),
        "python_executable": sys.executable,
        "python_version": sys.version.replace(os.linesep, " "),
        "platform": platform.platform(),
        "package_versions": pip_freeze(),
        "cuda": torch_environment(),
        "model_names": {
            "classical": CLASSICAL_MODELS,
            "transformers": TRANSFORMER_MODEL_NAMES,
            "requested": models,
        },
        "seeds": {
            "base_seed": args.seed,
            "fold_seed_rule": "base_seed + split_number - 1",
        },
        "folds": args.folds,
        "repeats": args.repeats,
        "test_size": args.test_size,
        "data_source_paths": data_source_paths(),
        "start_timestamp": start_time.isoformat(timespec="seconds"),
        "end_timestamp": None,
        "total_runtime_seconds": None,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "config_signature": model_config_signature(args, models),
        "oom_notes": [],
        "errors": [],
    }
    if prior_manifest is not None:
        training_provenance = prior_manifest.get("training_run_provenance")
        if isinstance(training_provenance, dict):
            manifest["training_run_provenance"] = training_provenance
    write_manifest(run_dir, manifest)
    return manifest


def read_article_csv(path: Path, source: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty dataset file: {path}")
    frame = pd.read_csv(path, usecols=["label", "article"])
    if frame.empty:
        raise ValueError(f"Dataset has no rows: {path}")
    if frame[["label", "article"]].isna().any().any():
        raise ValueError(f"Dataset has missing label/article values: {path}")
    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    frame["article"] = frame["article"].astype(str)
    labels = set(frame["label"].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected labels {{0, 1}} in {path}, got {labels}")
    frame["source_dataset"] = source
    frame["source_row_number"] = np.arange(len(frame), dtype=int)
    frame["combined_row_id"] = [
        f"{source}:{index:06d}" for index in frame["source_row_number"].to_numpy()
    ]
    return frame


def build_joint_frame() -> pd.DataFrame:
    from rerun_common import load_dataset

    X, y, _ = load_dataset("Joint", include_lex_morph=True)
    X = X.reset_index(drop=True).copy()
    y = y.reset_index(drop=True).astype(int)

    cruz_articles = read_article_csv(ARTICLE_PATHS["Cruz"], "Cruz")
    lupac_articles = read_article_csv(ARTICLE_PATHS["Lupac"], "Lupac")
    article_frame = pd.concat([cruz_articles, lupac_articles], ignore_index=True)

    if len(X) != len(article_frame):
        raise ValueError(
            f"Feature/article row mismatch: features={len(X)} articles={len(article_frame)}"
        )
    if not X["article"].astype(str).equals(article_frame["article"].astype(str)):
        raise ValueError("Feature frame article order does not match Cruz then Lupac articles")
    if not y.equals(article_frame["label"].astype(int)):
        raise ValueError("Feature frame labels do not match article labels")

    joint = X.copy()
    joint["label"] = y
    joint["source_dataset"] = article_frame["source_dataset"].to_numpy()
    joint["source_row_number"] = article_frame["source_row_number"].to_numpy()
    joint["combined_row_id"] = article_frame["combined_row_id"].to_numpy()
    joint["combined_row_number"] = np.arange(len(joint), dtype=int)
    return joint


def make_train_partition(
    joint: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, holdout_frame = train_test_split(
        joint,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=joint["label"],
    )
    train_frame = train_frame.reset_index(drop=True).copy()
    holdout_frame = holdout_frame.reset_index(drop=True).copy()
    train_frame["train_partition_position"] = np.arange(len(train_frame), dtype=int)
    holdout_frame["train_partition_position"] = np.nan
    return train_frame, holdout_frame


def make_fold_specs(train_frame: pd.DataFrame, args: argparse.Namespace) -> list[FoldSpec]:
    cv = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )
    specs: list[FoldSpec] = []
    for split_number, (train_index, val_index) in enumerate(
        cv.split(train_frame["article"], train_frame["label"]),
        start=1,
    ):
        repeat = ((split_number - 1) // args.folds) + 1
        fold = ((split_number - 1) % args.folds) + 1
        specs.append(
            FoldSpec(
                split_number=split_number,
                repeat=repeat,
                fold=fold,
                train_index=train_index,
                val_index=val_index,
            )
        )
    if args.fold_limit is not None:
        specs = specs[: args.fold_limit]
    return specs


def save_fold_splits(
    run_dir: Path,
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    specs: list[FoldSpec],
) -> Path:
    path = run_dir / "fold_splits.csv"
    if path.exists():
        return path

    rows: list[dict[str, object]] = []
    for role, frame in [("outer_train", train_frame), ("outer_holdout", holdout_frame)]:
        for _, row in frame.iterrows():
            rows.append(
                {
                    "split_number": 0,
                    "repeat": 0,
                    "fold": 0,
                    "role": role,
                    "combined_row_number": int(row["combined_row_number"]),
                    "train_partition_position": (
                        ""
                        if pd.isna(row["train_partition_position"])
                        else int(row["train_partition_position"])
                    ),
                    "combined_row_id": row["combined_row_id"],
                    "y_true": int(row["label"]),
                }
            )

    for spec in specs:
        for role, indices in [
            ("cv_train", spec.train_index),
            ("cv_validation", spec.val_index),
        ]:
            fold_frame = train_frame.iloc[indices]
            for _, row in fold_frame.iterrows():
                rows.append(
                    {
                        "split_number": spec.split_number,
                        "repeat": spec.repeat,
                        "fold": spec.fold,
                        "role": role,
                        "combined_row_number": int(row["combined_row_number"]),
                        "train_partition_position": int(row["train_partition_position"]),
                        "combined_row_id": row["combined_row_id"],
                        "y_true": int(row["label"]),
                    }
                )

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def best_params_path() -> Path:
    return TRAINING_DIR / "results" / "stopwords_fix_rerun" / "tuned_best_params.json"


def load_best_params() -> dict | None:
    path = best_params_path()
    if not path.exists():
        return None
    return read_json(path)


def prediction_path(run_dir: Path, model: str, spec: FoldSpec) -> Path:
    return (
        run_dir
        / "predictions"
        / model
        / f"repeat_{spec.repeat}_fold_{spec.fold}_predictions.csv.gz"
    )


def roc_path(run_dir: Path, model: str, spec: FoldSpec) -> Path:
    return (
        run_dir
        / "roc_raw"
        / model
        / f"repeat_{spec.repeat}_fold_{spec.fold}_roc.csv"
    )


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FOLD_METRIC_COLUMNS)
    frame = pd.read_csv(path)
    for column in FOLD_METRIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[FOLD_METRIC_COLUMNS]


def metric_key(row: pd.Series | dict[str, object]) -> tuple[str, int, int, int]:
    return (
        str(row["model"]),
        int(row["repeat"]),
        int(row["fold"]),
        int(row["split_number"]),
    )


def has_matching_metric(metrics_path: Path, model: str, spec: FoldSpec) -> bool:
    metrics = read_metrics(metrics_path)
    if metrics.empty:
        return False
    mask = (
        (metrics["model"].astype(str) == model)
        & (metrics["repeat"].astype(int) == spec.repeat)
        & (metrics["fold"].astype(int) == spec.fold)
        & (metrics["split_number"].astype(int) == spec.split_number)
    )
    return bool(mask.any())


def valid_prediction_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        frame = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return set(PREDICTION_COLUMNS).issubset(frame.columns) and not frame.empty


def valid_roc_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        frame = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return {"model", "repeat", "fold", "fpr", "tpr", "threshold", "roc_auc"}.issubset(
        frame.columns
    )


def fold_complete(run_dir: Path, metrics_path: Path, model: str, spec: FoldSpec) -> bool:
    return (
        valid_prediction_file(prediction_path(run_dir, model, spec))
        and valid_roc_file(roc_path(run_dir, model, spec))
        and has_matching_metric(metrics_path, model, spec)
    )


def atomic_to_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    frame.to_csv(temp_path, index=False, **kwargs)
    temp_path.replace(path)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def save_roc_file(
    run_dir: Path,
    model: str,
    spec: FoldSpec,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[Path, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    auc = float(roc_auc_score(y_true, y_score))
    frame = pd.DataFrame(
        {
            "model": model,
            "repeat": spec.repeat,
            "fold": spec.fold,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
            "roc_auc": auc,
        }
    )
    path = roc_path(run_dir, model, spec)
    atomic_to_csv(frame, path)
    return path, auc


def build_prediction_frame(
    model: str,
    spec: FoldSpec,
    val_frame: pd.DataFrame,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    elapsed_seconds: float,
    batch_size: int | str,
    eval_batch_size: int | str,
    logits: np.ndarray | None = None,
) -> pd.DataFrame:
    if logits is None:
        logit_0: list[object] = [""] * len(val_frame)
        logit_1: list[object] = [""] * len(val_frame)
    else:
        logit_0 = logits[:, 0].astype(float).tolist()
        logit_1 = logits[:, 1].astype(float).tolist()
    return pd.DataFrame(
        {
            "model": model,
            "repeat": spec.repeat,
            "fold": spec.fold,
            "split_number": spec.split_number,
            "seed": spec.seed,
            "combined_row_id": val_frame["combined_row_id"].to_numpy(),
            "y_true": val_frame["label"].astype(int).to_numpy(),
            "y_score": y_score.astype(float),
            "y_pred": y_pred.astype(int),
            "logit_0": logit_0,
            "logit_1": logit_1,
            "elapsed_seconds": elapsed_seconds,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
        }
    )[PREDICTION_COLUMNS]


def upsert_metric_row(metrics_path: Path, row: dict[str, object]) -> None:
    metrics = read_metrics(metrics_path)
    if metrics.empty:
        metrics = pd.DataFrame([row])
    else:
        mask = (
            (metrics["model"].astype(str) == str(row["model"]))
            & (metrics["repeat"].astype(int) == int(row["repeat"]))
            & (metrics["fold"].astype(int) == int(row["fold"]))
            & (metrics["split_number"].astype(int) == int(row["split_number"]))
        )
        metrics = metrics.loc[~mask].copy()
        metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
    order_map = {model: index for index, model in enumerate(MODEL_ORDER)}
    metrics["_model_order"] = metrics["model"].map(order_map).fillna(999)
    metrics = metrics.sort_values(["_model_order", "split_number"]).drop(columns=["_model_order"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False, columns=FOLD_METRIC_COLUMNS)


def persist_fold_outputs(
    run_dir: Path,
    metrics_path: Path,
    model: str,
    spec: FoldSpec,
    val_frame: pd.DataFrame,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    elapsed_seconds: float,
    batch_size: int | str,
    eval_batch_size: int | str,
    logits: np.ndarray | None = None,
) -> None:
    pred_frame = build_prediction_frame(
        model=model,
        spec=spec,
        val_frame=val_frame,
        y_score=y_score,
        y_pred=y_pred,
        elapsed_seconds=elapsed_seconds,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        logits=logits,
    )
    pred_path = prediction_path(run_dir, model, spec)
    atomic_to_csv(pred_frame, pred_path, compression="gzip")
    raw_roc_path, auc = save_roc_file(
        run_dir,
        model,
        spec,
        pred_frame["y_true"].to_numpy(dtype=int),
        pred_frame["y_score"].to_numpy(dtype=float),
    )
    metric_values = compute_metrics(
        pred_frame["y_true"].to_numpy(dtype=int),
        pred_frame["y_score"].to_numpy(dtype=float),
        pred_frame["y_pred"].to_numpy(dtype=int),
    )
    metric_values["roc_auc"] = auc
    upsert_metric_row(
        metrics_path,
        {
            "model": model,
            "repeat": spec.repeat,
            "fold": spec.fold,
            "split_number": spec.split_number,
            "seed": spec.seed,
            "n_validation": len(pred_frame),
            "elapsed_seconds": elapsed_seconds,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "prediction_path": str(pred_path.relative_to(run_dir)).replace("\\", "/"),
            "roc_path": str(raw_roc_path.relative_to(run_dir)).replace("\\", "/"),
            **metric_values,
        },
    )


def classifier_for_fold(classifier: object, fold_seed: int) -> object:
    estimator = clone(classifier)
    try:
        params = estimator.get_params()
        if "random_state" in params:
            estimator.set_params(random_state=fold_seed)
        if "n_jobs" in params and estimator.__class__.__name__ != "LogisticRegression":
            estimator.set_params(n_jobs=1)
    except Exception:
        pass
    return estimator


def run_classical_fold(
    run_dir: Path,
    metrics_path: Path,
    model: str,
    classifier: object,
    train_frame: pd.DataFrame,
    spec: FoldSpec,
) -> None:
    fold_train = train_frame.iloc[spec.train_index]
    fold_val = train_frame.iloc[spec.val_index]
    feature_columns = [column for column in train_frame.columns if column != "label"]
    X_train = fold_train[feature_columns]
    y_train = fold_train["label"].astype(int)
    X_val = fold_val[feature_columns]
    y_true = fold_val["label"].astype(int).to_numpy()

    pipeline = make_pipeline(
        classifier_for_fold(classifier, spec.seed),
        include_lex_morph=True,
    )
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    y_score = positive_scores(pipeline, X_val)
    y_pred = pipeline.predict(X_val).astype(int)
    elapsed = time.perf_counter() - started

    persist_fold_outputs(
        run_dir=run_dir,
        metrics_path=metrics_path,
        model=model,
        spec=spec,
        val_frame=fold_val,
        y_score=np.asarray(y_score, dtype=float),
        y_pred=np.asarray(y_pred, dtype=int),
        elapsed_seconds=elapsed,
        batch_size="",
        eval_batch_size="",
        logits=None,
    )
    auc = roc_auc_score(y_true, y_score)
    print(
        f"{model}: repeat={spec.repeat} fold={spec.fold} split={spec.split_number} "
        f"auc={auc:.9f} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )


def make_encoded_dataset(frame: pd.DataFrame, tokenizer, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_pandas(frame[["article", "label"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["article"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    encoded = dataset.map(tokenize, batched=True, desc="Tokenizing")
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return encoded


def training_args_for_transformer(
    fold_output_dir: Path,
    args: argparse.Namespace,
    fold_seed: int,
    train_size: int,
    train_batch_size: int,
    eval_batch_size: int,
):
    import torch
    from transformers import TrainingArguments

    steps_per_epoch = math.ceil(train_size / train_batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    kwargs = {
        "output_dir": str(fold_output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "optim": "adamw_torch",
        "save_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "logging_first_step": True,
        "report_to": "none",
        "fp16": torch.cuda.is_available() and not args.no_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_accumulation_steps": 1,
        "dataloader_num_workers": 0,
        "seed": fold_seed,
        "data_seed": fold_seed,
        "remove_unused_columns": True,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**kwargs)


def make_transformer_model(model_name: str, seed: int):
    from transformers import AutoModelForSequenceClassification, set_seed

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Fake", 1: "Real"},
        label2id={"Fake": 0, "Real": 1},
    )
    model.config.problem_type = "single_label_classification"
    return model


def predict_logits(model, dataset, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            labels.append(batch_labels.cpu().numpy())
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            logits.append(output.logits.detach().cpu().numpy())
            del batch
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return np.concatenate(labels, axis=0), np.concatenate(logits, axis=0)


def clear_torch_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def fallback_batch_sizes(start: int) -> list[int]:
    values = [start]
    for candidate in [16, 8, 4, 2, 1]:
        if candidate < start and candidate not in values:
            values.append(candidate)
    return values


def trainer_kwargs(model, training_args, train_dataset, val_dataset, tokenizer) -> dict:
    from transformers import Trainer

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
    }
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def run_transformer_fold(
    run_dir: Path,
    metrics_path: Path,
    manifest: dict,
    model: str,
    tokenizer,
    train_frame: pd.DataFrame,
    spec: FoldSpec,
    args: argparse.Namespace,
    current_batch_size: int,
) -> int:
    from transformers import Trainer

    fold_train = train_frame.iloc[spec.train_index].reset_index(drop=True)
    fold_val = train_frame.iloc[spec.val_index].reset_index(drop=True)
    train_dataset = make_encoded_dataset(fold_train, tokenizer, args.max_length)
    val_dataset = make_encoded_dataset(fold_val, tokenizer, args.max_length)
    model_name = TRANSFORMER_MODEL_NAMES[model]

    for batch_size in fallback_batch_sizes(current_batch_size):
        eval_batch_size = min(args.eval_batch_size, batch_size)
        fold_dir = (
            run_dir
            / "_tmp_transformer_runs"
            / model
            / f"repeat_{spec.repeat}_fold_{spec.fold}"
        )
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        trainer = None
        transformer_model = None
        try:
            transformer_model = make_transformer_model(model_name, spec.seed)
            training_args = training_args_for_transformer(
                fold_output_dir=fold_dir,
                args=args,
                fold_seed=spec.seed,
                train_size=len(fold_train),
                train_batch_size=batch_size,
                eval_batch_size=eval_batch_size,
            )
            trainer = Trainer(
                **trainer_kwargs(
                    transformer_model,
                    training_args,
                    train_dataset,
                    val_dataset,
                    tokenizer,
                )
            )
            started = time.perf_counter()
            trainer.train()
            clear_torch_memory()
            labels, logits = predict_logits(trainer.model, val_dataset, eval_batch_size)
            elapsed = time.perf_counter() - started

            probabilities = softmax(logits)
            y_score = probabilities[:, 1]
            y_pred = np.argmax(logits, axis=1)
            expected = fold_val["label"].astype(int).to_numpy()
            if not np.array_equal(labels.astype(int), expected):
                raise RuntimeError(f"{model} label order mismatch for split {spec.split_number}")

            persist_fold_outputs(
                run_dir=run_dir,
                metrics_path=metrics_path,
                model=model,
                spec=spec,
                val_frame=fold_val,
                y_score=y_score,
                y_pred=y_pred,
                elapsed_seconds=elapsed,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                logits=logits,
            )
            auc = roc_auc_score(expected, y_score)
            if batch_size < current_batch_size:
                print(
                    f"{model}: using reduced batch size for remaining folds "
                    f"batch_size={batch_size} eval_batch_size={eval_batch_size}",
                    flush=True,
                )
            print(
                f"{model}: repeat={spec.repeat} fold={spec.fold} "
                f"split={spec.split_number} auc={auc:.9f} "
                f"elapsed_seconds={elapsed:.1f} batch_size={batch_size}",
                flush=True,
            )
            return batch_size
        except Exception as exc:
            clear_torch_memory()
            if is_cuda_oom(exc) and batch_size > 1:
                next_size = fallback_batch_sizes(batch_size)[1]
                note = {
                    "model": model,
                    "repeat": spec.repeat,
                    "fold": spec.fold,
                    "split_number": spec.split_number,
                    "failed_batch_size": batch_size,
                    "retry_batch_size": next_size,
                    "error": str(exc).splitlines()[0],
                }
                manifest.setdefault("oom_notes", []).append(note)
                write_manifest(run_dir, manifest)
                print(
                    f"CUDA OOM: model={model} repeat={spec.repeat} fold={spec.fold} "
                    f"failed_batch_size={batch_size} retry_batch_size={next_size}",
                    flush=True,
                )
                continue

            error = {
                "model": model,
                "repeat": spec.repeat,
                "fold": spec.fold,
                "split_number": spec.split_number,
                "batch_size": batch_size,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            manifest.setdefault("errors", []).append(error)
            write_manifest(run_dir, manifest)
            raise RuntimeError(
                f"{model} fold failed at repeat={spec.repeat} fold={spec.fold} "
                f"split={spec.split_number} batch_size={batch_size}"
            ) from exc
        finally:
            del trainer
            del transformer_model
            clear_torch_memory()
            if fold_dir.exists():
                shutil.rmtree(fold_dir, ignore_errors=True)

    raise RuntimeError(
        f"{model} fold failed after exhausting batch sizes at split {spec.split_number}"
    )


def run_model(
    run_dir: Path,
    metrics_path: Path,
    manifest: dict,
    model: str,
    train_frame: pd.DataFrame,
    specs: list[FoldSpec],
    args: argparse.Namespace,
    classical_classifiers: dict[str, object],
) -> None:
    print(f"\n=== Model: {model} ===", flush=True)
    completed = 0

    if model in CLASSICAL_MODELS:
        classifier = classical_classifiers[model]
        for spec in specs:
            if not args.force and fold_complete(run_dir, metrics_path, model, spec):
                completed += 1
                print(
                    f"{model}: skipping completed repeat={spec.repeat} "
                    f"fold={spec.fold} split={spec.split_number}",
                    flush=True,
                )
                continue
            run_classical_fold(run_dir, metrics_path, model, classifier, train_frame, spec)
            completed += 1
        print(f"{model}: completed_or_skipped_folds={completed}/{len(specs)}", flush=True)
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAMES[model])
    current_batch_size = args.batch_size
    for spec in specs:
        if not args.force and fold_complete(run_dir, metrics_path, model, spec):
            completed += 1
            print(
                f"{model}: skipping completed repeat={spec.repeat} "
                f"fold={spec.fold} split={spec.split_number}",
                flush=True,
            )
            continue
        current_batch_size = run_transformer_fold(
            run_dir=run_dir,
            metrics_path=metrics_path,
            manifest=manifest,
            model=model,
            tokenizer=tokenizer,
            train_frame=train_frame,
            spec=spec,
            args=args,
            current_batch_size=current_batch_size,
        )
        completed += 1
    print(f"{model}: completed_or_skipped_folds={completed}/{len(specs)}", flush=True)


def iter_prediction_files(run_dir: Path, model: str) -> list[Path]:
    model_dir = run_dir / "predictions" / model
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("repeat_*_fold_*_predictions.csv.gz"))


def prediction_metric_row(run_dir: Path, path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_score = frame["y_score"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    values = compute_metrics(y_true, y_score, y_pred)
    first = frame.iloc[0]
    model = str(first["model"])
    spec_like = {
        "model": model,
        "repeat": int(first["repeat"]),
        "fold": int(first["fold"]),
        "split_number": int(first["split_number"]),
        "seed": int(first["seed"]),
        "n_validation": len(frame),
        "elapsed_seconds": float(pd.to_numeric(frame["elapsed_seconds"]).max()),
        "batch_size": first["batch_size"] if "batch_size" in frame.columns else "",
        "eval_batch_size": first["eval_batch_size"] if "eval_batch_size" in frame.columns else "",
        "prediction_path": str(path.relative_to(run_dir)).replace("\\", "/"),
    }
    raw_roc = (
        run_dir
        / "roc_raw"
        / model
        / f"repeat_{int(first['repeat'])}_fold_{int(first['fold'])}_roc.csv"
    )
    if raw_roc.exists():
        roc_frame = pd.read_csv(raw_roc)
        if not roc_frame.empty:
            values["roc_auc"] = float(roc_frame["roc_auc"].iloc[0])
    spec_like["roc_path"] = str(raw_roc.relative_to(run_dir)).replace("\\", "/")
    return {**spec_like, **values}


def rebuild_metrics_from_predictions(run_dir: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        for path in iter_prediction_files(run_dir, model):
            rows.append(prediction_metric_row(run_dir, path))
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        metrics = pd.DataFrame(columns=FOLD_METRIC_COLUMNS)
    else:
        order_map = {model: index for index, model in enumerate(MODEL_ORDER)}
        metrics["_model_order"] = metrics["model"].map(order_map).fillna(999)
        metrics = metrics.sort_values(["_model_order", "split_number"]).drop(
            columns=["_model_order"]
        )
    metrics.to_csv(run_dir / "fold_metrics.csv", index=False, columns=FOLD_METRIC_COLUMNS)
    return metrics


def interpolate_prediction_curve(path: Path) -> tuple[str, np.ndarray, float]:
    frame = pd.read_csv(path)
    model = str(frame["model"].iloc[0])
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_score = frame["y_score"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    auc = float(roc_auc_score(y_true, y_score))
    return model, interp_tpr, auc


def make_roc_tpr_summary(run_dir: Path, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in models:
        tprs = []
        aucs = []
        for path in iter_prediction_files(run_dir, model):
            _, interp_tpr, auc = interpolate_prediction_curve(path)
            tprs.append(interp_tpr)
            aucs.append(auc)
        if not tprs:
            continue
        tpr_matrix = np.vstack(tprs)
        auc_array = np.asarray(aucs, dtype=float)
        mean_tpr = tpr_matrix.mean(axis=0)
        sd_tpr = (
            tpr_matrix.std(axis=0, ddof=1)
            if len(tprs) > 1
            else np.zeros_like(mean_tpr)
        )
        mean_auc = float(auc_array.mean())
        sd_auc = float(auc_array.std(ddof=1)) if len(auc_array) > 1 else 0.0
        for mean_fpr, tpr_value, sd_value in zip(MEAN_FPR, mean_tpr, sd_tpr):
            rows.append(
                {
                    "model": model,
                    "mean_fpr": mean_fpr,
                    "mean_tpr": tpr_value,
                    "sd_tpr": sd_value,
                    "lower_tpr": max(tpr_value - sd_value, 0.0),
                    "upper_tpr": min(tpr_value + sd_value, 1.0),
                    "n_folds": len(tprs),
                    "mean_auc": mean_auc,
                    "sd_auc": sd_auc,
                }
            )
    summary = pd.DataFrame(
        rows,
        columns=[
            "model",
            "mean_fpr",
            "mean_tpr",
            "sd_tpr",
            "lower_tpr",
            "upper_tpr",
            "n_folds",
            "mean_auc",
            "sd_auc",
        ],
    )
    summary.to_csv(run_dir / "roc_tpr_summary.csv", index=False)
    return summary


def summarize_model_auc(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        summary = pd.DataFrame(
            columns=[
                "model",
                "n_folds",
                "mean_auc",
                "sd_auc",
                "sem_auc",
                "ci95_low",
                "ci95_high",
                "median_auc",
                "min_auc",
                "max_auc",
                "q25_auc",
                "q75_auc",
                "mean_accuracy",
                "sd_accuracy",
                "mean_f1_macro",
                "sd_f1_macro",
            ]
        )
        return summary
    for model, group in metrics.groupby("model", sort=False):
        aucs = group["roc_auc"].astype(float)
        n = len(aucs)
        sd_auc = float(aucs.std(ddof=1)) if n > 1 else 0.0
        sem_auc = float(sd_auc / math.sqrt(n)) if n > 0 else 0.0
        if n > 1:
            margin = float(stats.t.ppf(0.975, df=n - 1) * sem_auc)
        else:
            margin = 0.0
        accuracy = group["accuracy"].astype(float)
        f1_macro = group["f1_macro"].astype(float)
        rows.append(
            {
                "model": model,
                "n_folds": n,
                "mean_auc": float(aucs.mean()),
                "sd_auc": sd_auc,
                "sem_auc": sem_auc,
                "ci95_low": float(aucs.mean() - margin),
                "ci95_high": float(aucs.mean() + margin),
                "median_auc": float(aucs.median()),
                "min_auc": float(aucs.min()),
                "max_auc": float(aucs.max()),
                "q25_auc": float(aucs.quantile(0.25)),
                "q75_auc": float(aucs.quantile(0.75)),
                "mean_accuracy": float(accuracy.mean()),
                "sd_accuracy": float(accuracy.std(ddof=1)) if n > 1 else 0.0,
                "mean_f1_macro": float(f1_macro.mean()),
                "sd_f1_macro": float(f1_macro.std(ddof=1)) if n > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(rows)
    order_map = {model: index for index, model in enumerate(MODEL_ORDER)}
    summary["_model_order"] = summary["model"].map(order_map).fillna(999)
    summary = summary.sort_values("_model_order").drop(columns=["_model_order"])
    return summary


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_roc_curves(
    run_dir: Path,
    roc_summary: pd.DataFrame,
    auc_summary: pd.DataFrame,
    model_order: list[str],
    title: str,
    output_stem: str,
) -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    auc_lookup = {
        row["model"]: (float(row["mean_auc"]), float(row["sd_auc"]))
        for _, row in auc_summary.iterrows()
    }

    for model in model_order:
        rows = roc_summary[roc_summary["model"] == model]
        if rows.empty:
            continue
        mean_auc, sd_auc = auc_lookup[model]
        color = MODEL_COLORS[model]
        linestyle = "--" if model in TRANSFORMER_MODELS else "-"
        label = f"{model} (AUC = {mean_auc:.3f} {PM} {sd_auc:.3f})"
        ax.fill_between(
            rows["mean_fpr"].to_numpy(dtype=float),
            rows["lower_tpr"].to_numpy(dtype=float),
            rows["upper_tpr"].to_numpy(dtype=float),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            rows["mean_fpr"].to_numpy(dtype=float),
            rows["mean_tpr"].to_numpy(dtype=float),
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=label,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1.25)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, color="0.82", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.75")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    fig.tight_layout()

    png_path = run_dir / f"{output_stem}.png"
    pdf_path = run_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}", flush=True)
    print(f"Saved {pdf_path}", flush=True)


def write_report(
    run_dir: Path,
    models: list[str],
    metrics: pd.DataFrame,
    auc_summary: pd.DataFrame,
) -> Path:
    report_path = run_dir / "roc_report.md"
    lines: list[str] = []
    lines.append("# True 30-Fold Joint-Corpus ROC-AUC Report")
    lines.append("")
    lines.append(
        "These are true mean ROC curves with shaded +/- 1 SD bands computed over "
        "the saved 30 unified CV folds. In the full run, each model contributes "
        "30 fold-level probability/logit files generated from the same "
        "stratified 80/20 training partition and the same repeated stratified "
        "CV splits."
    )
    lines.append("")
    lines.append(
        "Possible differences from prior manuscript tables are expected because "
        "the unified-split statistics may differ from earlier tables "
        "that used earlier protocol-specific runs. The values here prioritize "
        "apples-to-apples comparison across classical ML and transformer models."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Joint corpus order: Cruz rows, then Lupac rows, with stable `combined_row_id`.")
    lines.append(
        "- Outer partition: `train_test_split(test_size=0.2, stratify=y, random_state=42)`."
    )
    lines.append(
        "- Inner evaluation: `RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=42)`."
    )
    lines.append("- Positive ROC score target: class `1`.")
    lines.append("- Figures are regenerated from saved fold prediction files.")
    lines.append("")
    lines.append("## AUC Summary")
    lines.append("")
    lines.append(
        "| Model | n folds | Mean AUC | SD AUC | 95% CI | Mean accuracy | Mean macro F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in auc_summary.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['n_folds'])} | "
            f"{float(row['mean_auc']):.6f} | {float(row['sd_auc']):.6f} | "
            f"{float(row['ci95_low']):.6f}-{float(row['ci95_high']):.6f} | "
            f"{float(row['mean_accuracy']):.6f} | {float(row['mean_f1_macro']):.6f} |"
        )
    lines.append("")
    lines.append("## Required Outputs")
    lines.append("")
    lines.append(
        "Fold-level prediction files and raw ROC point files are saved for "
        "independent audit and replotting."
    )
    lines.append("")
    for filename in [
        "run_true_30fold_roc_auc.py",
        "run_manifest.json",
        "run.log",
        "fold_splits.csv",
        "fold_metrics.csv",
        "model_auc_summary.csv",
        "roc_tpr_summary.csv",
        "roc_auc_ml_models.png",
        "roc_auc_ml_models.pdf",
        "roc_auc_all_models.png",
        "roc_auc_all_models.pdf",
    ]:
        lines.append(f"- `{filename}`")
    lines.append("- `predictions/<model>/repeat_<r>_fold_<f>_predictions.csv.gz`")
    lines.append("- `roc_raw/<model>/repeat_<r>_fold_<f>_roc.csv`")
    lines.append("")
    lines.append("## Figure Caption Wording")
    lines.append("")
    lines.append(
        f"Figure. ROC Curves {TITLE_DASH} Classical ML Classifiers "
        "(Joint Corpus, 30-Run CV). Lines show the mean ROC curve across unified "
        "CV folds; shaded bands show +/- 1 SD."
    )
    lines.append("")
    lines.append(
        f"Figure. ROC Curves {TITLE_DASH} All Models (Joint Corpus, 30-Run CV). "
        "Classical ML models are shown with solid lines and transformer models "
        "with dashed lines. Legend entries report mean ROC-AUC +/- SD from the "
        "saved fold-level predictions."
    )
    lines.append("")
    lines.append("## Completion")
    lines.append("")
    expected_full = len(MODEL_ORDER) * N_SPLITS * N_REPEATS
    lines.append(f"- Requested models: {', '.join(models)}")
    lines.append(f"- Saved fold metric rows: {len(metrics)}")
    lines.append(f"- Full all-model target rows: {expected_full}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def summarize_and_plot(run_dir: Path, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = rebuild_metrics_from_predictions(run_dir, models)
    auc_summary = summarize_model_auc(metrics)
    auc_summary.to_csv(run_dir / "model_auc_summary.csv", index=False)
    roc_summary = make_roc_tpr_summary(run_dir, models)

    present_classical = [
        model for model in ["MNB", "RF", "LR", "SVC"] if model in set(auc_summary["model"])
    ]
    if present_classical:
        plot_roc_curves(
            run_dir=run_dir,
            roc_summary=roc_summary,
            auc_summary=auc_summary,
            model_order=present_classical,
            title=f"ROC Curves {TITLE_DASH} Classical ML Classifiers (Joint Corpus, 30-Run CV)",
            output_stem="roc_auc_ml_models",
        )

    present_all = auc_summary.sort_values("mean_auc", ascending=False)["model"].tolist()
    if present_all:
        plot_roc_curves(
            run_dir=run_dir,
            roc_summary=roc_summary,
            auc_summary=auc_summary,
            model_order=present_all,
            title=f"ROC Curves {TITLE_DASH} All Models (Joint Corpus, 30-Run CV)",
            output_stem="roc_auc_all_models",
        )
    write_report(run_dir, models, metrics, auc_summary)
    return metrics, auc_summary


def dry_run(args: argparse.Namespace, models: list[str], run_dir: Path) -> None:
    joint = build_joint_frame()
    train_frame, holdout_frame = make_train_partition(joint, args)
    specs = make_fold_specs(train_frame, args)
    print("Dry run OK")
    print(f"training_dir={TRAINING_DIR}")
    print(f"output_root={OUTPUT_ROOT}")
    print(f"resolved_run_dir={run_dir}")
    print(f"models={models}")
    print(f"joint_rows={len(joint)} class_counts={joint['label'].value_counts().to_dict()}")
    print(
        f"train_rows={len(train_frame)} holdout_rows={len(holdout_frame)} "
        f"folds={len(specs)}"
    )
    if specs:
        first = specs[0]
        print(
            f"first_split repeat={first.repeat} fold={first.fold} "
            f"train={len(first.train_index)} validation={len(first.val_index)}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.seed != RANDOM_STATE:
        raise ValueError("This reproducible appendix run requires --seed 42")
    if args.folds != N_SPLITS:
        raise ValueError("This reproducible appendix run requires --folds 5")
    if args.repeats != N_REPEATS:
        raise ValueError("This reproducible appendix run requires --repeats 6")
    if abs(args.test_size - TEST_SIZE) > 1e-12:
        raise ValueError("This reproducible appendix run requires --test-size 0.2")


def main() -> None:
    args = parse_args()
    validate_args(args)
    models = selected_models(args.models)
    run_dir = resolve_run_dir(args, models)

    if args.dry_run:
        dry_run(args, models, run_dir)
        return

    start_time = datetime.now()
    log_path = configure_logging(run_dir)
    manifest = initialize_manifest(args, models, run_dir, log_path, start_time)
    metrics_path = run_dir / "fold_metrics.csv"

    try:
        print("True 30-fold joint ROC-AUC run")
        print(f"started_at={start_time.isoformat(timespec='seconds')}")
        print(f"run_dir={run_dir}")
        print(f"models={models}")
        print(
            f"resume={args.resume} force={args.force} "
            f"aggregate_only={args.aggregate_only} fold_limit={args.fold_limit}"
        )
        print(f"python={sys.executable}")
        print(f"cuda={manifest['cuda']}")

        joint = build_joint_frame()
        train_frame, holdout_frame = make_train_partition(joint, args)
        specs = make_fold_specs(train_frame, args)
        save_fold_splits(run_dir, train_frame, holdout_frame, specs)
        print(
            f"joint_rows={len(joint)} train_rows={len(train_frame)} "
            f"holdout_rows={len(holdout_frame)} cv_folds={len(specs)}"
        )
        print(f"class_counts={joint['label'].value_counts().to_dict()}")

        if args.aggregate_only:
            print("aggregate_only=True; skipped model execution")
        else:
            classical_classifiers = tuned_classifiers(
                load_best_params(),
                probability_for_svc=True,
            )
            for model in models:
                run_model(
                    run_dir=run_dir,
                    metrics_path=metrics_path,
                    manifest=manifest,
                    model=model,
                    train_frame=train_frame,
                    specs=specs,
                    args=args,
                    classical_classifiers=classical_classifiers,
                )
                write_manifest(run_dir, manifest)

        metrics, auc_summary = summarize_and_plot(run_dir, models)
        print("\nAUC summary")
        print(auc_summary.to_string(index=False))
        print(f"\nfold_metric_rows={len(metrics)}")

        manifest["status"] = "complete"
        manifest["end_timestamp"] = datetime.now().isoformat(timespec="seconds")
        manifest["total_runtime_seconds"] = (datetime.now() - start_time).total_seconds()
        if args.aggregate_only and isinstance(
            manifest.get("training_run_provenance"),
            dict,
        ):
            refresh = {
                "command_used": manifest.get("command_used"),
                "start_timestamp": manifest.get("start_timestamp"),
                "end_timestamp": manifest.get("end_timestamp"),
                "total_runtime_seconds": manifest.get("total_runtime_seconds"),
                "purpose": (
                    "Regenerated summaries, figures, report wording, copied script, "
                    "and output file list from saved prediction files."
                ),
            }
            training_provenance = manifest["training_run_provenance"]
            manifest["aggregate_refresh"] = refresh
            for key in [
                "command_used",
                "start_timestamp",
                "end_timestamp",
                "total_runtime_seconds",
            ]:
                if key in training_provenance:
                    manifest[key] = training_provenance[key]
        write_manifest(run_dir, manifest)
        print(f"Completed run: {run_dir}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["end_timestamp"] = datetime.now().isoformat(timespec="seconds")
        manifest["total_runtime_seconds"] = (datetime.now() - start_time).total_seconds()
        manifest.setdefault("errors", []).append(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_manifest(run_dir, manifest)
        raise


if __name__ == "__main__":
    main()
