"""Fine-tune DistilBERT-Tagalog for the FaKe reviewer baseline.

This script follows the evaluation layout used by the existing cross-validation
scripts in this directory:

* load the local Fake News Filipino CSV files under root/datasets;
* create the same stratified 80% training partition with random_state=42;
* evaluate six repetitions of five-fold cross-validation; and
* compute sklearn accuracy, F1, ROC-AUC, and a representative confusion matrix.

Run from the repository root with:

    training\venv\Scripts\python.exe training\distilbert_tagalog_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODEL_NAME = "jcblaise/distilbert-tagalog-base-cased"
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 6
TEST_SIZE = 0.2
DEFAULT_OUTPUT_DIR = Path("results") / "distilbert_tagalog_baseline"
PM = "\u00b1"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    article_path: Path


DATASET_SPECS = {
    "cruz": DatasetSpec(
        key="cruz",
        display_name="Fake News Filipino 2020",
        article_path=Path("root") / "datasets" / "Cruz" / "FakeNewsFilipino_Cruz2020.csv",
    ),
    "lupac": DatasetSpec(
        key="lupac",
        display_name="Fake News Filipino 2024",
        article_path=Path("root")
        / "datasets"
        / "Lupac"
        / "FakeNewsPhilippines2024_Lupac.csv",
    ),
    "combined": DatasetSpec(
        key="combined",
        display_name="Combined",
        article_path=Path(""),
    ),
}

CLASSICAL_ACCURACY_ROWS = [
    ("Logistic Regression", "Fake News Filipino 2020", "0.00960", "0.951"),
    ("", "Fake News Filipino 2024", "0.00987", "0.947"),
    ("", "Combined", "0.00939", "0.924"),
    ("Multinomial Naive Bayes", "Fake News Filipino 2020", "0.0111", "0.923"),
    ("", "Fake News Filipino 2024", "0.0157", "0.885"),
    ("", "Combined", "0.0140", "0.860"),
    ("Random Forest", "Fake News Filipino 2020", "0.0145", "0.919"),
    ("", "Fake News Filipino 2024", "0.0118", "0.926"),
    ("", "Combined", "0.0115", "0.888"),
    ("Support Vector Classifier", "Fake News Filipino 2020", "0.00913", "0.951"),
    ("", "Fake News Filipino 2024", "0.00945", "0.947"),
    ("", "Combined", "0.00907", "0.922"),
]

CLASSICAL_AUC_ROWS = [
    ("Multinomial Naive Bayes", f"0.925 {PM} 0.010"),
    ("Logistic Regression", f"0.976 {PM} 0.005"),
    ("Random Forest", f"0.960 {PM} 0.006"),
    ("Support Vector Classifier", f"0.973 {PM} 0.005"),
]


class Tee:
    """Write console output to the terminal and the run log."""

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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cruz", "lupac", "combined", "all"],
        default=["all"],
        help="Datasets to evaluate. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--folds", type=positive_int, default=N_SPLITS)
    parser.add_argument("--repeats", type=positive_int, default=N_REPEATS)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--eval-batch-size", type=positive_int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument(
        "--fold-limit",
        type=positive_int,
        default=None,
        help="Optional smoke-test limit. Omit for the full 30 folds per dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run completed folds instead of resuming from metrics.csv.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 mixed precision. Default uses fp16 when CUDA is available.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing if GPU memory is constrained.",
    )
    return parser.parse_args()


def selected_dataset_keys(raw_keys: Iterable[str]) -> list[str]:
    keys = list(raw_keys)
    if "all" in keys:
        return ["cruz", "lupac", "combined"]
    return keys


def configure_tee(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "distilbert_tagalog_baseline.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    warnings.simplefilter("default")
    return log_path


def read_article_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty dataset file: {path.resolve()}")

    frame = pd.read_csv(path, usecols=["label", "article"])
    if frame.empty:
        raise ValueError(f"Dataset has no rows: {path.resolve()}")
    if frame[["label", "article"]].isna().any().any():
        raise ValueError(f"Dataset has missing label/article values: {path.resolve()}")

    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    frame["article"] = frame["article"].astype(str)
    labels = set(frame["label"].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected labels {{0, 1}} in {path.resolve()}, got {labels}")
    return frame


def load_dataset_frame(dataset_key: str) -> tuple[str, pd.DataFrame]:
    if dataset_key == "combined":
        cruz = read_article_csv(DATASET_SPECS["cruz"].article_path)
        lupac = read_article_csv(DATASET_SPECS["lupac"].article_path)
        return DATASET_SPECS["combined"].display_name, pd.concat(
            [cruz, lupac], ignore_index=True
        )

    spec = DATASET_SPECS[dataset_key]
    return spec.display_name, read_article_csv(spec.article_path)


def make_encoded_dataset(
    frame: pd.DataFrame,
    tokenizer,
    max_length: int,
) -> Dataset:
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


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_prediction_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = softmax(logits)[:, 1]
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_fake": float(f1_score(labels, preds, pos_label=0)),
        "f1_real": float(f1_score(labels, preds, pos_label=1)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def completed_fold_keys(metrics_path: Path) -> set[tuple[str, int]]:
    if not metrics_path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["dataset_key"], int(row["fold_number"])))
    return completed


def append_metric_row(metrics_path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "dataset_key",
        "dataset",
        "fold_number",
        "repeat_number",
        "split_number",
        "seed",
        "train_size",
        "validation_size",
        "accuracy",
        "f1_macro",
        "f1_fake",
        "f1_real",
        "roc_auc",
        "elapsed_seconds",
    ]
    write_header = not metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_model(model_name: str, seed: int):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Fake", 1: "Real"},
        label2id={"Fake": 0, "Real": 1},
    )
    model.config.problem_type = "single_label_classification"
    return model


def training_args_for_fold(
    fold_output_dir: Path,
    args: argparse.Namespace,
    fold_seed: int,
    train_size: int,
) -> TrainingArguments:
    steps_per_epoch = math.ceil(train_size / args.batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    return TrainingArguments(
        output_dir=str(fold_output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        logging_first_step=True,
        report_to="none",
        fp16=torch.cuda.is_available() and not args.no_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_accumulation_steps=1,
        dataloader_num_workers=0,
        seed=fold_seed,
        data_seed=fold_seed,
        remove_unused_columns=True,
    )


def summarize(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def model_dir_size_mb(path: Path) -> float:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024**2)


def benchmark_inference(
    model_dir: Path,
    tokenizer,
    joint_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    print("\nInference benchmark on 30 joint-corpus articles")
    fake_sample = joint_frame[joint_frame["label"] == 0].sample(
        n=15, random_state=args.seed
    )
    real_sample = joint_frame[joint_frame["label"] == 1].sample(
        n=15, random_state=args.seed
    )
    sample = pd.concat([fake_sample, real_sample], ignore_index=True).sample(
        frac=1, random_state=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    warmup_texts = sample["article"].head(5).tolist()
    with torch.no_grad():
        for text in warmup_texts:
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            _ = model(**encoded)
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings_ms: list[float] = []
    predictions: list[int] = []
    with torch.no_grad():
        for idx, row in sample.reset_index(drop=True).iterrows():
            encoded = tokenizer(
                row["article"],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(**encoded)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            pred = int(torch.argmax(output.logits, dim=1).item())
            timings_ms.append(elapsed_ms)
            predictions.append(pred)
            print(
                "benchmark article "
                f"{idx + 1:02d}/30 actual={int(row['label'])} "
                f"pred={pred} elapsed_ms={elapsed_ms:.3f}"
            )

    timing_array = np.array(timings_ms, dtype=float)
    print(
        "Mean inference time per article: "
        f"{timing_array.mean():.3f} ms "
        f"(SD={timing_array.std(ddof=1):.3f} ms, n=30)"
    )
    return {
        "mean_ms": float(timing_array.mean()),
        "sd_ms": float(timing_array.std(ddof=1)),
        "median_ms": float(np.median(timing_array)),
        "timings_ms": timings_ms,
        "predictions": predictions,
        "labels": [int(value) for value in sample["label"].tolist()],
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def predict_with_model(
    model,
    dataset: Dataset,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def create_report(
    run_dir: Path,
    log_path: Path,
    metrics_path: Path,
    representative_cm: list[list[int]],
    parameter_count: int,
    model_size_mb: float,
    inference_summary: dict[str, object],
    args: argparse.Namespace,
) -> Path:
    metrics = pd.read_csv(metrics_path)
    report_path = run_dir / "distilbert_tagalog_baseline_results.md"

    lines: list[str] = []
    lines.append("# DistilBERT-Tagalog Baseline Results")
    lines.append("")
    lines.append(f"Model: `{args.model_name}`")
    lines.append(
        "Protocol: stratified 80% training partition, six repetitions of five-fold "
        "stratified cross-validation, `random_state = 42` base seed."
    )
    lines.append(
        "Hyperparameters: learning rate 2e-5, batch size 16, epochs 3, "
        "max length 512, weight decay 0.01, warmup ratio 0.10, AdamW."
    )
    lines.append("F1 values below use macro averaging across Fake and Real.")
    lines.append("")

    lines.append("## Accuracy Across Datasets")
    lines.append("")
    lines.append("| Classifier | Dataset | Standard Deviation | Accuracy |")
    lines.append("|---|---|---:|---:|")
    for classifier, dataset, sd, accuracy in CLASSICAL_ACCURACY_ROWS:
        lines.append(f"| {classifier} | {dataset} | {sd} | {accuracy} |")
    for dataset_key in ["cruz", "lupac", "combined"]:
        dataset_rows = metrics[metrics["dataset_key"] == dataset_key]
        if dataset_rows.empty:
            continue
        mean_acc, sd_acc = summarize(dataset_rows["accuracy"])
        dataset_name = str(dataset_rows["dataset"].iloc[0])
        classifier = "DistilBERT-Tagalog" if dataset_key == "cruz" else ""
        lines.append(f"| {classifier} | {dataset_name} | {sd_acc:.5f} | {mean_acc:.3f} |")
    lines.append("")

    lines.append("## DistilBERT F1 And ROC-AUC")
    lines.append("")
    lines.append("| Dataset | Accuracy | Macro F1 | ROC-AUC |")
    lines.append("|---|---:|---:|---:|")
    for dataset_key in ["cruz", "lupac", "combined"]:
        dataset_rows = metrics[metrics["dataset_key"] == dataset_key]
        if dataset_rows.empty:
            continue
        dataset_name = str(dataset_rows["dataset"].iloc[0])
        mean_acc, sd_acc = summarize(dataset_rows["accuracy"])
        mean_f1, sd_f1 = summarize(dataset_rows["f1_macro"])
        mean_auc, sd_auc = summarize(dataset_rows["roc_auc"])
        lines.append(
            f"| {dataset_name} | {mean_acc:.3f} {PM} {sd_acc:.3f} | "
            f"{mean_f1:.3f} {PM} {sd_f1:.3f} | {mean_auc:.3f} {PM} {sd_auc:.3f} |"
        )
    lines.append("")

    lines.append("## ROC-AUC On Joint Corpus")
    lines.append("")
    lines.append("| Classifier | ROC-AUC |")
    lines.append("|---|---:|")
    for classifier, auc in CLASSICAL_AUC_ROWS:
        lines.append(f"| {classifier} | {auc} |")
    joint_rows = metrics[metrics["dataset_key"] == "combined"]
    if not joint_rows.empty:
        mean_auc, sd_auc = summarize(joint_rows["roc_auc"])
        lines.append(f"| DistilBERT-Tagalog | {mean_auc:.3f} {PM} {sd_auc:.3f} |")
    lines.append("")

    lines.append("## Representative Confusion Matrix")
    lines.append("")
    lines.append("Joint corpus, fold 1. Labels: Fake = 0, Real = 1.")
    lines.append("")
    lines.append("| Classifier | Actual Class | Predicted Fake | Predicted Real |")
    lines.append("|---|---|---:|---:|")
    lines.append(
        "| DistilBERT-Tagalog | Actual Fake | "
        f"{representative_cm[0][0]} | {representative_cm[0][1]} |"
    )
    lines.append(
        "|  | Actual Real | "
        f"{representative_cm[1][0]} | {representative_cm[1][1]} |"
    )
    lines.append("")

    lines.append("## Model Size And Inference")
    lines.append("")
    lines.append("| Model | Parameters | Model Size On Disk |")
    lines.append("|---|---:|---:|")
    lines.append("| Logistic Regression | 1,200,000 | 55.64 MB |")
    lines.append(
        f"| DistilBERT-Tagalog | {parameter_count:,} | {model_size_mb:.2f} MB |"
    )
    lines.append("")
    lines.append(
        "Mean inference time per article on 30 joint-corpus articles: "
        f"{float(inference_summary['mean_ms']):.3f} ms "
        f"(SD={float(inference_summary['sd_ms']):.3f} ms, "
        f"median={float(inference_summary['median_ms']):.3f} ms)."
    )
    lines.append("")

    sys.stdout.flush()
    sys.stderr.flush()
    raw_log = log_path.read_text(encoding="utf-8", errors="replace")
    lines.append("## Raw Console Output")
    lines.append("")
    lines.append("```text")
    lines.append(raw_log.rstrip())
    lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_environment(args: argparse.Namespace, run_dir: Path) -> None:
    print("DistilBERT-Tagalog baseline run")
    print(f"started_at={datetime.now().isoformat(timespec='seconds')}")
    print(f"script_dir={SCRIPT_DIR}")
    print(f"run_dir={run_dir}")
    print(f"python={sys.version.replace(os.linesep, ' ')}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"model_name={args.model_name}")
    print(
        "cv="
        f"{args.repeats} repetitions x {args.folds} folds "
        f"(fold_limit={args.fold_limit})"
    )
    print(
        "hyperparameters="
        f"lr={args.learning_rate}, batch_size={args.batch_size}, "
        f"eval_batch_size={args.eval_batch_size}, epochs={args.epochs}, "
        f"max_length={args.max_length}, "
        f"weight_decay={args.weight_decay}, warmup_ratio={args.warmup_ratio}, "
        "optimizer=AdamW"
    )


def evaluate_dataset(
    dataset_key: str,
    dataset_name: str,
    frame: pd.DataFrame,
    tokenizer,
    run_dir: Path,
    metrics_path: Path,
    args: argparse.Namespace,
    completed: set[tuple[str, int]],
) -> tuple[list[list[int]] | None, Path | None, int | None]:
    print(f"\nDataset: {dataset_name} ({dataset_key})")
    print(f"rows={len(frame)} class_counts={frame['label'].value_counts().to_dict()}")
    train_frame, _ = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=args.seed,
        stratify=frame["label"],
    )
    train_frame = train_frame.reset_index(drop=True)
    print(
        f"training_partition_rows={len(train_frame)} "
        f"test_size={TEST_SIZE} random_state={args.seed}"
    )

    cv = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )

    representative_cm = None
    representative_model_dir = None
    representative_parameter_count = None

    total_folds = args.folds * args.repeats
    split_iter = cv.split(train_frame["article"], train_frame["label"])
    for fold_number, (train_index, val_index) in enumerate(split_iter, start=1):
        if args.fold_limit is not None and fold_number > args.fold_limit:
            print(f"Reached fold_limit={args.fold_limit}; stopping {dataset_key}.")
            break

        repeat_number = ((fold_number - 1) // args.folds) + 1
        split_number = ((fold_number - 1) % args.folds) + 1
        if (dataset_key, fold_number) in completed and not args.force:
            print(f"Skipping completed fold {dataset_key} {fold_number}/{total_folds}.")
            continue

        fold_seed = args.seed + fold_number - 1
        fold_dir = run_dir / "fold_runs" / dataset_key / f"fold_{fold_number:02d}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train_fold = train_frame.iloc[train_index].reset_index(drop=True)
        X_val_fold = train_frame.iloc[val_index].reset_index(drop=True)
        print(
            f"\n{dataset_key}: fold {fold_number}/{total_folds} "
            f"(repeat={repeat_number}, split={split_number}, seed={fold_seed})"
        )
        print(
            "fold_sizes="
            f"train={len(X_train_fold)} validation={len(X_val_fold)} "
            f"train_counts={X_train_fold['label'].value_counts().to_dict()} "
            f"validation_counts={X_val_fold['label'].value_counts().to_dict()}"
        )

        train_dataset = make_encoded_dataset(X_train_fold, tokenizer, args.max_length)
        val_dataset = make_encoded_dataset(X_val_fold, tokenizer, args.max_length)
        model = make_model(args.model_name, fold_seed)
        if representative_parameter_count is None:
            representative_parameter_count = int(sum(p.numel() for p in model.parameters()))

        trainer = Trainer(
            model=model,
            args=training_args_for_fold(
                fold_dir,
                args,
                fold_seed,
                train_size=len(X_train_fold),
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )

        started = time.perf_counter()
        trainer.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        labels, logits = predict_with_model(
            trainer.model,
            val_dataset,
            batch_size=args.eval_batch_size,
        )
        elapsed = time.perf_counter() - started

        metrics = compute_prediction_metrics(labels, logits)
        preds = np.argmax(logits, axis=1)
        cm = confusion_matrix(labels, preds, labels=[0, 1]).astype(int).tolist()

        row = {
            "dataset_key": dataset_key,
            "dataset": dataset_name,
            "fold_number": fold_number,
            "repeat_number": repeat_number,
            "split_number": split_number,
            "seed": fold_seed,
            "train_size": len(X_train_fold),
            "validation_size": len(X_val_fold),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_fake": metrics["f1_fake"],
            "f1_real": metrics["f1_real"],
            "roc_auc": metrics["roc_auc"],
            "elapsed_seconds": elapsed,
        }
        append_metric_row(metrics_path, row)
        completed.add((dataset_key, fold_number))

        print(
            f"{dataset_key}: fold {fold_number}/{total_folds} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} "
            f"f1_fake={metrics['f1_fake']:.4f} "
            f"f1_real={metrics['f1_real']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"elapsed_seconds={elapsed:.1f}"
        )
        print(
            "confusion_matrix="
            f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"
        )

        if dataset_key == "combined" and fold_number == 1:
            representative_cm = cm
            representative_model_dir = run_dir / "representative_joint_model"
            if representative_model_dir.exists():
                shutil.rmtree(representative_model_dir)
            trainer.save_model(str(representative_model_dir))
            tokenizer.save_pretrained(str(representative_model_dir))
            print(f"Saved representative joint model to {representative_model_dir}")

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return representative_cm, representative_model_dir, representative_parameter_count


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    log_path = configure_tee(run_dir)
    print_environment(args, run_dir)

    metrics_path = run_dir / "metrics.csv"
    if args.force and metrics_path.exists():
        metrics_path.unlink()
    completed = completed_fold_keys(metrics_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset_keys = selected_dataset_keys(args.datasets)

    representative_cm = None
    representative_model_dir = None
    parameter_count = None
    loaded_frames: dict[str, pd.DataFrame] = {}

    for dataset_key in dataset_keys:
        dataset_name, frame = load_dataset_frame(dataset_key)
        loaded_frames[dataset_key] = frame
        cm, model_dir, fold_parameter_count = evaluate_dataset(
            dataset_key,
            dataset_name,
            frame,
            tokenizer,
            run_dir,
            metrics_path,
            args,
            completed,
        )
        if cm is not None:
            representative_cm = cm
        if model_dir is not None:
            representative_model_dir = model_dir
        if parameter_count is None and fold_parameter_count is not None:
            parameter_count = fold_parameter_count

    if representative_cm is None or representative_model_dir is None:
        raise RuntimeError(
            "No representative joint fold was produced. Include the combined dataset "
            "and ensure fold 1 is not skipped without an existing saved model."
        )
    if parameter_count is None:
        parameter_count = int(
            sum(p.numel() for p in make_model(args.model_name, args.seed).parameters())
        )

    model_size_mb = model_dir_size_mb(representative_model_dir)
    _, joint_frame = load_dataset_frame("combined")
    inference_summary = benchmark_inference(
        representative_model_dir,
        tokenizer,
        joint_frame,
        args,
    )

    write_json(run_dir / "representative_joint_confusion_matrix.json", representative_cm)
    write_json(run_dir / "inference_benchmark.json", inference_summary)
    write_json(
        run_dir / "model_size.json",
        {
            "model": args.model_name,
            "parameter_count": parameter_count,
            "model_size_mb": model_size_mb,
            "representative_model_dir": str(representative_model_dir),
        },
    )
    report_path = create_report(
        run_dir,
        log_path,
        metrics_path,
        representative_cm,
        parameter_count,
        model_size_mb,
        inference_summary,
        args,
    )
    print(f"\nReport written to {report_path}")
    print(f"Raw log written to {log_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
