r"""Save the closest-to-mean DistilBERT joint representative model.

This is a small compatibility helper for the local inference comparison. The
original DistilBERT baseline output saved a representative joint model, but it
did not include metadata confirming the closest-to-mean fold. This script uses
the completed DistilBERT metrics.csv to rerun that exact fold and save it under:

    training\results\distilbert_tagalog_baseline\2026-06-03_full\representative_joint_model_closest_to_mean
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import distilbert_tagalog_baseline as baseline
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


RESULTS_DIR = (
    Path("results") / "distilbert_tagalog_baseline" / "2026-06-03_full"
)
MODEL_DIR = RESULTS_DIR / "representative_joint_model_closest_to_mean"
TRAINING_DIR = RESULTS_DIR / "representative_joint_model_closest_to_mean_training"
LOG_PATH = RESULTS_DIR / "distilbert_closest_representative.log"
METADATA_PATH = RESULTS_DIR / "representative_joint_model_closest_to_mean_metadata.json"


class Tee:
    def __init__(self, *streams: Any) -> None:
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


def configure_log() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    log_file = LOG_PATH.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)


def select_representative_fold() -> dict[str, object]:
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    joint = metrics[metrics["dataset_key"] == "combined"].copy()
    if joint.empty:
        raise RuntimeError("No combined DistilBERT metric rows found.")
    joint_mean = float(joint["accuracy"].mean())
    joint["distance_from_mean_accuracy"] = (joint["accuracy"] - joint_mean).abs()
    joint = joint.sort_values(["distance_from_mean_accuracy", "fold_number"])
    row = joint.iloc[0].to_dict()
    row["joint_mean_accuracy"] = joint_mean
    row["distance_from_mean_accuracy"] = float(row["distance_from_mean_accuracy"])
    return row


def combined_fold_frames(fold_number: int, args: argparse.Namespace):
    _, frame = baseline.load_dataset_frame("combined")
    train_frame, _ = train_test_split(
        frame,
        test_size=baseline.TEST_SIZE,
        random_state=args.seed,
        stratify=frame["label"],
    )
    train_frame = train_frame.reset_index(drop=True)
    cv = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.repeats,
        random_state=args.seed,
    )
    for current_fold, (train_index, val_index) in enumerate(
        cv.split(train_frame["article"], train_frame["label"]),
        start=1,
    ):
        if current_fold == fold_number:
            return (
                train_frame.iloc[train_index].reset_index(drop=True),
                train_frame.iloc[val_index].reset_index(drop=True),
            )
    raise ValueError(f"Fold {fold_number} is outside the configured CV splits.")


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    configure_log()
    args = argparse.Namespace(
        model_name=baseline.MODEL_NAME,
        seed=baseline.RANDOM_STATE,
        folds=baseline.N_SPLITS,
        repeats=baseline.N_REPEATS,
        learning_rate=2e-5,
        batch_size=16,
        eval_batch_size=16,
        epochs=3.0,
        max_length=512,
        weight_decay=0.01,
        warmup_ratio=0.10,
        no_fp16=False,
        gradient_checkpointing=False,
    )
    representative = select_representative_fold()
    fold_number = int(representative["fold_number"])
    fold_seed = int(representative["seed"])
    print("# DistilBERT closest-to-mean representative")
    print(f"started_at={datetime.now().isoformat(timespec='seconds')}")
    print(f"fold={fold_number} seed={fold_seed}")
    print(f"fold_accuracy={float(representative['accuracy']):.9f}")
    print(f"joint_mean_accuracy={float(representative['joint_mean_accuracy']):.9f}")

    tokenizer = baseline.AutoTokenizer.from_pretrained(args.model_name)
    X_train_fold, X_val_fold = combined_fold_frames(fold_number, args)
    train_dataset = baseline.make_encoded_dataset(X_train_fold, tokenizer, args.max_length)
    val_dataset = baseline.make_encoded_dataset(X_val_fold, tokenizer, args.max_length)
    model = baseline.make_model(args.model_name, fold_seed)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    if TRAINING_DIR.exists():
        shutil.rmtree(TRAINING_DIR)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    trainer = baseline.Trainer(
        model=model,
        args=baseline.training_args_for_fold(
            TRAINING_DIR,
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
    labels, logits = baseline.predict_with_model(
        trainer.model,
        val_dataset,
        batch_size=args.eval_batch_size,
    )
    elapsed = time.perf_counter() - started
    metrics = baseline.compute_prediction_metrics(labels, logits)
    cm = confusion_matrix(labels, np.argmax(logits, axis=1), labels=[0, 1]).astype(int)

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    metadata = {
        **representative,
        "model_dir": str(MODEL_DIR),
        "parameter_count": parameter_count,
        "rerun_elapsed_seconds": elapsed,
        "rerun_accuracy": metrics["accuracy"],
        "rerun_f1_macro": metrics["f1_macro"],
        "rerun_roc_auc": metrics["roc_auc"],
        "rerun_confusion_matrix": cm.tolist(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"rerun_accuracy={metrics['accuracy']:.9f}")
    print(f"rerun_f1_macro={metrics['f1_macro']:.9f}")
    print(f"rerun_roc_auc={metrics['roc_auc']:.9f}")
    print(f"confusion_matrix={cm.tolist()}")
    print(f"saved_model={MODEL_DIR}")
    print(f"metadata={METADATA_PATH}")


if __name__ == "__main__":
    main()
