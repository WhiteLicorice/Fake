r"""Compare local inference speed for LR, DistilBERT, and RoBERTa.

Run from the repository root with:

    training\venv\Scripts\python.exe training\local_inference_comparison.py --output-dir training\results\roberta_tagalog_baseline\<run>
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import io
import json
import os
import pickle
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_DIR / "training"
SERVER_DIR = REPO_DIR / "server"
DISTILBERT_RESULTS_DIR = (
    TRAINING_DIR / "results" / "distilbert_tagalog_baseline" / "2026-06-03_full"
)
LR_MODEL_PATH = SERVER_DIR / "root" / "models" / "LogisticRegression.pkl"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "results" / "local_inference_comparison"
DEFAULT_ROBERTA_RESULTS_DIR = TRAINING_DIR / "results" / "roberta_tagalog_baseline"
RANDOM_STATE = 42
MAX_LENGTH = 512
WARMUP_ITERATIONS = 5
TIMED_ITERATIONS = 100


@dataclass(frozen=True)
class ModelSpec:
    name: str
    device: str
    params: int | str
    size_mb: float
    predict_one: Callable[[str], int]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distilbert-results-dir", type=Path, default=DISTILBERT_RESULTS_DIR)
    parser.add_argument("--roberta-results-dir", type=Path, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--timed-iterations", type=int, default=TIMED_ITERATIONS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    return parser.parse_args()


def configure_output(output_dir: Path) -> tuple[Path, Path, io.StringIO]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    report_path = output_dir / "local_inference_comparison.md"
    log_path = output_dir / "local_inference_comparison.log"
    capture = io.StringIO()
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file, capture)
    sys.stderr = Tee(sys.__stderr__, log_file, capture)
    shutil.copy2(Path(__file__).resolve(), output_dir / Path(__file__).name)
    return report_path, log_path, capture


def read_article_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["label", "article"])
    frame["label"] = frame["label"].astype(int)
    frame["article"] = frame["article"].astype(str)
    return frame


def load_joint_frame() -> pd.DataFrame:
    cruz = read_article_csv(
        TRAINING_DIR / "root" / "datasets" / "Cruz" / "FakeNewsFilipino_Cruz2020.csv"
    )
    lupac = read_article_csv(
        TRAINING_DIR
        / "root"
        / "datasets"
        / "Lupac"
        / "FakeNewsPhilippines2024_Lupac.csv"
    )
    return pd.concat([cruz, lupac], ignore_index=True)


def benchmark_articles() -> list[str]:
    joint_frame = load_joint_frame()
    fake_sample = joint_frame[joint_frame["label"] == 0].sample(
        n=15, random_state=RANDOM_STATE
    )
    real_sample = joint_frame[joint_frame["label"] == 1].sample(
        n=15, random_state=RANDOM_STATE
    )
    sample = pd.concat([fake_sample, real_sample], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )
    return sample["article"].tolist()


def run_command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gpu_name() -> str:
    queried = run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if queried:
        return queried.splitlines()[0].strip()
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CUDA unavailable"


def cpu_name() -> str:
    queried = run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ]
    )
    return queried or platform.processor() or platform.machine()


def ram_gb() -> float | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    if os.name != "nt":
        return None
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return status.ullTotalPhys / (1024**3)
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024**2)
    total = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total / (1024**2)


def lr_parameter_count(pipeline: Any) -> int:
    classifier = pipeline.named_steps["classifier"]
    count = 0
    if hasattr(classifier, "coef_"):
        count += int(np.asarray(classifier.coef_).size)
    if hasattr(classifier, "intercept_"):
        count += int(np.asarray(classifier.intercept_).size)
    return count


def load_lr_pipeline() -> tuple[Any, int, float]:
    sys.path.insert(0, str(SERVER_DIR))
    with LR_MODEL_PATH.open("rb") as file:
        pipeline = pickle.load(file)
    return pipeline, lr_parameter_count(pipeline), model_size_mb(LR_MODEL_PATH)


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def transformer_parameter_count(model_dir: Path) -> int:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    count = int(sum(parameter.numel() for parameter in model.parameters()))
    del model
    return count


def transformer_size_and_params(results_dir: Path, model_dir: Path) -> tuple[int, float]:
    size_path = results_dir / "model_size.json"
    if size_path.exists():
        data = load_json(size_path)
        return int(data["parameter_count"]), float(data["model_size_mb"])
    return transformer_parameter_count(model_dir), model_size_mb(model_dir)


def latest_roberta_results_dir() -> Path:
    if not DEFAULT_ROBERTA_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Missing RoBERTa results root: {DEFAULT_ROBERTA_RESULTS_DIR}")
    candidates = [
        path
        for path in DEFAULT_ROBERTA_RESULTS_DIR.iterdir()
        if path.is_dir() and (path / "representative_joint_model").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No RoBERTa representative_joint_model directory found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_representative_model(results_dir: Path) -> tuple[Path, str]:
    closest = results_dir / "representative_joint_model_closest_to_mean"
    if closest.exists():
        return closest, "closest-to-mean representative model"
    standard = results_dir / "representative_joint_model"
    if not standard.exists():
        raise FileNotFoundError(f"Missing representative model under {results_dir}")
    metadata_path = results_dir / "representative_joint_model_metadata.json"
    if metadata_path.exists():
        return standard, "closest-to-mean representative model"
    return standard, "available saved representative model; metadata did not confirm closest-to-mean"


def make_lr_spec(pipeline: Any, params: int, size_mb: float) -> ModelSpec:
    def predict_one(text: str) -> int:
        with working_directory(SERVER_DIR):
            return int(pipeline.predict([text])[0])

    return ModelSpec(
        name="LR pipeline",
        device="CPU",
        params=params,
        size_mb=size_mb,
        predict_one=predict_one,
    )


def make_transformer_spec(
    name: str,
    results_dir: Path,
    model_dir: Path,
    device_name: str,
    max_length: int,
) -> ModelSpec:
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    params, size_mb = transformer_size_and_params(results_dir, model_dir)

    def predict_one(text: str) -> int:
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        return int(torch.argmax(output.logits, dim=1).item())

    return ModelSpec(
        name=name,
        device=device.type.upper(),
        params=params,
        size_mb=size_mb,
        predict_one=predict_one,
    )


def benchmark_model(
    spec: ModelSpec,
    articles: list[str],
    warmup_iterations: int,
    timed_iterations: int,
) -> dict[str, Any]:
    print(f"\nBenchmarking {spec.name} on {spec.device}")
    for iteration in range(1, warmup_iterations + 1):
        for text in articles:
            _ = spec.predict_one(text)
        if spec.device == "GPU" or spec.device == "CUDA":
            torch.cuda.synchronize()
        print(f"warmup_iteration={iteration}/{warmup_iterations} complete")

    timings: list[float] = []
    predictions: list[int] = []
    for iteration in range(1, timed_iterations + 1):
        for article_index, text in enumerate(articles, start=1):
            if spec.device == "GPU" or spec.device == "CUDA":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = spec.predict_one(text)
            if spec.device == "GPU" or spec.device == "CUDA":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings.append(elapsed_ms)
            predictions.append(prediction)
            print(
                "timed_prediction | "
                f"model={spec.name} | device={spec.device} | "
                f"iteration={iteration:03d} | article={article_index:02d} | "
                f"prediction={prediction} | elapsed_ms={elapsed_ms:.3f}"
            )

    return {
        "model": spec.name,
        "device": "GPU" if spec.device == "CUDA" else spec.device,
        "params": spec.params,
        "size_mb": spec.size_mb,
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "sd_ms": statistics.stdev(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "timings_ms": timings,
        "predictions": predictions,
    }


def fmt_params(value: int | str) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def fmt_float(value: float) -> str:
    return f"{value:.3f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Device",
        "Params",
        "Size (MB)",
        "Mean (ms)",
        "Median (ms)",
        "SD (ms)",
        "Min (ms)",
        "Max (ms)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["device"]),
                    fmt_params(row["params"]),
                    fmt_float(float(row["size_mb"])),
                    fmt_float(float(row["mean_ms"])),
                    fmt_float(float(row["median_ms"])),
                    fmt_float(float(row["sd_ms"])),
                    fmt_float(float(row["min_ms"])),
                    fmt_float(float(row["max_ms"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    report_path: Path,
    rows: list[dict[str, Any]],
    notes: list[str],
    raw_console_output: str,
    args: argparse.Namespace,
) -> None:
    ram = ram_gb()
    hardware = f"{gpu_name()}, {cpu_name()}"
    if ram is not None:
        hardware += f", {ram:.1f} GB RAM"
    lines = [
        "# Local Inference Speed Comparison",
        "",
        f"Hardware: {hardware}",
        f"Date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        (
            "Articles: 30, Iterations: "
            f"{args.timed_iterations} ({args.warmup_iterations} warmup excluded)"
        ),
        "",
        "## Results",
        "",
        markdown_table(rows),
        "",
        "## Notes",
        "",
        "- LR pipeline inference includes feature extraction (TF-IDF, BOW, linguistic features).",
        "- Transformer inference includes tokenization.",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## Raw Console Output",
            "",
            "```text",
            raw_console_output.rstrip(),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.roberta_results_dir is None:
        args.roberta_results_dir = latest_roberta_results_dir()
    report_path, log_path, capture = configure_output(args.output_dir)

    print("# Local Inference Speed Comparison")
    print(f"started_at={datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"report_path={report_path}")
    print(f"log_path={log_path}")
    print(f"distilbert_results_dir={args.distilbert_results_dir}")
    print(f"roberta_results_dir={args.roberta_results_dir}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")

    articles = benchmark_articles()
    print(f"benchmark_articles={len(articles)}")

    notes: list[str] = []
    distilbert_model_dir, distilbert_note = resolve_representative_model(
        args.distilbert_results_dir
    )
    roberta_model_dir, roberta_note = resolve_representative_model(
        args.roberta_results_dir
    )
    notes.append(f"DistilBERT model source: {distilbert_note}.")
    notes.append(f"RoBERTa model source: {roberta_note}.")

    rows: list[dict[str, Any]] = []
    lr_pipeline, lr_params, lr_size_mb = load_lr_pipeline()
    lr_spec = make_lr_spec(lr_pipeline, lr_params, lr_size_mb)
    rows.append(
        benchmark_model(
            lr_spec,
            articles,
            warmup_iterations=args.warmup_iterations,
            timed_iterations=args.timed_iterations,
        )
    )
    del lr_spec
    del lr_pipeline
    gc.collect()

    transformer_jobs: list[tuple[str, Path, Path, str]] = []
    if torch.cuda.is_available():
        transformer_jobs.extend(
            [
                ("DistilBERT", args.distilbert_results_dir, distilbert_model_dir, "cuda"),
                ("RoBERTa", args.roberta_results_dir, roberta_model_dir, "cuda"),
            ]
        )
    else:
        notes.append("GPU transformer benchmark skipped because CUDA was unavailable.")
    transformer_jobs.extend(
        [
            ("DistilBERT", args.distilbert_results_dir, distilbert_model_dir, "cpu"),
            ("RoBERTa", args.roberta_results_dir, roberta_model_dir, "cpu"),
        ]
    )

    for model_name, results_dir, model_dir, device_name in transformer_jobs:
        spec = make_transformer_spec(
            model_name,
            results_dir,
            model_dir,
            device_name,
            args.max_length,
        )
        rows.append(
            benchmark_model(
                spec,
                articles,
                warmup_iterations=args.warmup_iterations,
                timed_iterations=args.timed_iterations,
            )
        )
        del spec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(args.output_dir / "local_inference_comparison.json", rows)
    raw_console_output = capture.getvalue()
    write_report(report_path, rows, notes, raw_console_output, args)
    print(f"\nReport written to {report_path}")
    print(f"Raw log written to {log_path}")


if __name__ == "__main__":
    main()
