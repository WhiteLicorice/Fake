"""Generate appendix ROC-AUC figures from saved run metadata.

The classical and transformer runs saved 30-fold scalar AUC values, but not
fold-level FPR/TPR arrays. By default this script uses representative-fold ROC
curves and reports the 30-fold mean AUC +/- SD in the legends. If point-level
classical curves are needed, run with ``--classical-mode full`` to recompute
all 30 classical CV folds from the original rerun pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent
TRAINING_DIR = SCRIPT_PATH.parents[2]
REPO_DIR = TRAINING_DIR.parent

os.chdir(TRAINING_DIR)
sys.path.insert(0, str(TRAINING_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

from rerun_common import (  # noqa: E402
    CLASSIFIER_ORDER,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR as CLASSICAL_RESULTS_DIR,
    TEST_SIZE,
    load_dataset,
    make_pipeline,
    positive_scores,
    train_partition,
    tuned_classifiers,
)


CLASSICAL_RAW_CSV = CLASSICAL_RESULTS_DIR / "roc_auc_30run_raw.csv"
CLASSICAL_BEST_PARAMS = CLASSICAL_RESULTS_DIR / "tuned_best_params.json"
CLASSICAL_CACHE = OUTPUT_DIR / "classical_roc_curves_30fold.npz"

DISTILBERT_RUN_DIR = (
    TRAINING_DIR / "results" / "distilbert_tagalog_baseline" / "2026-06-03_full"
)
ROBERTA_RUN_DIR = (
    TRAINING_DIR / "results" / "roberta_tagalog_baseline" / "2026-06-04_full"
)

MEAN_FPR = np.linspace(0.0, 1.0, 200)
PM = "\u00b1"
EM_DASH = "\N{EM DASH}"

MODEL_DISPLAY = {
    "LR": "LR",
    "MNB": "MNB",
    "RF": "RF",
    "SVC": "SVC",
    "DistilBERT": "DistilBERT",
    "RoBERTa": "RoBERTa",
}

MODEL_COLORS = {
    "LR": "#0173B2",
    "MNB": "#DE8F05",
    "RF": "#029E73",
    "SVC": "#D55E00",
    "DistilBERT": "#CC78BC",
    "RoBERTa": "#CA9161",
}

ML_MODELS = ["LR", "MNB", "RF", "SVC"]
TRANSFORMER_MODELS = ["DistilBERT", "RoBERTa"]


@dataclass
class CurveData:
    model: str
    mean_fpr: np.ndarray
    mean_tpr: np.ndarray
    sd_tpr: np.ndarray
    auc_mean: float
    auc_sd: float
    n_curves: int
    source: str
    representative_auc: float | None = None
    representative_fold: int | None = None
    representative_repeat: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical-mode",
        choices=["auto", "representative", "full"],
        default="auto",
        help=(
            "auto loads cached 30-fold classical curves when available, otherwise "
            "uses representative folds; full recomputes all 30 folds."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel jobs for --classical-mode full.",
    )
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=16,
        help="Inference batch size for representative transformer curves.",
    )
    return parser.parse_args()


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_classical_auc_stats() -> dict[str, dict[str, float | int]]:
    raw = pd.read_csv(CLASSICAL_RAW_CSV)
    required = {"classifier", "repeat", "fold", "auc"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in {CLASSICAL_RAW_CSV}: {sorted(missing)}")

    stats: dict[str, dict[str, float | int]] = {}
    for classifier, group in raw.groupby("classifier"):
        aucs = group["auc"].astype(float)
        mean_auc = float(aucs.mean())
        sd_auc = float(aucs.std(ddof=1))
        representative = group.iloc[(aucs - mean_auc).abs().to_numpy().argmin()]
        stats[str(classifier)] = {
            "auc_mean": mean_auc,
            "auc_sd": sd_auc,
            "repeat": int(representative["repeat"]),
            "fold": int(representative["fold"]),
            "representative_auc": float(representative["auc"]),
        }
    return stats


def interpolate_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    auc = float(roc_auc_score(y_true, y_score))
    return interp_tpr, auc


def summarize_fold_curves(
    model: str,
    fold_tprs: list[np.ndarray],
    fold_aucs: list[float],
    source: str,
    representative_fold: int | None = None,
    representative_repeat: int | None = None,
) -> CurveData:
    tpr_matrix = np.vstack(fold_tprs)
    auc_array = np.asarray(fold_aucs, dtype=float)
    return CurveData(
        model=model,
        mean_fpr=MEAN_FPR,
        mean_tpr=tpr_matrix.mean(axis=0),
        sd_tpr=tpr_matrix.std(axis=0, ddof=1) if len(fold_tprs) > 1 else np.zeros_like(MEAN_FPR),
        auc_mean=float(auc_array.mean()),
        auc_sd=float(auc_array.std(ddof=1)) if len(fold_aucs) > 1 else 0.0,
        n_curves=len(fold_tprs),
        source=source,
        representative_auc=float(auc_array[0]) if len(fold_aucs) == 1 else None,
        representative_fold=representative_fold,
        representative_repeat=representative_repeat,
    )


def load_tuned_classifiers() -> dict[str, object]:
    best_params = load_json(CLASSICAL_BEST_PARAMS) if CLASSICAL_BEST_PARAMS.exists() else None
    classifiers = tuned_classifiers(best_params, probability_for_svc=True)
    for classifier in classifiers.values():
        params = classifier.get_params()
        if "n_jobs" in params:
            classifier.set_params(n_jobs=1)
    return classifiers


def make_classical_splits():
    X, y, _ = load_dataset("Joint", include_lex_morph=True)
    X_train, _, y_train, _ = train_partition(X, y)
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    splits = list(enumerate(cv.split(X_train), start=1))
    return X_train, y_train, splits


def fit_classical_fold(
    classifier_id: str,
    classifier: object,
    run_index: int,
    train_index: np.ndarray,
    val_index: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, float]:
    repeat = ((run_index - 1) // N_SPLITS) + 1
    fold = ((run_index - 1) % N_SPLITS) + 1
    started = time.perf_counter()
    pipeline = make_pipeline(clone(classifier), include_lex_morph=True)
    pipeline.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    y_score = positive_scores(pipeline, X_train.iloc[val_index])
    y_true = y_train.iloc[val_index].to_numpy()
    interp_tpr, auc = interpolate_curve(y_true, y_score)
    elapsed = time.perf_counter() - started
    print(
        f"{classifier_id}: repeat={repeat} fold={fold} "
        f"AUC={auc:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return interp_tpr, auc


def compute_full_classical_curves(n_jobs: int) -> dict[str, CurveData]:
    classifiers = load_tuned_classifiers()
    X_train, y_train, splits = make_classical_splits()
    curves: dict[str, CurveData] = {}

    for classifier_id in CLASSIFIER_ORDER:
        print(f"\nRecomputing 30-fold ROC curves for {classifier_id}", flush=True)
        rows = Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs")(
            delayed(fit_classical_fold)(
                classifier_id,
                classifiers[classifier_id],
                run_index,
                train_index,
                val_index,
                X_train,
                y_train,
            )
            for run_index, (train_index, val_index) in splits
        )
        fold_tprs = [row[0] for row in rows]
        fold_aucs = [row[1] for row in rows]
        curves[classifier_id] = summarize_fold_curves(
            classifier_id,
            fold_tprs,
            fold_aucs,
            source="30-fold recomputed ROC curves",
        )

    save_classical_cache(curves)
    return curves


def compute_representative_classical_curves() -> dict[str, CurveData]:
    stats = load_classical_auc_stats()
    classifiers = load_tuned_classifiers()
    X_train, y_train, splits = make_classical_splits()
    curves: dict[str, CurveData] = {}

    print(
        "\nPoint-level classical ROC data were not saved. "
        "Using folds whose scalar AUC is closest to each model mean.",
        flush=True,
    )
    for classifier_id in CLASSIFIER_ORDER:
        model_stats = stats[classifier_id]
        run_index = (int(model_stats["repeat"]) - 1) * N_SPLITS + int(model_stats["fold"])
        train_index, val_index = splits[run_index - 1][1]
        fold_tpr, fold_auc = fit_classical_fold(
            classifier_id,
            classifiers[classifier_id],
            run_index,
            train_index,
            val_index,
            X_train,
            y_train,
        )
        curve = summarize_fold_curves(
            classifier_id,
            [fold_tpr],
            [fold_auc],
            source="representative fold; 30-fold scalar AUC stats from saved CSV",
            representative_fold=int(model_stats["fold"]),
            representative_repeat=int(model_stats["repeat"]),
        )
        curve.auc_mean = float(model_stats["auc_mean"])
        curve.auc_sd = float(model_stats["auc_sd"])
        curve.representative_auc = float(model_stats["representative_auc"])
        curves[classifier_id] = curve
    return curves


def save_classical_cache(curves: dict[str, CurveData]) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, float | int | str | None]] = {}
    for model, curve in curves.items():
        arrays[f"{model}_mean_tpr"] = curve.mean_tpr
        arrays[f"{model}_sd_tpr"] = curve.sd_tpr
        metadata[model] = curve_metadata(curve)
    arrays["mean_fpr"] = MEAN_FPR
    arrays["metadata_json"] = np.asarray(json.dumps(metadata), dtype=object)
    np.savez_compressed(CLASSICAL_CACHE, **arrays)
    print(f"Saved classical ROC cache: {CLASSICAL_CACHE}", flush=True)


def load_classical_cache() -> dict[str, CurveData]:
    data = np.load(CLASSICAL_CACHE, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"].item()))
    curves: dict[str, CurveData] = {}
    for model in CLASSIFIER_ORDER:
        meta = metadata[model]
        curves[model] = CurveData(
            model=model,
            mean_fpr=np.asarray(data["mean_fpr"], dtype=float),
            mean_tpr=np.asarray(data[f"{model}_mean_tpr"], dtype=float),
            sd_tpr=np.asarray(data[f"{model}_sd_tpr"], dtype=float),
            auc_mean=float(meta["auc_mean"]),
            auc_sd=float(meta["auc_sd"]),
            n_curves=int(meta["n_curves"]),
            source=str(meta["source"]),
            representative_auc=meta["representative_auc"],
            representative_fold=meta["representative_fold"],
            representative_repeat=meta["representative_repeat"],
        )
    return curves


def get_classical_curves(mode: str, n_jobs: int) -> dict[str, CurveData]:
    if mode == "full":
        return compute_full_classical_curves(n_jobs=n_jobs)
    if mode == "auto" and CLASSICAL_CACHE.exists():
        print(f"Loading cached classical 30-fold ROC curves: {CLASSICAL_CACHE}", flush=True)
        return load_classical_cache()
    return compute_representative_classical_curves()


def read_article_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["label", "article"])
    if frame.empty:
        raise ValueError(f"Empty dataset: {path}")
    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    frame["article"] = frame["article"].astype(str)
    return frame


def load_combined_article_frame() -> pd.DataFrame:
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


def transformer_validation_frame(fold_number: int) -> pd.DataFrame:
    frame = load_combined_article_frame()
    train_frame, _ = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )
    train_frame = train_frame.reset_index(drop=True)
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    for current_fold, (_, val_index) in enumerate(
        cv.split(train_frame["article"], train_frame["label"]),
        start=1,
    ):
        if current_fold == fold_number:
            return train_frame.iloc[val_index].reset_index(drop=True)
    raise ValueError(f"Fold {fold_number} is outside the 30-fold transformer CV")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_transformer_scores(
    model_dir: Path,
    frame: pd.DataFrame,
    batch_size: int,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    texts = frame["article"].astype(str).tolist()
    y_true = frame["label"].astype(int).to_numpy()
    scores: list[np.ndarray] = []

    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            scores.append(softmax(logits)[:, 1])

    elapsed = time.perf_counter() - started
    y_score = np.concatenate(scores)
    print(
        f"Transformer inference complete for {model_dir.name}: "
        f"{len(y_true)} rows in {elapsed:.1f}s on {device}",
        flush=True,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return y_true, y_score


def load_transformer_auc_stats(metrics_path: Path) -> tuple[float, float]:
    metrics = pd.read_csv(metrics_path)
    joint = metrics[metrics["dataset_key"].astype(str).str.lower() == "combined"]
    if len(joint) != N_SPLITS * N_REPEATS:
        raise ValueError(f"Expected 30 combined rows in {metrics_path}, got {len(joint)}")
    aucs = joint["roc_auc"].astype(float)
    return float(aucs.mean()), float(aucs.std(ddof=1))


def resolve_model_dir(run_dir: Path, metadata: dict, keys: list[str], fallback: str) -> Path:
    for key in keys:
        value = metadata.get(key)
        if value:
            path = Path(str(value))
            if not path.is_absolute():
                path = TRAINING_DIR / path
            if path.exists():
                return path
    path = run_dir / fallback
    if not path.exists():
        raise FileNotFoundError(f"Missing representative model directory: {path}")
    return path


def compute_transformer_curve(
    model: str,
    run_dir: Path,
    metadata_file: str,
    model_dir_keys: list[str],
    fallback_model_dir: str,
    batch_size: int,
) -> CurveData:
    metadata = load_json(run_dir / metadata_file)
    fold_number = int(metadata["fold_number"])
    repeat_number = int(metadata.get("repeat_number", ((fold_number - 1) // N_SPLITS) + 1))
    model_dir = resolve_model_dir(run_dir, metadata, model_dir_keys, fallback_model_dir)
    auc_mean, auc_sd = load_transformer_auc_stats(run_dir / "metrics.csv")

    print(
        f"\nUsing representative {model} fold {fold_number} "
        f"from {model_dir}",
        flush=True,
    )
    val_frame = transformer_validation_frame(fold_number)
    y_true, y_score = predict_transformer_scores(
        model_dir,
        val_frame,
        batch_size=batch_size,
    )
    fold_tpr, fold_auc = interpolate_curve(y_true, y_score)
    curve = summarize_fold_curves(
        model,
        [fold_tpr],
        [fold_auc],
        source="representative fold; 30-fold scalar AUC stats from metrics.csv",
        representative_fold=fold_number,
        representative_repeat=repeat_number,
    )
    curve.auc_mean = auc_mean
    curve.auc_sd = auc_sd
    curve.representative_auc = fold_auc
    return curve


def get_transformer_curves(batch_size: int) -> dict[str, CurveData]:
    return {
        "DistilBERT": compute_transformer_curve(
            model="DistilBERT",
            run_dir=DISTILBERT_RUN_DIR,
            metadata_file="representative_joint_model_closest_to_mean_metadata.json",
            model_dir_keys=["model_dir"],
            fallback_model_dir="representative_joint_model_closest_to_mean",
            batch_size=batch_size,
        ),
        "RoBERTa": compute_transformer_curve(
            model="RoBERTa",
            run_dir=ROBERTA_RUN_DIR,
            metadata_file="representative_joint_model_metadata.json",
            model_dir_keys=["representative_model_dir", "model_dir"],
            fallback_model_dir="representative_joint_model",
            batch_size=batch_size,
        ),
    }


def curve_metadata(curve: CurveData) -> dict[str, float | int | str | None]:
    return {
        "model": curve.model,
        "auc_mean": curve.auc_mean,
        "auc_sd": curve.auc_sd,
        "n_curves": curve.n_curves,
        "source": curve.source,
        "representative_auc": curve.representative_auc,
        "representative_fold": curve.representative_fold,
        "representative_repeat": curve.representative_repeat,
    }


def save_run_metadata(curves: dict[str, CurveData]) -> None:
    payload = {
        "mean_fpr_points": len(MEAN_FPR),
        "classical_raw_csv": str(CLASSICAL_RAW_CSV),
        "distilbert_run_dir": str(DISTILBERT_RUN_DIR),
        "roberta_run_dir": str(ROBERTA_RUN_DIR),
        "curves": {model: curve_metadata(curve) for model, curve in curves.items()},
    }
    path = OUTPUT_DIR / "roc_auc_figure_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metadata: {path}", flush=True)


def plot_roc_curves(
    curves: dict[str, CurveData],
    model_order: list[str],
    title: str,
    output_stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    for model in model_order:
        curve = curves[model]
        color = MODEL_COLORS[model]
        linestyle = "--" if model in TRANSFORMER_MODELS else "-"
        label = (
            f"{MODEL_DISPLAY[model]} "
            f"(AUC = {curve.auc_mean:.3f} {PM} {curve.auc_sd:.3f})"
        )
        if curve.n_curves > 1 and np.any(curve.sd_tpr > 0):
            lower = np.maximum(curve.mean_tpr - curve.sd_tpr, 0.0)
            upper = np.minimum(curve.mean_tpr + curve.sd_tpr, 1.0)
            ax.fill_between(curve.mean_fpr, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            curve.mean_fpr,
            curve.mean_tpr,
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
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}", flush=True)
    print(f"Saved {pdf_path}", flush=True)


def print_auc_summary(curves: dict[str, CurveData]) -> None:
    print("\nAUC summary used in legends", flush=True)
    for model in sorted(curves, key=lambda key: curves[key].auc_mean, reverse=True):
        curve = curves[model]
        rep = ""
        if curve.representative_fold is not None:
            rep = (
                f" | representative repeat={curve.representative_repeat} "
                f"fold={curve.representative_fold} "
                f"AUC={curve.representative_auc:.6f}"
            )
        print(
            f"{MODEL_DISPLAY[model]:<10} AUC = "
            f"{curve.auc_mean:.6f} {PM} {curve.auc_sd:.6f} "
            f"| curves={curve.n_curves} | {curve.source}{rep}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classical_curves = get_classical_curves(args.classical_mode, args.n_jobs)
    transformer_curves = get_transformer_curves(args.transformer_batch_size)
    curves = {**classical_curves, **transformer_curves}

    plot_roc_curves(
        curves,
        model_order=["LR", "MNB", "RF", "SVC"],
        title=f"ROC Curves {EM_DASH} Classical ML Classifiers (Joint Corpus, 30-Run CV)",
        output_stem="roc_auc_ml_models",
    )

    all_model_order = sorted(curves, key=lambda key: curves[key].auc_mean, reverse=True)
    plot_roc_curves(
        curves,
        model_order=all_model_order,
        title=f"ROC Curves {EM_DASH} All Models (Joint Corpus, 30-Run CV)",
        output_stem="roc_auc_all_models",
    )

    save_run_metadata(curves)
    print_auc_summary(curves)

    if any(curves[model].n_curves == 1 for model in curves):
        print(
            "\nNote: one or more plotted ROC curves are representative-fold curves "
            "because saved fold-level FPR/TPR or probability arrays were unavailable. "
            "The legend AUC values remain the saved 30-fold mean +/- SD.",
            flush=True,
        )


if __name__ == "__main__":
    main()
