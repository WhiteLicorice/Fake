"""Render API benchmark and deployed Logistic Regression coefficient report.

Run from the repository root with the training virtual environment:
    training\\venv\\Scripts\\python.exe server\\benchmark_api_runtime.py

The script sends one warmup request to the deployed Render API, excludes that
request from timing summaries, then sends 30 measured requests: 10 short, 10
medium, and 10 long articles. It also loads the deployed model pickle from
server/root/models/LogisticRegression.pkl and reports coefficients from that
model, not a locally retrained pipeline.
"""

from __future__ import annotations

import csv
import io
import pickle
import statistics
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


REPO_DIR = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_DIR / "server"
RESULTS_DIR = REPO_DIR / "training" / "results" / "stopwords_fix_rerun"
REPORT_PATH = RESULTS_DIR / "render_benchmark_deployment_coefficients.md"
LOG_PATH = RESULTS_DIR / "render_benchmark_deployment_coefficients.log"
MODEL_PATH = SERVER_DIR / "root" / "models" / "LogisticRegression.pkl"

API_ENDPOINT = "https://fph-ml.onrender.com/check-news"
TIMEOUT_SECONDS = 180
REQUESTS_PER_LENGTH = 10

SERVER_TIMING_KEYS_MS = {
    "inference_ms",
    "inference_time_ms",
    "prediction_ms",
    "prediction_time_ms",
    "server_ms",
    "server_time_ms",
    "elapsed_ms",
}

CUSTOM_FEATURE_NAMES = {
    "read": ["readability_score"],
    "oov": ["count_oov_words"],
    "sw": ["count_stopwords"],
    "trad": [
        "word_count",
        "sentence_count",
        "polysyll_count",
        "ave_word_length",
        "ave_phrase_count",
        "ave_syllable_count_of_word",
        "word_count_per_sentence",
    ],
    "syll": [
        "consonant_cluster",
        "v_density",
        "cv_density",
        "vc_density",
        "cvc_density",
        "vcc_density",
        "cvcc_density",
        "ccvcc_density",
        "ccvccc_density",
    ],
}


@dataclass(frozen=True)
class ArticleSample:
    length_label: str
    sample_id: str
    text: str
    source: str

    @property
    def word_count(self) -> int:
        return word_count(self.text)


@dataclass(frozen=True)
class RequestResult:
    length_label: str
    sample_id: str
    word_count: int
    request_number: int
    round_trip_ms: float
    server_side_ms: float | None
    response_json: dict[str, Any]

    @property
    def measured_ms(self) -> float:
        return self.server_side_ms if self.server_side_ms is not None else self.round_trip_ms


PREVIOUS_BENCHMARK_SAMPLES = {
    "Short (~50 words)": (
        "previous_short",
        (
            "Patay ang isang 5-anyos na lalaki sa Lapu-Lapu City, Cebu matapos "
            "umano siyang pukpukin sa ulo at ihagis sa dagat ng kaniyang 14-anyos "
            "na kapatid. Sa ulat ng ABS-CBN News, nangyari ang insidente sa Sitio "
            "Lawis, Barangay Suba-Basbas nitong Biyernes ng umaga, Mayo 10."
        ),
    ),
    "Medium (~100 words)": (
        "previous_medium",
        (
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
    ),
    "Long (~200 words)": (
        "previous_long",
        (
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
    ),
}

LENGTH_TARGETS = {
    "Short (~50 words)": 50,
    "Medium (~100 words)": 100,
    "Long (~200 words)": 200,
}


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


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def word_count(text: str) -> int:
    return len(text.split())


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def load_dataset_articles() -> list[tuple[str, str]]:
    dataset_paths = [
        (
            "Cruz/FakeNewsFilipino_Cruz2020",
            REPO_DIR
            / "training"
            / "root"
            / "datasets"
            / "Cruz"
            / "FakeNewsFilipino_Cruz2020.csv",
        ),
        (
            "Lupac/FakeNewsPhilippines2024_Lupac",
            REPO_DIR
            / "training"
            / "root"
            / "datasets"
            / "Lupac"
            / "FakeNewsPhilippines2024_Lupac.csv",
        ),
    ]
    articles: list[tuple[str, str]] = []
    for source_name, path in dataset_paths:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                text = normalize_text(row.get("article", ""))
                if text:
                    articles.append((f"{source_name} row {index}", text))
    return articles


def select_samples() -> list[ArticleSample]:
    dataset_articles = load_dataset_articles()
    used_texts: set[str] = set()
    samples: list[ArticleSample] = []

    for length_label, target_words in LENGTH_TARGETS.items():
        previous_id, previous_text = PREVIOUS_BENCHMARK_SAMPLES[length_label]
        previous_text = normalize_text(previous_text)
        length_samples = [
            ArticleSample(
                length_label=length_label,
                sample_id=previous_id,
                text=previous_text,
                source="training.rerun_common.BENCHMARK_ARTICLES",
            )
        ]
        used_texts.add(previous_text)

        ranked = sorted(
            (
                (abs(word_count(text) - target_words), word_count(text), source, text)
                for source, text in dataset_articles
                if text not in used_texts
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        for _, _, source, text in ranked:
            if len(length_samples) >= REQUESTS_PER_LENGTH:
                break
            length_samples.append(
                ArticleSample(
                    length_label=length_label,
                    sample_id=f"{length_label.split()[0].lower()}_{len(length_samples):02d}",
                    text=text,
                    source=source,
                )
            )
            used_texts.add(text)

        samples.extend(length_samples)

    return samples


def find_server_side_ms(payload: Any) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in SERVER_TIMING_KEYS_MS and isinstance(value, (int, float)):
                return float(value)
        for value in payload.values():
            nested = find_server_side_ms(value)
            if nested is not None:
                return nested
    return None


def post_article(session: requests.Session, text: str) -> tuple[float, float | None, dict[str, Any]]:
    started = time.perf_counter()
    response = session.post(
        API_ENDPOINT,
        json={"news_body": text},
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )
    round_trip_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    response_json = response.json()
    return round_trip_ms, find_server_side_ms(response_json), response_json


def warmup(session: requests.Session, sample: ArticleSample) -> tuple[float, dict[str, Any]]:
    try:
        round_trip_ms, _, response_json = post_article(session, sample.text)
    except requests.RequestException as exc:
        print(f"Warmup attempt failed: {exc}")
        print("Sleeping 30 seconds before retrying warmup.")
        time.sleep(30)
        round_trip_ms, _, response_json = post_article(session, sample.text)
    return round_trip_ms, response_json


def benchmark_render(samples: list[ArticleSample]) -> tuple[float, dict[str, Any], list[RequestResult]]:
    session = requests.Session()
    warmup_sample = samples[0]

    print("# Render API benchmark")
    print(f"Endpoint: {API_ENDPOINT}")
    print('Request JSON format: {"news_body": "<article text>"}')
    print(f"Warmup sample: {warmup_sample.sample_id} ({warmup_sample.word_count} words)")
    warmup_ms, warmup_response = warmup(session, warmup_sample)
    print(f"Warmup response time: {warmup_ms:.3f} ms (excluded)")
    print(f"Warmup response JSON: {warmup_response}")
    print()

    results: list[RequestResult] = []
    for request_number, sample in enumerate(samples, start=1):
        round_trip_ms, server_side_ms, response_json = post_article(session, sample.text)
        result = RequestResult(
            length_label=sample.length_label,
            sample_id=sample.sample_id,
            word_count=sample.word_count,
            request_number=request_number,
            round_trip_ms=round_trip_ms,
            server_side_ms=server_side_ms,
            response_json=response_json,
        )
        results.append(result)
        primary_timing = (
            f"server_ms={server_side_ms:.3f} | round_trip_ms={round_trip_ms:.3f}"
            if server_side_ms is not None
            else f"round_trip_ms={round_trip_ms:.3f}"
        )
        print(
            "Benchmark request | "
            f"request={request_number:02d} | "
            f"length={sample.length_label} | "
            f"sample={sample.sample_id} | "
            f"words={sample.word_count} | "
            f"{primary_timing} | "
            f"response={response_json}"
        )
    print()
    return warmup_ms, warmup_response, results


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "sd_ms": statistics.stdev(values),
    }


def benchmark_summary(results: list[RequestResult]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for length_label in LENGTH_TARGETS:
        values = [result.measured_ms for result in results if result.length_label == length_label]
        rows[length_label] = summarize(values)
    rows["Overall"] = summarize([result.measured_ms for result in results])
    return rows


def feature_names_for_transformer(name: str, transformer: Any) -> list[str]:
    if name in CUSTOM_FEATURE_NAMES:
        return [f"{name}__{feature}" for feature in CUSTOM_FEATURE_NAMES[name]]
    if hasattr(transformer, "get_feature_names_out"):
        return [f"{name}__{feature}" for feature in transformer.get_feature_names_out()]
    if hasattr(transformer, "transformer_list"):
        names: list[str] = []
        for child_name, child_transformer in transformer.transformer_list:
            child_names = feature_names_for_transformer(child_name, child_transformer)
            names.extend([f"{name}__{child_feature}" for child_feature in child_names])
        return names
    raise ValueError(f"Cannot recover feature names for transformer {name!r}")


def is_vectorizer_feature(feature_name: str) -> bool:
    return (
        feature_name.startswith("tfidf__")
        or feature_name.startswith("bow__")
        or feature_name.startswith("vectorizers__")
    )


def load_deployment_coefficients() -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    sys.path.insert(0, str(SERVER_DIR))
    with MODEL_PATH.open("rb") as file:
        pipeline = pickle.load(file)

    feature_names: list[str] = []
    for name, transformer in pipeline.named_steps["features"].transformer_list:
        feature_names.extend(feature_names_for_transformer(name, transformer))

    coefficients = [float(value) for value in pipeline.named_steps["classifier"].coef_[0]]
    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Feature-name count {len(feature_names)} does not match coefficient count {len(coefficients)}"
        )

    rows = [
        {"feature": feature_name, "coefficient": coefficient}
        for feature_name, coefficient in zip(feature_names, coefficients)
    ]
    linguistic = sorted(
        [row for row in rows if not is_vectorizer_feature(str(row["feature"]))],
        key=lambda row: float(row["coefficient"]),
    )
    vectorizer = sorted(
        [row for row in rows if is_vectorizer_feature(str(row["feature"]))],
        key=lambda row: float(row["coefficient"]),
    )
    top_vectorizer = vectorizer[:3] + sorted(vectorizer[-3:], key=lambda row: float(row["coefficient"]), reverse=True)

    print("# Deployment model coefficients")
    print(f"Model path: {MODEL_PATH}")
    print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    print(
        "FeatureUnion transformers: "
        f"{[name for name, _ in pipeline.named_steps['features'].transformer_list]}"
    )
    print(f"Total coefficients: {len(coefficients)}")
    print(f"Vectorizer coefficients: {len(vectorizer)}")
    print(f"Linguistic coefficients: {len(linguistic)}")
    print()

    return linguistic, top_vectorizer


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(title)
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print()


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def fmt_coef(value: float | str) -> str:
    return f"{float(value):.6f}"


def build_report(
    warmup_ms: float,
    summary_rows: dict[str, dict[str, float]],
    measurement_note: str,
    linguistic: list[dict[str, float | str]],
    top_vectorizer: list[dict[str, float | str]],
    raw_console_output: str,
) -> str:
    benchmark_rows = [
        [
            length_label,
            fmt_ms(summary["mean_ms"]),
            fmt_ms(summary["median_ms"]),
            fmt_ms(summary["sd_ms"]),
        ]
        for length_label, summary in summary_rows.items()
    ]
    linguistic_rows = [[str(row["feature"]), fmt_coef(row["coefficient"])] for row in linguistic]
    vectorizer_rows = [[str(row["feature"]), fmt_coef(row["coefficient"])] for row in top_vectorizer]

    measured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return "\n".join(
        [
            "# Render Benchmark and Deployment Coefficients",
            "",
            "## Table 8 replacement (Render inference benchmark)",
            "",
            f"Endpoint: {API_ENDPOINT}",
            f"Warmup response time: {fmt_ms(warmup_ms)} ms (excluded)",
            "",
            markdown_table(["Article length", "Mean ms", "Median ms", "SD ms"], benchmark_rows),
            "",
            f"Note: {measurement_note}; measured from the local Codex workspace/client at {measured_at}.",
            "",
            "## Table 9 replacement (deployment model linguistic coefficients)",
            "",
            markdown_table(["Feature", "Coefficient"], linguistic_rows),
            "",
            "## Table 10 replacement (deployment model top vectorizer predictors)",
            "",
            markdown_table(["Feature", "Coefficient"], vectorizer_rows),
            "",
            "## Raw console output",
            "",
            "```text",
            raw_console_output.rstrip(),
            "```",
            "",
        ]
    )


def main() -> None:
    configure_stdout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    capture = io.StringIO()
    with LOG_PATH.open("w", encoding="utf-8", newline="") as log_file:
        stdout_tee = Tee(sys.__stdout__, log_file, capture)
        stderr_tee = Tee(sys.__stderr__, log_file, capture)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print("# Render Benchmark and Deployment Coefficients")
            print(f"Started: {datetime.now().astimezone().isoformat(timespec='seconds')}")
            print(f"Log path: {LOG_PATH}")
            print(f"Report path: {REPORT_PATH}")
            print()

            samples = select_samples()
            print("Selected benchmark samples:")
            for sample in samples:
                print(
                    "Sample | "
                    f"length={sample.length_label} | "
                    f"id={sample.sample_id} | "
                    f"words={sample.word_count} | "
                    f"source={sample.source}"
                )
            print()

            warmup_ms, _, benchmark_results = benchmark_render(samples)
            summary_rows = benchmark_summary(benchmark_results)
            all_server_side = all(result.server_side_ms is not None for result in benchmark_results)
            measurement_note = (
                "server-side inference time from API response"
                if all_server_side
                else "round-trip HTTP response time because the API response did not include server-side inference time"
            )

            benchmark_table_rows = [
                [
                    length_label,
                    fmt_ms(summary["mean_ms"]),
                    fmt_ms(summary["median_ms"]),
                    fmt_ms(summary["sd_ms"]),
                ]
                for length_label, summary in summary_rows.items()
            ]
            print_table(
                "Table 8 replacement (Render inference benchmark)",
                ["Article length", "Mean ms", "Median ms", "SD ms"],
                benchmark_table_rows,
            )
            print(f"Measurement note: {measurement_note}")
            print()

            linguistic, top_vectorizer = load_deployment_coefficients()
            print_table(
                "Table 9 replacement (deployment model linguistic coefficients)",
                ["Feature", "Coefficient"],
                [[str(row["feature"]), fmt_coef(row["coefficient"])] for row in linguistic],
            )
            print_table(
                "Table 10 replacement (deployment model top vectorizer predictors)",
                ["Feature", "Coefficient"],
                [[str(row["feature"]), fmt_coef(row["coefficient"])] for row in top_vectorizer],
            )

            raw_console_output = capture.getvalue()
            report = build_report(
                warmup_ms=warmup_ms,
                summary_rows=summary_rows,
                measurement_note=measurement_note,
                linguistic=linguistic,
                top_vectorizer=top_vectorizer,
                raw_console_output=raw_console_output,
            )
            REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
