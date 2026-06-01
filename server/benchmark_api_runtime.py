"""Benchmark the live FaKe Render API with warmed-up inference requests.

Run from the repository root with:
    python server/benchmark_api_runtime.py

The script sends one warmup POST request to the deployed FastAPI endpoint,
then sends 10 timed requests for each of three fixed Filipino sample articles
and prints response-time summaries in milliseconds. The warmup request is
excluded from the benchmark.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import requests


API_ENDPOINT = "https://fph-ml.onrender.com/check-news"
REQUESTS_PER_ARTICLE = 10
TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ArticleSample:
    title: str
    text: str


SAMPLES = [
    ArticleSample(
        "Short article (~50 words)",
        """Patay ang isang 5-anyos na lalaki sa Lapu-Lapu City, Cebu matapos
umano siyang pukpukin sa ulo at ihagis sa dagat ng kaniyang 14-anyos
na kapatid. Sa ulat ng ABS-CBN News, nangyari ang insidente sa Sitio
Lawis, Barangay Suba-Basbas nitong Biyernes ng umaga, Mayo 10.""",
    ),
    ArticleSample(
        "Medium article (~100 words)",
        """SAN CARLOS CITY, Pangasinan – Dalawang hinihinalang carnapper na
nagpapanggap na miyembro ng Criminal Investigation and Detection Group
(CIDG) ang naaresto sa San Carlos City, Pangasinan. Sa kanyang report
kay Pangasinan Police Provincial Office director Senior Supt. Reynaldo
Biay, kinilala ni San Carlos City Police chief Supt. Charlie Umayam
ang mga nadakip na sina Michael Edades, 34, may asawa, negosyante, at
residente ng Barangay Mangin, Dagupan City; at Daniel Salopagio Jr.,
29, binata, bus driver, ng Bgy. Nalsian Norte, Bayambang.""",
    ),
    ArticleSample(
        "Long article (~200 words)",
        """Mahaharap sa kasong administratibo ang isang opisyal ng pulisya matapos
magwala sa mismong himpilan, pinasok sa opisina ang kanyang hepe at
pinagsasalitaan umano ng masama, Lunes ng gabi, sa Bacoor City, Cavite.
Kasong grave misconduct ang kakaharapin ni Chief Insp. Virgilio Rubio,
deputy chief ng Bacoor City Police, batay sa reklamo ni Supt. Rommel
Estolano. Sa ulat sa tanggapan ni Cavite Police Provincial Office
director Senior Supt. Joselito Esquivel, bandang 7:30 ng gabi nang
magtungo sa istasyon ng pulisya si Rubio na lasing na lasing, biglang
pinaghahagis ang mga upuan at iba pang gamit sa opisina hanggang pumasok
sa tanggapan ni Rubio habang may hawak na baril at nagbitiw ng kung
anu-anong masasamang salita. Gayunman, naawat ni Senior Insp. Chey Chey
Saulog si Rubio at pinalabas sa himpilan ng pulisya. Kaugnay nito,
sinabi ni Estolano na hindi niya alam kung bakit nagalit si Rubio sa
kanya at binanggit niya na palagi itong magalang sa kanya noon.""",
    ),
]


def post_article(session: requests.Session, text: str) -> tuple[int, dict]:
    start = time.perf_counter()
    response = session.post(
        API_ENDPOINT,
        json={"news_body": text},
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    response.raise_for_status()
    return elapsed_ms, response.json()


def warmup(session: requests.Session) -> int:
    try:
        elapsed_ms, _ = post_article(session, SAMPLES[0].text)
        return elapsed_ms
    except requests.RequestException:
        time.sleep(30)
        elapsed_ms, _ = post_article(session, SAMPLES[0].text)
        return elapsed_ms


def summarize(values: list[int]) -> tuple[float, float, float]:
    return (
        statistics.mean(values),
        statistics.median(values),
        statistics.stdev(values),
    )


def main() -> None:
    session = requests.Session()
    warmup_ms = warmup(session)

    results: dict[str, list[int]] = {}
    for sample in SAMPLES:
        sample_times = []
        for _ in range(REQUESTS_PER_ARTICLE):
            elapsed_ms, _ = post_article(session, sample.text)
            sample_times.append(elapsed_ms)
        results[sample.title] = sample_times

    all_times = [elapsed_ms for values in results.values() for elapsed_ms in values]

    print(f"API endpoint: {API_ENDPOINT}")
    print(f"Warmup response time: {warmup_ms} ms (excluded from benchmark)")
    print()

    for sample in SAMPLES:
        values = results[sample.title]
        mean_ms, median_ms, sd_ms = summarize(values)
        print(sample.title)
        print(f"  Responses (ms): {values}")
        print(f"  Mean: {mean_ms:.0f} ms,  Median: {median_ms:.0f} ms,  SD: {sd_ms:.0f} ms")
        print()

    overall_mean = statistics.mean(all_times)
    print(f"Overall mean across all 30 requests: {overall_mean:.0f} ms")
    print(f"Overall range: {min(all_times)} ms – {max(all_times)} ms")


if __name__ == "__main__":
    main()
