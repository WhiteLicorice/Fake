"""Step 1: verify StopWordsExtractor(from_csv=True) reads count_stopwords."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from root.scripts.FILTRANS import StopWordsExtractor  # noqa: E402


DATASETS = [
    ("Cruz", "Fake News Filipino 2020", "FakeNewsFilipino_Cruz2020.csv"),
    ("Lupac", "Fake News Filipino 2024", "FakeNewsPhilippines2024_Lupac.csv"),
]


def verify_dataset(dataset_key: str, dataset_name: str, data_file: str) -> None:
    dataset_dir = BASE_DIR / "root" / "datasets" / dataset_key
    articles = pd.read_csv(dataset_dir / data_file)
    sw = pd.read_csv(dataset_dir / "SwFeatures.csv")
    oov = pd.read_csv(dataset_dir / "OovFeatures.csv")
    frame = pd.concat(
        [
            articles.reset_index(drop=True),
            sw.reset_index(drop=True),
            oov.reset_index(drop=True),
        ],
        axis=1,
    )

    sample = frame.head(10)
    extractor = StopWordsExtractor(from_csv=True)
    extracted = [row[0] for row in extractor.transform(sample)]

    result = pd.DataFrame(
        {
            "row_index": sample.index,
            "extractor_from_csv": extracted,
            "csv_count_stopwords": sample["count_stopwords"].to_numpy(),
            "csv_count_oov_words": sample["count_oov_words"].to_numpy(),
        }
    )
    result["matches_sw_csv"] = (
        result["extractor_from_csv"] == result["csv_count_stopwords"]
    )
    result["matches_oov_csv"] = (
        result["extractor_from_csv"] == result["csv_count_oov_words"]
    )

    print(f"\n=== {dataset_name} ({dataset_key}) ===")
    print(result.to_string(index=False))
    print(f"All sample extractor outputs match SW CSV: {bool(result['matches_sw_csv'].all())}")
    print(f"All sample extractor outputs match OOV CSV: {bool(result['matches_oov_csv'].all())}")

    if not result["matches_sw_csv"].all():
        raise AssertionError(f"{dataset_key}: from_csv extractor did not match count_stopwords")
    if result["extractor_from_csv"].equals(result["csv_count_oov_words"]):
        raise AssertionError(f"{dataset_key}: from_csv extractor output is identical to OOV")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("# Step 1: Verify StopWordsExtractor(from_csv=True)")
    for dataset in DATASETS:
        verify_dataset(*dataset)
    print("\nStep 1 completed.")


if __name__ == "__main__":
    main()
