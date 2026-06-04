"""Step 0: regenerate only the cached stop-word feature CSVs.

Run from the repository root:
    training\\venv\\Scripts\\python.exe training\\rerun_step0_recache_sw.py

The script writes before/after copies and diff artifacts under
training/results/stopwords_rerun/latest while replacing only:
    training/root/datasets/Cruz/SwFeatures.csv
    training/root/datasets/Lupac/SwFeatures.csv
"""

from __future__ import annotations

import difflib
import os
import shutil
import sys
from datetime import datetime
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


def read_csv_text(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def write_unified_diff(before_path: Path, after_path: Path, diff_path: Path) -> bool:
    before_lines = read_csv_text(before_path)
    after_lines = read_csv_text(after_path)
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=before_path.as_posix(),
            tofile=after_path.as_posix(),
            lineterm="",
        )
    )

    if diff_lines:
        diff_path.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
        return True

    diff_path.write_text("No differences.\n", encoding="utf-8")
    return False


def write_row_diff(before: pd.Series, after: pd.Series, path: Path) -> int:
    changed = before.ne(after)
    diff = pd.DataFrame(
        {
            "row_index": before.index[changed],
            "before_count_stopwords": before[changed].to_numpy(),
            "after_count_stopwords": after[changed].to_numpy(),
        }
    )
    diff["delta"] = diff["after_count_stopwords"] - diff["before_count_stopwords"]
    diff.to_csv(path, index=False)
    return int(changed.sum())


def regenerate_dataset(dataset_key: str, dataset_name: str, data_file: str, output_dir: Path) -> None:
    dataset_dir = BASE_DIR / "root" / "datasets" / dataset_key
    article_path = dataset_dir / data_file
    sw_path = dataset_dir / "SwFeatures.csv"
    oov_path = dataset_dir / "OovFeatures.csv"

    print(f"\n=== {dataset_name} ({dataset_key}) ===")
    print(f"Article CSV: {article_path}")
    print(f"SW CSV:      {sw_path}")
    print(f"OOV CSV:     {oov_path}")

    articles = pd.read_csv(article_path)
    before = pd.read_csv(sw_path)
    oov = pd.read_csv(oov_path)

    if "article" not in articles.columns:
        raise ValueError(f"Missing 'article' column in {article_path}")
    if "count_stopwords" not in before.columns:
        raise ValueError(f"Missing 'count_stopwords' column in {sw_path}")
    if "count_oov_words" not in oov.columns:
        raise ValueError(f"Missing 'count_oov_words' column in {oov_path}")
    if len(articles) != len(before) or len(articles) != len(oov):
        raise ValueError(
            f"Row count mismatch for {dataset_key}: "
            f"articles={len(articles)}, sw={len(before)}, oov={len(oov)}"
        )

    dataset_output_dir = output_dir / dataset_key
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    before_copy = dataset_output_dir / "SwFeatures.before.csv"
    after_copy = dataset_output_dir / "SwFeatures.after.csv"
    unified_diff_path = dataset_output_dir / "SwFeatures.before_vs_after.diff"
    row_diff_path = dataset_output_dir / "SwFeatures.changed_rows.csv"

    shutil.copy2(sw_path, before_copy)

    extractor = StopWordsExtractor(from_csv=False)
    regenerated_values = extractor.transform(articles["article"])
    regenerated = pd.DataFrame(regenerated_values, columns=["count_stopwords"])

    if len(regenerated) != len(articles):
        raise ValueError(
            f"Regenerated SW row count mismatch for {dataset_key}: "
            f"{len(regenerated)} generated for {len(articles)} articles"
        )

    regenerated.to_csv(sw_path, index=False)
    shutil.copy2(sw_path, after_copy)

    before_series = pd.to_numeric(before["count_stopwords"], errors="raise")
    after_series = pd.to_numeric(regenerated["count_stopwords"], errors="raise")
    oov_series = pd.to_numeric(oov["count_oov_words"], errors="raise")

    changed_rows = write_row_diff(before_series, after_series, row_diff_path)
    has_unified_diff = write_unified_diff(before_copy, after_copy, unified_diff_path)
    sw_matches_original = before_series.equals(after_series)
    sw_identical_to_oov = after_series.equals(oov_series)

    print(f"Rows processed: {len(articles)}")
    print(f"Changed SW rows after regeneration: {changed_rows}")
    print(f"Regenerated SW matches original CSV: {sw_matches_original}")
    print(f"Regenerated SW column identical to OOV column: {sw_identical_to_oov}")
    print(f"Unified diff contains changes: {has_unified_diff}")
    print(f"Before copy: {before_copy}")
    print(f"After copy:  {after_copy}")
    print(f"Unified diff artifact: {unified_diff_path}")
    print(f"Row diff artifact:     {row_diff_path}")

    if not sw_matches_original:
        max_delta = int((after_series - before_series).abs().max())
        print(f"Maximum absolute SW change: {max_delta}")
    if sw_identical_to_oov:
        raise AssertionError(f"Regenerated SW is identical to OOV for {dataset_key}")

    side_by_side = pd.DataFrame(
        {
            "row_index": range(10),
            "count_stopwords": after_series.head(10).to_numpy(),
            "count_oov_words": oov_series.head(10).to_numpy(),
        }
    )
    print("\nFirst 10 rows, SW and OOV side-by-side:")
    print(side_by_side.to_string(index=False))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = BASE_DIR / "results" / "stopwords_rerun"
    output_dir = base_output_dir / timestamp
    latest_dir = base_output_dir / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("# Step 0: Re-cache SW Features")
    print(f"Started: {timestamp}")
    print(f"Output directory: {output_dir}")
    print("Only SwFeatures.csv files will be regenerated.")

    for dataset in DATASETS:
        regenerate_dataset(*dataset, output_dir=output_dir)

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(output_dir, latest_dir)
    print(f"\nLatest output copy: {latest_dir}")
    print("Step 0 completed.")


if __name__ == "__main__":
    main()
