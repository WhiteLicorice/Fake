"""Print corpus word-frequency tables for the raw Cruz and Lupac article CSVs.

The script produces:
  * Table A: top 30 words by label in the combined corpus.
  * Table B: top 30 words by dataset for Cruz, Lupac, and combined.

Each table is printed twice: unfiltered, then content words only after removing
the same English and Filipino stop-word lists loaded by
training/root/scripts/SW.py. Frequencies include counts and percentages of the
token total for each group. Run from the repository root with:

    python training/word_frequency_analysis.py

If the BPE tokenizer dependency is unavailable, the script falls back to regex
word tokenization and prints a note before the tables.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "root" / "datasets"
STOPWORDS_DIR = SCRIPT_DIR / "root" / "scripts" / "stopwords"
WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)
FOCAL_WORDS = ("gma", "source", "upang", "ngunit")
TOP_N = 30

DATASETS = {
    "Cruz": DATA_ROOT / "Cruz" / "FakeNewsFilipino_Cruz2020.csv",
    "Lupac": DATA_ROOT / "Lupac" / "FakeNewsPhilippines2024_Lupac.csv",
}
LABEL_NAMES = {
    0: "Fake articles",
    1: "Real articles",
}


@dataclass
class FrequencyGroup:
    all_words: Counter[str] = field(default_factory=Counter)
    content_words: Counter[str] = field(default_factory=Counter)
    total_tokens: int = 0
    total_content_tokens: int = 0

    def update(self, words: Iterable[str], stopwords: set[str]) -> None:
        word_list = list(words)
        content_list = [word for word in word_list if word not in stopwords]

        self.all_words.update(word_list)
        self.content_words.update(content_list)
        self.total_tokens += len(word_list)
        self.total_content_tokens += len(content_list)


def extract_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def make_tokenizer() -> tuple[Callable[[str], list[str]], str | None]:
    original_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        os.chdir(SCRIPT_DIR)
        from root.scripts.BPE import BPETokenizer  # noqa: WPS433

        bpe_tokenizer = BPETokenizer()

        def tokenize_with_bpe(text: str) -> list[str]:
            encoding = bpe_tokenizer.tokenizer.encode(text)
            decoded_text = bpe_tokenizer.tokenizer.decode(
                encoding.ids, skip_special_tokens=True
            )
            return extract_words(decoded_text)

        return tokenize_with_bpe, None
    except Exception as exc:  # pragma: no cover - exercised only without deps/files
        fallback_note = (
            "Tokenizer note: BPE tokenizer could not be loaded "
            f"({exc.__class__.__name__}: {exc}); using regex word tokenization."
        )
        return extract_words, fallback_note
    finally:
        os.chdir(original_cwd)


def load_stopwords() -> set[str]:
    stopwords: set[str] = set()
    for file_name in ("en.sw", "fil.sw"):
        path = STOPWORDS_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing stop-word file: {path}")

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    return stopwords


def validate_csv_fields(path: Path, fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")

    missing = {"article", "label"} - set(fieldnames)
    if missing:
        details = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s) in {path}: {details}")


def iter_articles(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        validate_csv_fields(path, reader.fieldnames)

        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label in {path} at row {row_number}: {row['label']!r}"
                ) from exc

            if label not in LABEL_NAMES:
                raise ValueError(
                    f"Unexpected label in {path} at row {row_number}: {label}"
                )

            yield label, row["article"] or ""


def build_frequency_groups(
    tokenize: Callable[[str], list[str]], stopwords: set[str]
) -> tuple[dict[int, FrequencyGroup], dict[str, FrequencyGroup]]:
    by_label = {label: FrequencyGroup() for label in LABEL_NAMES}
    by_dataset = {
        "Cruz": FrequencyGroup(),
        "Lupac": FrequencyGroup(),
        "Combined": FrequencyGroup(),
    }

    for dataset, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset CSV: {path}")

        for label, article in iter_articles(path):
            words = tokenize(article)
            by_label[label].update(words, stopwords)
            by_dataset[dataset].update(words, stopwords)
            by_dataset["Combined"].update(words, stopwords)

    return by_label, by_dataset


def top_items(counter: Counter[str], limit: int = TOP_N) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def print_frequency_rows(counter: Counter[str], total: int) -> None:
    print("    Rank  Word            Count   %")
    top_words = top_items(counter)
    for rank, (word, count) in enumerate(top_words, start=1):
        percent = (count / total * 100) if total else 0.0
        print(f"    {rank:>4}  {word:<14} {count:>7}  {percent:>5.2f}")

    flagged = [word for word in FOCAL_WORDS if word in dict(top_words)]
    if flagged:
        print(
            f"    Note: flagged manuscript terms in top {TOP_N}: "
            + ", ".join(flagged)
        )


def print_group(
    title: str,
    group: FrequencyGroup,
    *,
    content_only: bool,
) -> None:
    if content_only:
        removed = group.total_tokens - group.total_content_tokens
        print(
            f"  {title} "
            f"(n={group.total_content_tokens} content tokens; "
            f"removed {removed} stop-word tokens from {group.total_tokens})"
        )
        print_frequency_rows(group.content_words, group.total_content_tokens)
    else:
        print(f"  {title} (n={group.total_tokens} tokens)")
        print_frequency_rows(group.all_words, group.total_tokens)


def print_label_table(by_label: dict[int, FrequencyGroup]) -> None:
    print("=== Table A: Top 30 words by label (combined corpus) ===")
    print()
    print("Unfiltered:")
    for label in (0, 1):
        print_group(LABEL_NAMES[label], by_label[label], content_only=False)
    print()
    print("Content words only:")
    for label in (0, 1):
        print_group(LABEL_NAMES[label], by_label[label], content_only=True)


def print_dataset_table(by_dataset: dict[str, FrequencyGroup]) -> None:
    print("=== Table B: Top 30 words by dataset ===")
    print()
    print("Unfiltered:")
    for dataset in ("Cruz", "Lupac", "Combined"):
        print_group(f"{dataset} articles", by_dataset[dataset], content_only=False)
    print()
    print("Content words only:")
    for dataset in ("Cruz", "Lupac", "Combined"):
        print_group(f"{dataset} articles", by_dataset[dataset], content_only=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    stopwords = load_stopwords()
    tokenize, fallback_note = make_tokenizer()
    by_label, by_dataset = build_frequency_groups(tokenize, stopwords)

    if fallback_note:
        print(fallback_note)
        print()

    print_label_table(by_label)
    print()
    print_dataset_table(by_dataset)


if __name__ == "__main__":
    main()
