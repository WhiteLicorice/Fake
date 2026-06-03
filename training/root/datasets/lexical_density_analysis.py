# 2026-06-03: Computes lexical_density structure checks and corpus summary statistics.
import os
import pandas as pd


BASE_DIR = r"D:\Creative Corner\Projects\Software\Fake\training\root\datasets"
LEXICAL_COLUMN = "lexical_density"
FEATURE_COLUMNS = {
    "ttr",
    "root_ttr",
    "corr_ttr",
    "log_ttr",
    "noun_tr",
    "verb_tr",
    "lexical_density",
    "foreign_tr",
    "compound_tr",
}
LABEL_VALUE_NAMES = {0: "Fake", 1: "Real"}


def rel(path):
    return os.path.relpath(path, BASE_DIR)


def fmt(value):
    return f"{value:.6f}"


def sorted_csv_paths(base_dir):
    paths = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for filename in sorted(files):
            if filename.lower().endswith(".csv"):
                paths.append(os.path.join(root, filename))
    return paths


def display_values(values):
    clean_values = []
    for value in values:
        if hasattr(value, "item"):
            clean_values.append(value.item())
        else:
            clean_values.append(value)
    clean_values = sorted(clean_values, key=lambda item: str(item))
    return "[" + ", ".join(repr(value) for value in clean_values) + "]"


def find_label_column(df):
    preferred_names = ["label", "class", "target", "truth", "is_fake", "is_real"]
    for name in preferred_names:
        if name in df.columns:
            return name

    for column in df.columns:
        if column in FEATURE_COLUMNS:
            continue
        unique_count = df[column].dropna().nunique()
        if 1 < unique_count <= 10:
            return column
    return None


def load_labels_for_lex_file(lex_path, expected_rows):
    lex_df = pd.read_csv(lex_path)
    label_column = find_label_column(lex_df)
    if label_column is not None:
        return lex_df[label_column], label_column, lex_path, "feature file"

    folder = os.path.dirname(lex_path)
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".csv"):
            continue
        candidate_path = os.path.join(folder, filename)
        if os.path.abspath(candidate_path) == os.path.abspath(lex_path):
            continue

        candidate_df = pd.read_csv(candidate_path)
        label_column = find_label_column(candidate_df)
        if label_column is None:
            continue
        if len(candidate_df) != expected_rows:
            continue
        return candidate_df[label_column], label_column, candidate_path, "matching sibling file by row order"

    raise RuntimeError("Could not find a label column matching " + rel(lex_path))


def summarize(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        {
            "Mean": numeric.mean(),
            "SD": numeric.std(),
            "Min": numeric.min(),
            "Max": numeric.max(),
            "P10": numeric.quantile(0.10),
            "P90": numeric.quantile(0.90),
        }
    )


def print_formatted_table(table):
    formatted = table.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(fmt)
    print(formatted.to_string())


csv_paths = sorted_csv_paths(BASE_DIR)
lex_files = []

for path in csv_paths:
    df = pd.read_csv(path, nrows=1)
    if LEXICAL_COLUMN in df.columns:
        lex_files.append(path)

print("CSV files found under dataset directory:")
for path in csv_paths:
    print("- " + rel(path))

print()
print("LEX feature files used:")
for path in lex_files:
    print("- " + rel(path))

if not lex_files:
    raise RuntimeError("No CSV files with lexical_density were found.")

print()
print("Step 1 - Confirm structure")

dataset_frames = []
dataset_stats = {}
structure_warnings = []
row_count_ok = True
missing_ok = True

for lex_path in lex_files:
    lex_df = pd.read_csv(lex_path)
    lexical = pd.to_numeric(lex_df[LEXICAL_COLUMN], errors="coerce")
    labels, label_column, label_source_path, label_source_type = load_labels_for_lex_file(
        lex_path, len(lex_df)
    )

    if len(labels) != len(lex_df):
        row_count_ok = False
    null_count = int(lexical.isna().sum())
    if null_count != 0:
        missing_ok = False
    above_one_count = int((lexical > 1).sum())
    if above_one_count:
        structure_warnings.append(
            f"{rel(lex_path)} has {above_one_count} lexical_density values above 1.0"
        )

    unique_labels = labels.dropna().unique().tolist()
    label_interpretation = ", ".join(
        f"{key}={LABEL_VALUE_NAMES[key]}" for key in sorted(LABEL_VALUE_NAMES)
    )

    print("Filename: " + rel(lex_path))
    print("Rows: " + str(len(lex_df)))
    print("Label source file: " + rel(label_source_path))
    print("Label source type: " + label_source_type)
    print("Label column: " + label_column)
    print("Label unique values: " + display_values(unique_labels))
    print("Label interpretation: " + label_interpretation)
    print("lexical_density min: " + fmt(lexical.min()))
    print("lexical_density max: " + fmt(lexical.max()))
    print("lexical_density count (non-null): " + str(int(lexical.count())))
    print("lexical_density null count: " + str(null_count))
    print("lexical_density > 1 count: " + str(above_one_count))
    if above_one_count:
        print("WARNING: lexical_density contains values above 1.0; retained as-is.")
    print()

    dataset_name = os.path.basename(os.path.dirname(lex_path))
    merged = pd.DataFrame(
        {
            "dataset": dataset_name,
            "source_file": rel(lex_path),
            LEXICAL_COLUMN: lexical,
            "label": labels.reset_index(drop=True),
        }
    )
    dataset_frames.append(merged)
    dataset_stats[dataset_name] = summarize(lexical)

if row_count_ok and missing_ok and not structure_warnings:
    print("Step 1 clean check: PASS")
elif row_count_ok and missing_ok:
    print("Step 1 clean check: PASS WITH WARNING")
else:
    print("Step 1 clean check: FAIL")

print("- Label row counts match feature row counts: " + str(row_count_ok))
print("- Missing lexical_density values absent: " + str(missing_ok))
if structure_warnings:
    for warning in structure_warnings:
        print("- Warning: " + warning + "; retained as-is for transparent corpus statistics.")
else:
    print("- No lexical_density values above 1.0.")

if not row_count_ok or not missing_ok:
    raise RuntimeError("Step 1 failed; summary statistics were not computed.")

print()
print("Step 2 - Summary statistics")
print()
print("A. Each dataset separately:")
dataset_table = pd.DataFrame(dataset_stats)
print_formatted_table(dataset_table)

combined = pd.concat(dataset_frames, ignore_index=True)
fake = combined[combined["label"] == 0][LEXICAL_COLUMN]
real = combined[combined["label"] == 1][LEXICAL_COLUMN]

combined_table = pd.DataFrame(
    {
        "Full Corpus": summarize(combined[LEXICAL_COLUMN]),
        "Fake only": summarize(fake),
        "Real only": summarize(real),
    }
)

print()
print("B. Combined corpus:")
print_formatted_table(combined_table)

fake_mean = fake.mean()
real_mean = real.mean()
mean_difference = abs(fake_mean - real_mean)
full_mean = combined[LEXICAL_COLUMN].mean()
full_sd = combined[LEXICAL_COLUMN].std()
coefficient_of_variation = full_sd / full_mean

print()
print("Absolute fake-real mean difference: " + fmt(mean_difference))
print(
    "Combined coefficient of variation: "
    + fmt(coefficient_of_variation)
    + " ("
    + f"{coefficient_of_variation * 100:.2f}%"
    + ")"
)

print()
print("Step 3 - Interpretation")
if coefficient_of_variation < 0.15:
    print(
        "Lexical density is approximately stable across the combined corpus: "
        + "CV = "
        + f"{coefficient_of_variation * 100:.2f}%"
        + " (< 15%), suggesting total word count is a near-linear proxy for content word count."
    )
else:
    print(
        "Lexical density is not approximately stable by the requested threshold: "
        + "CV = "
        + f"{coefficient_of_variation * 100:.2f}%"
        + " (>= 15%), so the near-linear proxy claim should be qualified."
    )

if mean_difference > 0.05:
    print(
        "Fake and real articles differ substantially in mean lexical density: absolute difference = "
        + fmt(mean_difference)
        + " (> 0.05)."
    )
else:
    print(
        "Fake and real articles do not differ substantially in mean lexical density by the > 0.05 threshold: absolute difference = "
        + fmt(mean_difference)
        + "."
    )
