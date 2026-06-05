# ROC-AUC Figure Report for Appendix II

## Purpose

This report documents the generated ROC-AUC figures in this directory and the methodological caveat that must accompany them. It is written for downstream LLM or reviewer consumption, so the key assumptions and limitations are explicit.

## Generated Files

Directory:

```text
D:\Creative Corner\Projects\Software\Fake\training\results\appendix_roc_auc
```

Figure outputs:

```text
roc_auc_ml_models.png
roc_auc_ml_models.pdf
roc_auc_all_models.png
roc_auc_all_models.pdf
```

Generation script:

```text
generate_roc_figures_from_metadata.py
```

Machine-readable metadata:

```text
roc_auc_figure_metadata.json
```

## Data Sources Used

Classical ML scalar AUC source:

```text
D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\roc_auc_30run_raw.csv
```

DistilBERT scalar AUC source:

```text
D:\Creative Corner\Projects\Software\Fake\training\results\distilbert_tagalog_baseline\2026-06-03_full\metrics.csv
```

RoBERTa scalar AUC source:

```text
D:\Creative Corner\Projects\Software\Fake\training\results\roberta_tagalog_baseline\2026-06-04_full\metrics.csv
```

DistilBERT representative model:

```text
D:\Creative Corner\Projects\Software\Fake\training\results\distilbert_tagalog_baseline\2026-06-03_full\representative_joint_model_closest_to_mean
```

RoBERTa representative model:

```text
D:\Creative Corner\Projects\Software\Fake\training\results\roberta_tagalog_baseline\2026-06-04_full\representative_joint_model
```

## Main Result

The figures were successfully generated, but they are not true mean ROC curves with true +/- 1 SD TPR bands. They are representative-fold ROC curves with 30-run mean ROC-AUC +/- SD values shown in the legends.

This caveat applies to all six plotted curves:

```text
LR
MNB
RF
SVC
DistilBERT
RoBERTa
```

The plotted ROC curves are apples-to-apples in the following limited sense:

```text
All plotted ROC curves are representative-fold curves.
All legend AUC values are 30-run mean ROC-AUC +/- SD values.
```

The plotted ROC curves are not apples-to-apples as true 30-fold mean ROC curves because fold-level probability arrays or FPR/TPR arrays were not available for all models.

## AUC Values Used in Figure Legends

Values below are mean ROC-AUC +/- SD over 30 CV runs on the joint corpus:

| Model | Mean ROC-AUC | SD | Representative Repeat | Representative Fold | Representative Fold AUC |
|---|---:|---:|---:|---:|---:|
| RoBERTa | 0.990182 | 0.002838 | 2 | 10 | 0.993882 |
| LR | 0.975503 | 0.004597 | 3 | 3 | 0.975470 |
| DistilBERT | 0.973804 | 0.003926 | 3 | 15 | 0.973699 |
| SVC | 0.973595 | 0.005171 | 5 | 3 | 0.973810 |
| RF | 0.960274 | 0.005898 | 3 | 1 | 0.960553 |
| MNB | 0.921877 | 0.009800 | 4 | 5 | 0.921942 |

Legend order in `roc_auc_all_models` is descending by mean ROC-AUC.

## What Was Available

The saved result artifacts contained:

```text
30 scalar ROC-AUC values per model.
Representative saved transformer models for DistilBERT and RoBERTa.
Enough source code and dataset inputs to reconstruct representative validation folds.
```

The saved result artifacts did not contain:

```text
Fold-level y_true arrays for all models.
Fold-level y_score/probability arrays for all models.
Fold-level FPR/TPR arrays for all models.
Persisted non-empty transformer checkpoints for every fold.
```

Because of that, true mean ROC curve computation across all 30 folds was not possible from saved metadata alone.

## Method Actually Used

For each model, the plotted curve is based on one representative fold.

For classical ML models:

```text
The script selected the fold whose saved scalar AUC was closest to that model's 30-run mean AUC.
It refit the classical model on that fold using the existing rerun pipeline.
It computed y_score on the fold validation partition.
It computed and plotted the ROC curve for that representative fold.
```

For DistilBERT and RoBERTa:

```text
The script loaded the saved representative joint model.
It reconstructed that representative fold's validation partition from the original split protocol.
It ran inference on the validation partition.
It computed and plotted the ROC curve for that representative fold.
```

For all models:

```text
The legend reports the 30-run mean ROC-AUC +/- SD from saved scalar metrics.
The curve line is from one representative fold.
No true +/- 1 SD shaded TPR band is shown, because there is only one plotted ROC curve per model in the generated output.
```

## Important Caveat

Do not describe these figures as:

```text
mean ROC curves with shaded +/- 1 SD bands across 30 CV runs
```

That statement would be inaccurate for the current generated files.

Accurate wording:

```text
Representative-fold ROC curves on the joint corpus. Legend values report mean ROC-AUC +/- SD across 30 CV runs.
```

Suggested figure caption:

```text
ROC curves on the joint corpus. Curves are from representative validation folds selected from the 30-run CV protocol; legend values report mean ROC-AUC +/- SD across the 30 CV runs.
```

## Implication for Apples-to-Apples Comparison

The current figures are apples-to-apples only for visual comparison of representative-fold ROC shape. They are apples-to-apples for legend statistics because all legend AUC values are 30-run mean +/- SD values.

They are not sufficient for an apples-to-apples comparison of mean ROC curve shape or TPR variability across folds, because those require fold-level probability arrays or FPR/TPR arrays for every model.

## Requirements for True Mean ROC +/- SD Figures

To generate true mean ROC curves with shaded +/- 1 SD TPR bands, the project needs one of the following for every fold and every model:

```text
y_true and y_score arrays
```

or:

```text
FPR and TPR arrays
```

Classical ML can be recomputed from the existing rerun pipeline, but doing all 30 folds is expensive. The existing runtime logs suggest LR and SVC are the slowest classical recomputations.

For DistilBERT and RoBERTa, true 30-fold ROC curves would require either:

```text
saved fold-level logits/probabilities from the original run
```

or:

```text
saved non-empty checkpoints for all 30 folds
```

or:

```text
rerunning transformer evaluation/training to regenerate fold-level probabilities
```

## Recommendation for Manuscript Use

Use the generated figures only if the appendix caption explicitly states that the ROC curves are representative-fold curves and that the AUC values are 30-run mean +/- SD summaries.

If the journal or reviewers expect true mean ROC curves with variability bands, regenerate the underlying fold-level probabilities first and then recreate the figures from those arrays.
