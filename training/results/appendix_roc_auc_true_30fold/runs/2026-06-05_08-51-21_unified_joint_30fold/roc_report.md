# True 30-Fold Joint-Corpus ROC-AUC Report

These are true mean ROC curves with shaded +/- 1 SD bands computed over the saved 30 unified CV folds. In the full run, each model contributes 30 fold-level probability/logit files generated from the same stratified 80/20 training partition and the same repeated stratified CV splits.

Possible differences from prior manuscript tables are expected because the unified-split statistics may differ from earlier tables that used earlier protocol-specific runs. The values here prioritize apples-to-apples comparison across classical ML and transformer models.

## Methodology

- Joint corpus order: Cruz rows, then Lupac rows, with stable `combined_row_id`.
- Outer partition: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Inner evaluation: `RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=42)`.
- Positive ROC score target: class `1`.
- `LR_deploy` uses the deployment feature set (TF-IDF, BOW, READ, OOV, SW, TRAD, SYLL) and excludes LEX/MORPH.
- Figures are regenerated from saved fold prediction files.

## AUC Summary

| Model | n folds | Mean AUC | SD AUC | 95% CI | Mean accuracy | Mean macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| MNB | 30 | 0.922107 | 0.006930 | 0.919520-0.924695 | 0.858096 | 0.856341 |
| RF | 30 | 0.960543 | 0.005026 | 0.958666-0.962419 | 0.891305 | 0.891294 |
| LR | 30 | 0.975621 | 0.003664 | 0.974253-0.976989 | 0.923442 | 0.923432 |
| LR-deploy | 30 | 0.975448 | 0.003539 | 0.974126-0.976770 | 0.924027 | 0.924018 |
| SVC | 30 | 0.974054 | 0.003723 | 0.972664-0.975444 | 0.924417 | 0.924411 |
| DistilBERT | 30 | 0.973767 | 0.003872 | 0.972321-0.975213 | 0.916618 | 0.916590 |
| RoBERTa | 30 | 0.990177 | 0.002852 | 0.989112-0.991242 | 0.951777 | 0.951771 |

## Required Outputs

Fold-level prediction files and raw ROC point files are saved for independent audit and replotting.

- `run_true_30fold_roc_auc.py`
- `run_manifest.json`
- `run.log`
- `fold_splits.csv`
- `fold_metrics.csv`
- `model_auc_summary.csv`
- `roc_tpr_summary.csv`
- `roc_auc_ml_models.png`
- `roc_auc_ml_models.pdf`
- `roc_auc_all_models.png`
- `roc_auc_all_models.pdf`
- `predictions/<model>/repeat_<r>_fold_<f>_predictions.csv.gz`
- `roc_raw/<model>/repeat_<r>_fold_<f>_roc.csv`

## Figure Caption Wording

Figure. ROC Curves - Classical ML Classifiers (Joint Corpus, 30-Run CV). Lines show the mean ROC curve across unified CV folds; shaded bands show +/- 1 SD. LR-deploy is dotted.

Figure. ROC Curves - All Models (Joint Corpus, 30-Run CV). Classical ML models are shown with solid lines, LR-deploy with a dotted line, and transformer models with dashed lines. Legend entries report mean ROC-AUC +/- SD from the saved fold-level predictions.

## Completion

- Requested models: MNB, RF, LR, LR-deploy, SVC, DistilBERT, RoBERTa
- Saved fold metric rows: 210
- Full all-model target rows: 210
