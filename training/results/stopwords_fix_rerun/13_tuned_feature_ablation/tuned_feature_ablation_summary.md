# Tuned Hyperparameter Progressive Feature Ablation

Started: 2026-06-04 04:49:35
Finished: 2026-06-04 11:53:28

Protocol: stratified 80% training partition, followed by 6 repetitions of 5-fold cross-validation on the training partition, base random_state=42.

Tuned hyperparameters: MNB alpha=0.1; LR C=1.0, max_iter=2000; RF n_estimators=100, max_depth=20, min_samples_split=2; SVC C=0.1, kernel=linear.

Raw fold-level CSV: `D:/Creative Corner/Projects/Software/Fake/training/results/stopwords_fix_rerun/13_tuned_feature_ablation/tuned_feature_ablation_raw.csv`
Console log: `D:/Creative Corner/Projects/Software/Fake/training/results/stopwords_fix_rerun/13_tuned_feature_ablation/tuned_feature_ablation_console.log`

## Mean Accuracy Tables

### Fake News Filipino 2020

| Feature Set | MNB | LR | RF | SVC |
|---|---:|---:|---:|---:|
| Vectorizers (TF-IDF + BOW) | 0.9040 | 0.9475 | 0.9243 | 0.9487 |
| + Readability (READ) | 0.9027 | 0.9469 | 0.9222 | 0.9490 |
| + Out-of-vocabulary (OOV) | 0.9028 | 0.9495 | 0.9244 | 0.9497 |
| + Stop words (SW) | 0.8953 | 0.9492 | 0.9210 | 0.9497 |
| + Traditional features (TRAD) | 0.9199 | 0.9522 | 0.9215 | 0.9516 |
| + Syllabic features (SYLL) | 0.9210 | 0.9523 | 0.9206 | 0.9513 |
| + Lexical features (LEX) | 0.9207 | 0.9521 | 0.9189 | 0.9512 |
| + Morphological features (MORPH) [full set] | 0.9205 | 0.9523 | 0.9193 | 0.9519 |

### Fake News Filipino 2024

| Feature Set | MNB | LR | RF | SVC |
|---|---:|---:|---:|---:|
| Vectorizers (TF-IDF + BOW) | 0.8546 | 0.9478 | 0.9285 | 0.9494 |
| + Readability (READ) | 0.8524 | 0.9477 | 0.9293 | 0.9489 |
| + Out-of-vocabulary (OOV) | 0.8506 | 0.9468 | 0.9262 | 0.9485 |
| + Stop words (SW) | 0.8430 | 0.9467 | 0.9274 | 0.9486 |
| + Traditional features (TRAD) | 0.8726 | 0.9455 | 0.9280 | 0.9468 |
| + Syllabic features (SYLL) | 0.8772 | 0.9457 | 0.9264 | 0.9469 |
| + Lexical features (LEX) | 0.8808 | 0.9461 | 0.9263 | 0.9469 |
| + Morphological features (MORPH) [full set] | 0.8814 | 0.9455 | 0.9258 | 0.9470 |

### Joint corpus

| Feature Set | MNB | LR | RF | SVC |
|---|---:|---:|---:|---:|
| Vectorizers (TF-IDF + BOW) | 0.7864 | 0.9233 | 0.8941 | 0.9228 |
| + Readability (READ) | 0.7843 | 0.9225 | 0.8939 | 0.9226 |
| + Out-of-vocabulary (OOV) | 0.7801 | 0.9241 | 0.8936 | 0.9237 |
| + Stop words (SW) | 0.7708 | 0.9242 | 0.8920 | 0.9236 |
| + Traditional features (TRAD) | 0.8473 | 0.9237 | 0.8914 | 0.9236 |
| + Syllabic features (SYLL) | 0.8523 | 0.9237 | 0.8918 | 0.9236 |
| + Lexical features (LEX) | 0.8564 | 0.9233 | 0.8907 | 0.9232 |
| + Morphological features (MORPH) [full set] | 0.8563 | 0.9231 | 0.8905 | 0.9227 |

## Comparison With Default-Parameter Joint Ablation

Default-reference values are from Manuscript.docx Table 7, which reports the joint-corpus progressive ablation under default classifier parameters.

- Default SVC changed from 0.9212 at vectorizers only to 0.7825 when TRAD entered (-0.1387); from +SW to +TRAD, it changed -0.1187.
- Tuned SVC changed from 0.9228 at vectorizers only to 0.9236 when TRAD entered (+0.0008); from +SW to +TRAD, it changed +0.0000. The tuned linear SVC reduces the default SVC drop when TRAD enters the feature set.
- Full-set tuned minus default joint-corpus deltas: MNB: +0.2312; LR: -0.0282; RF: -0.0403; SVC: +0.1403.

## Raw Console Output

```text
Tuned Feature Ablation Results (2026-06-04 04:49:35)
Output directory: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\13_tuned_feature_ablation
Raw CSV: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\13_tuned_feature_ablation\tuned_feature_ablation_raw.csv
Console log: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\13_tuned_feature_ablation\tuned_feature_ablation_console.log
Protocol: train_test_split(test_size=0.2, stratify=y, random_state=42), then RepeatedKFold(n_splits=5, n_repeats=6, random_state=42).
Tuned classifiers: MNB(alpha=0.1); LR(C=1.0, max_iter=2000); RF(n_estimators=100, max_depth=20, min_samples_split=2); SVC(C=0.1, kernel=linear).
Using existing Step 0 stop-word caches; no feature CSVs are regenerated here.
Resuming from existing raw CSV | rows=480 | path=D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\13_tuned_feature_ablation\tuned_feature_ablation_raw.csv

Dataset loaded | key=Cruz | name=Fake News Filipino 2020 | rows=3206 | training_rows=2564
Skipping completed condition | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | rows=120
Skipping completed condition | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | rows=120
Skipping completed condition | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | rows=120
Skipping completed condition | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | rows=120

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=3 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=4 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.925781250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=5 | accuracy=0.929687500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=2 | accuracy=0.974658869
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.974658869
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.925781250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=2 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.929687500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=5 | accuracy=0.910156250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=5 | accuracy=0.962890625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=1 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=1 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=3 | accuracy=0.898635478
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.902343750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.949218750
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.919853979
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.952158717
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.921542626
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.951573541

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=4 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.925781250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=1 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=2 | accuracy=0.974658869
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.974658869
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=5 | accuracy=0.916015625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=1 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=2 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.929687500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.951171875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=5 | accuracy=0.962890625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=5 | accuracy=0.910156250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=1 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=3 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.910156250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.949218750
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.920959227
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.952288672
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.920633960
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.951313505

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=1 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.966861598
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=3 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=4 | accuracy=0.896686160
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=5 | accuracy=0.929687500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=1 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=2 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=5 | accuracy=0.925781250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=1 | accuracy=0.896686160
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=2 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.972709552
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.921875000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=1 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=2 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.951171875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=4 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=5 | accuracy=0.910156250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=3 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.949218750
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.920699191
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.952092598
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.918943663
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.951248655

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=1 | accuracy=0.970760234
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=3 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=4 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=5 | accuracy=0.960937500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=1 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=3 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=3 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=1 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=2 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=2 | accuracy=0.972709552
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=5 | accuracy=0.921875000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=1 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=1 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=2 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=2 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=4 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=5 | accuracy=0.951171875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=1 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=4 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=5 | accuracy=0.912109375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=3 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=3 | accuracy=0.886939571
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=5 | accuracy=0.947265625
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.920504132
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.952287783
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.919333653
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.951898174

Dataset loaded | key=Lupac | name=Fake News Filipino 2024 | rows=3206 | training_rows=2564

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.849902534
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.871345029
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.859649123
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.851562500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=5 | accuracy=0.921875000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.834307992
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.838206628
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.847953216
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.865234375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=1 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.849902534
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=2 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.849609375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.818713450
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.838206628
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=4 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.855468750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.834307992
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=3 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=4 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.847656250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.836257310
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.818713450
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=3 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=4 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.873046875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.953125000
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.854590161
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.947802068
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.928495827
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.949361522

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.844054581
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.849902534
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.869395712
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.847656250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.832358674
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.840155945
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.857421875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.849902534
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.853801170
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.847656250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.822612086
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.838206628
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.851562500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.853801170
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=2 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.830409357
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=3 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.871345029
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=4 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.966861598
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.847656250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=5 | accuracy=0.921875000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.836257310
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.814814815
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.869140625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.955078125
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.852444516
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.947736837
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.929342562
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.948906808

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.844054581
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.851851852
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.867446394
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=3 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.853801170
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.847656250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=5 | accuracy=0.951171875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=5 | accuracy=0.925781250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.828460039
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.840155945
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=3 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.857421875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.851851852
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.847656250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=5 | accuracy=0.929687500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.816764133
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.838206628
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.851562500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.828460039
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.871345029
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=4 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.845703125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=5 | accuracy=0.916015625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.832358674
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.812865497
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.847953216
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.867187500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.957031250
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.850559921
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.946762686
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.926220989
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.948517326

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.832358674
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.832031250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=5 | accuracy=0.951171875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.824561404
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.828460039
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.838206628
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.847656250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.846003899
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.840155945
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.865497076
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.828460039
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.837890625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.808966862
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.832358674
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.839843750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=5 | accuracy=0.916015625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.844054581
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.847953216
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.822612086
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=3 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=4 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.832031250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.826510721
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=2 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.808966862
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.840155945
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=4 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.861328125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.957031250
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.842953267
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.946697709
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.927392737
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.948582303

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.853801170
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.886939571
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.857421875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.898635478
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.859649123
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.847953216
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.865497076
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.871093750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=1 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.859649123
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.896686160
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.873046875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=1 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.861598441
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.861328125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=5 | accuracy=0.910156250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.867446394
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=3 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.884765625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=5 | accuracy=0.925781250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.847953216
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.867446394
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.896484375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.960937500
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.872595461
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.945526849
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.927976771
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.946826521

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.861598441
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=1 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=1 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.859375000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.865497076
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.853801170
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.867446394
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.875000000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=5 | accuracy=0.917968750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=1 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.861598441
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.881091618
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.878906250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.871345029
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.865497076
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.867187500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=5 | accuracy=0.908203125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.861598441
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.890625000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=5 | accuracy=0.917968750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.857699805
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.851851852
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.902343750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.960937500
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.877210750
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.945721907
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.926350943
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.946891498

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.861328125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=5 | accuracy=0.943359375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=1 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.871345029
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.861598441
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=3 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.871345029
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.878906250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=5 | accuracy=0.925781250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=1 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.886718750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.869395712
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.875243665
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=4 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.873046875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.865497076
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=4 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.884765625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.896686160
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.908203125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.958984375
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.880785768
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.946111771
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.926286600
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.946891498

Feature-set condition | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=1 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=3 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=3 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=4 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=5 | accuracy=0.863281250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=1 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=2 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=3 | accuracy=0.861598441
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=4 | accuracy=0.871345029
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=4 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=5 | accuracy=0.878906250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=1 | accuracy=0.881091618
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=1 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=2 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=3 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=4 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=5 | accuracy=0.884765625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=1 | accuracy=0.869395712
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=3 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=3 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=3 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=4 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=4 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=5 | accuracy=0.873046875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=1 | accuracy=0.877192982
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=2 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=2 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=3 | accuracy=0.865497076
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=3 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=4 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=4 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=5 | accuracy=0.886718750
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=1 | accuracy=0.863547758
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=1 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=1 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=2 | accuracy=0.898635478
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=3 | accuracy=0.855750487
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=4 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=4 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=5 | accuracy=0.908203125
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2024 | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=5 | accuracy=0.958984375
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.881370690
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.945461998
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.925767671
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.947021833

Dataset loaded | key=Joint | name=Joint corpus | rows=6412 | training_rows=5129

Feature-set condition | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.785575049
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=1 | accuracy=0.915204678
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.779727096
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=2 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=2 | accuracy=0.868421053
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.790448343
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.804093567
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.760975610
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=5 | accuracy=0.922926829
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=5 | accuracy=0.883902439
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.924878049
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.783625731
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=1 | accuracy=0.888888889
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.789473684
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=2 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=2 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.826510721
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=3 | accuracy=0.893762183
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.785575049
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=4 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=4 | accuracy=0.892787524
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.766829268
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=5 | accuracy=0.924878049
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=5 | accuracy=0.889756098
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.921951220
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.776803119
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=1 | accuracy=0.884990253
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.769980507
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=2 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=2 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.786549708
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=3 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=3 | accuracy=0.896686160
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.800194932
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=4 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=4 | accuracy=0.889863548
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.793170732
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=5 | accuracy=0.921951220
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=5 | accuracy=0.902439024
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.924878049
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.789473684
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=1 | accuracy=0.912280702
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.795321637
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.776803119
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=3 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.771929825
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=4 | accuracy=0.880116959
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.804878049
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=5 | accuracy=0.918048780
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=5 | accuracy=0.898536585
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.914146341
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.819688109
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=1 | accuracy=0.907407407
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.944444444
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.778752437
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=2 | accuracy=0.932748538
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=2 | accuracy=0.899610136
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.780701754
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=3 | accuracy=0.878167641
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.789473684
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=4 | accuracy=0.899610136
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.776585366
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=5 | accuracy=0.912195122
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=5 | accuracy=0.880000000
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.912195122
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.785575049
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=1 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=1 | accuracy=0.897660819
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.799220273
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=2 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.788499025
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=3 | accuracy=0.892787524
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.769980507
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.766829268
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=5 | accuracy=0.913170732
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=5 | accuracy=0.881951220
Fold metric | dataset=Joint corpus | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.915121951
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.786441497
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.923278531
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.894065516
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.922823690

Feature-set condition | dataset=Joint corpus | feature_set=+ Readability (READ) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.782651072
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=1 | accuracy=0.913255361
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.774853801
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=2 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=2 | accuracy=0.871345029
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.789473684
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=3 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.800194932
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=4 | accuracy=0.932748538
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.759024390
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=5 | accuracy=0.921951220
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=5 | accuracy=0.873170732
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.925853659
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.779727096
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=1 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.786549708
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.825536062
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=3 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.785575049
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=4 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=4 | accuracy=0.888888889
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.763902439
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=5 | accuracy=0.922926829
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=5 | accuracy=0.887804878
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.920975610
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.775828460
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=1 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.768031189
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=2 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=2 | accuracy=0.892787524
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.784600390
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=3 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=3 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.797270955
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=4 | accuracy=0.892787524
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.791219512
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=5 | accuracy=0.919024390
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=5 | accuracy=0.907317073
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.928780488
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.785575049
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=1 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=1 | accuracy=0.908382066
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.796296296
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=2 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.774853801
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=3 | accuracy=0.934697856
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=3 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.768031189
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=4 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=4 | accuracy=0.869395712
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.805853659
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=5 | accuracy=0.916097561
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=5 | accuracy=0.887804878
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.914146341
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.817738791
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.944444444
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.774853801
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.778752437
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=3 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=3 | accuracy=0.880116959
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.789473684
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=4 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=4 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.775609756
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=5 | accuracy=0.912195122
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=5 | accuracy=0.884878049
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.911219512
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.784600390
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=1 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.798245614
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.787524366
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=3 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=3 | accuracy=0.892787524
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.763157895
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=4 | accuracy=0.904483431
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.938596491
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.763902439
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=5 | accuracy=0.908292683
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=5 | accuracy=0.875121951
Fold metric | dataset=Joint corpus | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.915121951
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.784296930
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.922498391
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.893869919
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.922596364

Feature-set condition | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.776803119
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=1 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=1 | accuracy=0.899610136
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.770955166
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=2 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=2 | accuracy=0.863547758
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.790448343
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=3 | accuracy=0.908382066
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.795321637
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=4 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.759024390
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=5 | accuracy=0.922926829
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=5 | accuracy=0.873170732
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.927804878
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.772904483
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=1 | accuracy=0.879142300
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.778752437
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=2 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.819688109
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=3 | accuracy=0.893762183
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.779727096
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=4 | accuracy=0.897660819
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.760975610
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=5 | accuracy=0.924878049
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=5 | accuracy=0.893658537
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.918048780
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.770955166
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=1 | accuracy=0.882066277
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.763157895
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=2 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.779727096
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=3 | accuracy=0.903508772
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.787524366
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=4 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=4 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.783414634
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=5 | accuracy=0.922926829
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=5 | accuracy=0.902439024
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.926829268
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.785575049
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=1 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=1 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.791423002
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=2 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=2 | accuracy=0.901559454
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.768031189
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=3 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.768031189
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=4 | accuracy=0.907407407
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=4 | accuracy=0.877192982
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.803902439
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=5 | accuracy=0.915121951
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=5 | accuracy=0.889756098
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.912195122
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.813840156
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=1 | accuracy=0.907407407
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.946393762
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.773879142
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=2 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=2 | accuracy=0.901559454
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.774853801
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=3 | accuracy=0.915204678
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=3 | accuracy=0.878167641
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.915204678
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.787524366
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=4 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=4 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.771707317
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=5 | accuracy=0.915121951
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=5 | accuracy=0.879024390
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.910243902
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.780701754
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=1 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.795321637
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.782651072
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=3 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.756335283
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=4 | accuracy=0.936647173
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=4 | accuracy=0.913255361
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.938596491
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.760975610
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=5 | accuracy=0.914146341
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=5 | accuracy=0.879024390
Fold metric | dataset=Joint corpus | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.919024390
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.780137752
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.924058321
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.893610041
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.923700914

Feature-set condition | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.768031189
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=1 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=1 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.761208577
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=2 | accuracy=0.870370370
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.782651072
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=3 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=3 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.784600390
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=4 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.750243902
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=5 | accuracy=0.923902439
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=5 | accuracy=0.881951220
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.927804878
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.767056530
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=1 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=1 | accuracy=0.889863548
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.771929825
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=2 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.810916179
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=3 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.766081871
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=4 | accuracy=0.887914230
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.747317073
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=5 | accuracy=0.924878049
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=5 | accuracy=0.886829268
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.918048780
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.762183236
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=1 | accuracy=0.899610136
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.757309942
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=2 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=2 | accuracy=0.887914230
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.768031189
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=3 | accuracy=0.901559454
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.778752437
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=4 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=4 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.772682927
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=5 | accuracy=0.922926829
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=5 | accuracy=0.897560976
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.926829268
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.774853801
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=1 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.787524366
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=2 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=2 | accuracy=0.893762183
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.759259259
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=3 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.755360624
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=4 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=4 | accuracy=0.869395712
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.790243902
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=5 | accuracy=0.915121951
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=5 | accuracy=0.890731707
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.912195122
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.802144250
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=1 | accuracy=0.946393762
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.766081871
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=2 | accuracy=0.903508772
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.768031189
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=3 | accuracy=0.915204678
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.781676413
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=4 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=4 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.761951220
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=5 | accuracy=0.915121951
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=5 | accuracy=0.880975610
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.910243902
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.773879142
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=1 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.787524366
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=2 | accuracy=0.887914230
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.772904483
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=3 | accuracy=0.888888889
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.745614035
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=4 | accuracy=0.936647173
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=4 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.747317073
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=5 | accuracy=0.915121951
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=5 | accuracy=0.875121951
Fold metric | dataset=Joint corpus | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.919024390
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.770778745
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.924188339
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.891952995
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.923570960

Feature-set condition | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.858674464
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=1 | accuracy=0.898635478
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.836257310
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=2 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=2 | accuracy=0.856725146
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.861598441
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=3 | accuracy=0.898635478
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.860623782
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=4 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.816585366
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=1 | fold=5 | accuracy=0.927804878
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=1 | fold=5 | accuracy=0.873170732
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.932682927
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.841130604
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=1 | accuracy=0.896686160
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.850877193
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=2 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.876218324
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=3 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.835282651
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=4 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.832195122
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=2 | fold=5 | accuracy=0.920000000
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=2 | fold=5 | accuracy=0.893658537
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.919024390
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.837231969
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=1 | accuracy=0.883040936
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.838206628
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.865497076
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=3 | accuracy=0.896686160
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.851851852
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=4 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=4 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.847804878
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=3 | fold=5 | accuracy=0.915121951
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=3 | fold=5 | accuracy=0.896585366
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.925853659
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.849902534
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.854775828
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=2 | accuracy=0.884990253
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.837231969
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=3 | accuracy=0.899610136
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.936647173
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.846978558
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=4 | accuracy=0.882066277
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.849756098
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=4 | fold=5 | accuracy=0.920000000
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=4 | fold=5 | accuracy=0.894634146
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.919024390
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.864522417
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=1 | accuracy=0.946393762
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=1 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.948343080
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.849902534
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=2 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=2 | accuracy=0.899610136
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.843079922
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=3 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=3 | accuracy=0.876218324
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.841130604
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=4 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=4 | accuracy=0.885964912
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.843902439
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=5 | fold=5 | accuracy=0.910243902
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=5 | fold=5 | accuracy=0.880975610
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.908292683
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.845029240
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=1 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.872319688
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=2 | accuracy=0.896686160
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.848927875
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=3 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=3 | accuracy=0.889863548
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.837231969
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=4 | accuracy=0.931773879
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=4 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.824390244
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=LR | repeat=6 | fold=5 | accuracy=0.919024390
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=RF | repeat=6 | fold=5 | accuracy=0.870243902
Fold metric | dataset=Joint corpus | feature_set=+ Traditional features (TRAD) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.921951220
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.847303919
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.923733340
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.891433050
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.923603861

Feature-set condition | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.862573099
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=1 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.839181287
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=2 | accuracy=0.870370370
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.868421053
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=3 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=3 | accuracy=0.907407407
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.865497076
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.824390244
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=1 | fold=5 | accuracy=0.926829268
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=1 | fold=5 | accuracy=0.870243902
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.932682927
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.846003899
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=1 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.856725146
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=2 | accuracy=0.889863548
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.884015595
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=3 | accuracy=0.897660819
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.843079922
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=4 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.834146341
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=2 | fold=5 | accuracy=0.920975610
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=2 | fold=5 | accuracy=0.888780488
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.919024390
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.846003899
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=1 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.842105263
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=2 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=2 | accuracy=0.885964912
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.870370370
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=3 | accuracy=0.903508772
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.853801170
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=4 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=4 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.847804878
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=3 | fold=5 | accuracy=0.915121951
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=3 | fold=5 | accuracy=0.894634146
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.924878049
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.858674464
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.854775828
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=2 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=2 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.846978558
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=3 | accuracy=0.938596491
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=3 | accuracy=0.901559454
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.938596491
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.847953216
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=4 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=4 | accuracy=0.871345029
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.915204678
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.853658537
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=4 | fold=5 | accuracy=0.920000000
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=4 | fold=5 | accuracy=0.898536585
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.918048780
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.867446394
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=1 | accuracy=0.946393762
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=1 | accuracy=0.908382066
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.948343080
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.858674464
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=2 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.846978558
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=3 | accuracy=0.878167641
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.846978558
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=4 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.841951220
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=5 | fold=5 | accuracy=0.911219512
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=5 | fold=5 | accuracy=0.877073171
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.908292683
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.854775828
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=1 | accuracy=0.892787524
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.873294347
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.857699805
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=3 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=3 | accuracy=0.897660819
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.847953216
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=4 | accuracy=0.931773879
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=4 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.826341463
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=LR | repeat=6 | fold=5 | accuracy=0.918048780
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=RF | repeat=6 | fold=5 | accuracy=0.875121951
Fold metric | dataset=Joint corpus | feature_set=+ Syllabic features (SYLL) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.921951220
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.852275123
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.923668362
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.891790266
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.923636286

Feature-set condition | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.864522417
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=1 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.841130604
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=2 | accuracy=0.871345029
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=3 | accuracy=0.900584795
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.868421053
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=4 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=4 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.831219512
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=1 | fold=5 | accuracy=0.926829268
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=1 | fold=5 | accuracy=0.883902439
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.925853659
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.850877193
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=1 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=1 | accuracy=0.887914230
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.860623782
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=2 | accuracy=0.885964912
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.883040936
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=3 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.849902534
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=4 | accuracy=0.885964912
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.840000000
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=2 | fold=5 | accuracy=0.920000000
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=2 | fold=5 | accuracy=0.890731707
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.920975610
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.845029240
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=1 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=1 | accuracy=0.877192982
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.848927875
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=2 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=3 | accuracy=0.898635478
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.858674464
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=4 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=4 | accuracy=0.898635478
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.848780488
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=3 | fold=5 | accuracy=0.910243902
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=3 | fold=5 | accuracy=0.894634146
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.920975610
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.860623782
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=1 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.864522417
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=2 | accuracy=0.884990253
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.850877193
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=3 | accuracy=0.940545809
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=3 | accuracy=0.893762183
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.851851852
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=4 | accuracy=0.904483431
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=4 | accuracy=0.866471735
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.913255361
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.859512195
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=4 | fold=5 | accuracy=0.919024390
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=4 | fold=5 | accuracy=0.901463415
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.918048780
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.875243665
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.948343080
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.859649123
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=2 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=2 | accuracy=0.902534113
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.852826511
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=3 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=3 | accuracy=0.868421053
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.849902534
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=4 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=4 | accuracy=0.896686160
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.913255361
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.843902439
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=5 | fold=5 | accuracy=0.911219512
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=5 | fold=5 | accuracy=0.876097561
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.908292683
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.862573099
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=1 | accuracy=0.922027290
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=1 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.879142300
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=2 | accuracy=0.893762183
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.860623782
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=3 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=3 | accuracy=0.888888889
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.850877193
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=4 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.936647173
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.823414634
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=LR | repeat=6 | fold=5 | accuracy=0.916097561
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=RF | repeat=6 | fold=5 | accuracy=0.878048780
Fold metric | dataset=Joint corpus | feature_set=+ Lexical features (LEX) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.920975610
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.856369293
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.923310702
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.890718807
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.923213617

Feature-set condition | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=1 | accuracy=0.864522417
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=1 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=1 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=2 | accuracy=0.841130604
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=2 | accuracy=0.902534113
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=2 | accuracy=0.872319688
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=3 | accuracy=0.936647173
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=3 | accuracy=0.907407407
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=3 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=4 | accuracy=0.868421053
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=4 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=4 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=4 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=1 | fold=5 | accuracy=0.831219512
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=1 | fold=5 | accuracy=0.924878049
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=1 | fold=5 | accuracy=0.868292683
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=1 | fold=5 | accuracy=0.929756098
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=1 | accuracy=0.850877193
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=1 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=1 | accuracy=0.884990253
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=2 | accuracy=0.860623782
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=2 | accuracy=0.891812865
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=2 | accuracy=0.919103314
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=3 | accuracy=0.883040936
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=3 | accuracy=0.928849903
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=3 | accuracy=0.895711501
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=4 | accuracy=0.849902534
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=4 | accuracy=0.885964912
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=2 | fold=5 | accuracy=0.840000000
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=2 | fold=5 | accuracy=0.920975610
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=2 | fold=5 | accuracy=0.889756098
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=2 | fold=5 | accuracy=0.920975610
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=1 | accuracy=0.845029240
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=1 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=1 | accuracy=0.882066277
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=1 | accuracy=0.921052632
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=2 | accuracy=0.848927875
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=2 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=3 | accuracy=0.877192982
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=3 | accuracy=0.930799220
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=3 | accuracy=0.898635478
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=3 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=4 | accuracy=0.858674464
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=4 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=3 | fold=5 | accuracy=0.848780488
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=3 | fold=5 | accuracy=0.911219512
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=3 | fold=5 | accuracy=0.902439024
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=3 | fold=5 | accuracy=0.921951220
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=1 | accuracy=0.860623782
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=1 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=1 | accuracy=0.903508772
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=2 | accuracy=0.863547758
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=2 | accuracy=0.923001949
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=2 | accuracy=0.884015595
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=2 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=3 | accuracy=0.850877193
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=3 | accuracy=0.940545809
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=3 | accuracy=0.897660819
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=4 | accuracy=0.851851852
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=4 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=4 | accuracy=0.869395712
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=4 | accuracy=0.911306043
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=4 | fold=5 | accuracy=0.859512195
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=4 | fold=5 | accuracy=0.922926829
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=4 | fold=5 | accuracy=0.886829268
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=4 | fold=5 | accuracy=0.918048780
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=1 | accuracy=0.875243665
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=1 | accuracy=0.946393762
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=1 | accuracy=0.909356725
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=2 | accuracy=0.859649123
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=2 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=2 | accuracy=0.905458090
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=2 | accuracy=0.924951267
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=3 | accuracy=0.852826511
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=3 | accuracy=0.916179337
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=3 | accuracy=0.885964912
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=3 | accuracy=0.915204678
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=4 | accuracy=0.849902534
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=4 | accuracy=0.913255361
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=4 | accuracy=0.886939571
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=4 | accuracy=0.917153996
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=5 | fold=5 | accuracy=0.843902439
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=5 | fold=5 | accuracy=0.911219512
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=5 | fold=5 | accuracy=0.886829268
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=5 | fold=5 | accuracy=0.908292683
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=1 | accuracy=0.862573099
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=1 | accuracy=0.884990253
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=1 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=2 | accuracy=0.879142300
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=2 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=2 | accuracy=0.926900585
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=3 | accuracy=0.860623782
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=3 | accuracy=0.920077973
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=3 | accuracy=0.890838207
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=3 | accuracy=0.908382066
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=4 | accuracy=0.850877193
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=4 | accuracy=0.934697856
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=4 | accuracy=0.901559454
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=MNB | repeat=6 | fold=5 | accuracy=0.823414634
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=LR | repeat=6 | fold=5 | accuracy=0.919024390
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=RF | repeat=6 | fold=5 | accuracy=0.867317073
Fold metric | dataset=Joint corpus | feature_set=+ Morphological features (MORPH) [full set] | classifier=SVC | repeat=6 | fold=5 | accuracy=0.919024390
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.856336804
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.923083504
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.890523114
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.922726382

Completed tuned feature ablation at 2026-06-04 11:53:28
Total fold metrics: 2880
```
