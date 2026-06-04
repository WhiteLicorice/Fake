# Full Re-run Results After Stopwords Fix

## Fix Verification

Step 0 regenerated only `SwFeatures.csv` for Cruz/FNF2020 and Lupac/FNF2024. Both regenerated files matched the original CSVs exactly: `Changed SW rows after regeneration: 0`; `Unified diff contains changes: False`; SW was distinct from OOV in both datasets.

### Step 0 Side-by-side Samples

Cruz/FNF2020:
| row_index | count_stopwords | count_oov_words |
| --- | --- | --- |
| 0 | 117 | 20 |
| 1 | 60 | 7 |
| 2 | 11 | 2 |
| 3 | 28 | 2 |
| 4 | 9 | 2 |
| 5 | 17 | 7 |
| 6 | 14 | 4 |
| 7 | 30 | 1 |
| 8 | 20 | 2 |
| 9 | 85 | 5 |

Lupac/FNF2024:
| row_index | count_stopwords | count_oov_words |
| --- | --- | --- |
| 0 | 132 | 17 |
| 1 | 259 | 43 |
| 2 | 130 | 16 |
| 3 | 154 | 29 |
| 4 | 209 | 29 |
| 5 | 46 | 12 |
| 6 | 138 | 30 |
| 7 | 151 | 21 |
| 8 | 93 | 24 |
| 9 | 46 | 11 |

Step 1 verified `StopWordsExtractor(from_csv=True)` matches `csv_count_stopwords` and not `csv_count_oov_words` for the same samples in both datasets.

## Descriptive Statistics (replaces P10, L202-L208)

| Dataset | mean_oov | mean_readability | mean_stopwords |
| --- | --- | --- | --- |
| Fake News Filipino 2020 | 17.601996 | 20.514714 | 77.796319 |
| Fake News Filipino 2024 | 22.217093 | 19.664522 | 74.547723 |

## Mann-Whitney U Tests (replaces P13, L259-L263)

| Feature | U | p | Bonferroni p | rank-biserial r |
| --- | --- | --- | --- | --- |
| OOV count | 5814710.50 | 7.708579e-20 | 2.312574e-19 | 0.131439 |
| Readability index | 4745072.00 | 1.048735e-07 | 3.146206e-07 | -0.076694 |
| Stop word count | 4759872.00 | 3.079568e-07 | 9.238703e-07 | -0.073814 |

## Table 2 replacement (no tuning, joint corpus)

| Classifier | Params | Class | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| MNB |  | Fake | 1.000 | 0.162 | 0.279 | 0.581 |
| MNB |  | Real | 0.544 | 1.000 | 0.704 | 0.581 |
| LR |  | Fake | 0.923 | 0.935 | 0.929 | 0.928 |
| LR |  | Real | 0.934 | 0.922 | 0.928 | 0.928 |
| RF |  | Fake | 0.922 | 0.903 | 0.913 | 0.913 |
| RF |  | Real | 0.905 | 0.924 | 0.914 | 0.913 |
| SVC |  | Fake | 0.872 | 0.754 | 0.809 | 0.822 |
| SVC |  | Real | 0.783 | 0.889 | 0.833 | 0.822 |

## Table 3 replacement (tuned, joint corpus)

| Classifier | Best params | Best CV accuracy | Differs from previous prompt value |
| --- | --- | --- | --- |
| MNB | {"alpha": 0.1} | 0.857864 | No |
| LR | {"C": 1.0} | 0.923375 | No |
| RF | {"max_depth": 20, "min_samples_split": 2, "n_estimators": 100} | 0.890426 | Yes |
| SVC | {"C": 0.1, "kernel": "linear"} | 0.923960 | No |

| Classifier | Params | Class | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| MNB | {"alpha": 0.1} | Fake | 0.964 | 0.751 | 0.844 | 0.861 |
| MNB | {"alpha": 0.1} | Real | 0.796 | 0.972 | 0.875 | 0.861 |
| LR | {"C": 1.0} | Fake | 0.923 | 0.935 | 0.929 | 0.928 |
| LR | {"C": 1.0} | Real | 0.934 | 0.922 | 0.928 | 0.928 |
| RF | {"max_depth": 20, "min_samples_split": 2, "n_estimators": 100} | Fake | 0.894 | 0.889 | 0.891 | 0.892 |
| RF | {"max_depth": 20, "min_samples_split": 2, "n_estimators": 100} | Real | 0.890 | 0.894 | 0.892 | 0.892 |
| SVC | {"C": 0.1, "kernel": "linear"} | Fake | 0.919 | 0.936 | 0.927 | 0.927 |
| SVC | {"C": 0.1, "kernel": "linear"} | Real | 0.935 | 0.917 | 0.926 | 0.927 |

## Table 4 replacement (confusion matrices, no tuning)

| Classifier | Params | Fake->Fake | Fake->Real | Real->Fake | Real->Real |
| --- | --- | --- | --- | --- | --- |
| MNB |  | 104 | 538 | 0 | 641 |
| LR |  | 600 | 42 | 50 | 591 |
| RF |  | 580 | 62 | 49 | 592 |
| SVC |  | 484 | 158 | 71 | 570 |

## Table 5 replacement (confusion matrices, tuned)

| Classifier | Params | Fake->Fake | Fake->Real | Real->Fake | Real->Real |
| --- | --- | --- | --- | --- | --- |
| MNB | {"alpha": 0.1} | 482 | 160 | 18 | 623 |
| LR | {"C": 1.0} | 600 | 42 | 50 | 591 |
| RF | {"max_depth": 20, "min_samples_split": 2, "n_estimators": 100} | 571 | 71 | 68 | 573 |
| SVC | {"C": 0.1, "kernel": "linear"} | 601 | 41 | 53 | 588 |

## Table 6 replacement (30-run accuracies across datasets)

| Dataset | Classifier | Mean accuracy | SD |
| --- | --- | --- | --- |
| Fake News Filipino 2020 | LR | 0.952288 | 0.009114 |
| Fake News Filipino 2020 | MNB | 0.920504 | 0.011384 |
| Fake News Filipino 2020 | RF | 0.919334 | 0.012154 |
| Fake News Filipino 2020 | SVC | 0.951898 | 0.009530 |
| Fake News Filipino 2024 | LR | 0.945462 | 0.010354 |
| Fake News Filipino 2024 | MNB | 0.881371 | 0.015605 |
| Fake News Filipino 2024 | RF | 0.925768 | 0.010687 |
| Fake News Filipino 2024 | SVC | 0.947022 | 0.009949 |
| Joint corpus | LR | 0.923084 | 0.009553 |
| Joint corpus | MNB | 0.856337 | 0.013968 |
| Joint corpus | RF | 0.890523 | 0.011704 |
| Joint corpus | SVC | 0.922726 | 0.008679 |

## ANOVA replacement (replaces P12-P13, L246-L271)

### Shapiro-Wilk
| dataset | classifier | W | p |
| --- | --- | --- | --- |
| Fake News Filipino 2020 | MNB | 0.960576 | 0.320437 |
| Fake News Filipino 2020 | LR | 0.973468 | 0.637729 |
| Fake News Filipino 2020 | RF | 0.976488 | 0.726482 |
| Fake News Filipino 2020 | SVC | 0.971863 | 0.591321 |
| Fake News Filipino 2024 | MNB | 0.941648 | 0.100748 |
| Fake News Filipino 2024 | LR | 0.962666 | 0.361719 |
| Fake News Filipino 2024 | RF | 0.978086 | 0.772686 |
| Fake News Filipino 2024 | SVC | 0.979374 | 0.808671 |
| Joint corpus | MNB | 0.972845 | 0.619572 |
| Joint corpus | LR | 0.982882 | 0.895883 |
| Joint corpus | RF | 0.956926 | 0.257988 |
| Joint corpus | SVC | 0.948676 | 0.155791 |

### Levene
| W | p |
| --- | --- |
| 2.096062 | 0.020043 |

### Two-way ANOVA for Trimmed Means
| effect | statistic | p_value |
| --- | --- | --- |
| dataset | 721.356750 | 0.001000 |
| classifier | 1142.943020 | 0.001000 |
| dataset * classifier | 124.354493 | 0.001000 |

### Standard Two-way ANOVA Partial Eta Squared
| effect | F | p_value | partial_eta_squared |
| --- | --- | --- | --- |
| dataset | 359.880231 | 0.000000 | 0.674084 |
| classifier | 487.594495 | 0.000000 | 0.807818 |
| dataset * classifier | 28.735307 | 0.000000 | 0.331299 |

### Bonferroni: Classifier Pairs Within Dataset
| dataset | group_a | group_b | p_raw | p_bonferroni |
| --- | --- | --- | --- | --- |
| Fake News Filipino 2020 |  |  |  |  |
| Fake News Filipino 2020 |  |  |  |  |
| Fake News Filipino 2020 |  |  |  |  |
| Fake News Filipino 2020 |  |  |  |  |
| Fake News Filipino 2020 |  |  |  |  |
| Fake News Filipino 2020 |  |  |  |  |
| Fake News Filipino 2024 |  |  |  |  |
| Fake News Filipino 2024 |  |  |  |  |
| Fake News Filipino 2024 |  |  |  |  |
| Fake News Filipino 2024 |  |  |  |  |
| Fake News Filipino 2024 |  |  |  |  |
| Fake News Filipino 2024 |  |  |  |  |
| Joint corpus |  |  |  |  |
| Joint corpus |  |  |  |  |
| Joint corpus |  |  |  |  |
| Joint corpus |  |  |  |  |
| Joint corpus |  |  |  |  |
| Joint corpus |  |  |  |  |

### Bonferroni: Dataset Pairs Within Classifier
| classifier | group_a | group_b | p_raw | p_bonferroni |
| --- | --- | --- | --- | --- |
| MNB |  |  |  |  |
| MNB |  |  |  |  |
| MNB |  |  |  |  |
| LR |  |  |  |  |
| LR |  |  |  |  |
| LR |  |  |  |  |
| RF |  |  |  |  |
| RF |  |  |  |  |
| RF |  |  |  |  |
| SVC |  |  |  |  |
| SVC |  |  |  |  |
| SVC |  |  |  |  |

## Table 12 replacement (ROC-AUC)

| Classifier | Mean AUC | SD |
| --- | --- | --- |
| LR | 0.975503 | 0.004597 |
| MNB | 0.921877 | 0.009800 |
| RF | 0.960274 | 0.005898 |
| SVC | 0.973595 | 0.005171 |

## Table 9 replacement (linguistic feature coefficients)

| Feature | Coefficient |
| --- | --- |
| morph__prefix_derived_ratio | -0.465590 |
| lex__log_ttr | -0.402674 |
| syll__cvc_density | -0.395495 |
| lex__root_ttr | -0.337148 |
| trad__ave_phrase_count | -0.324248 |
| trad__ave_word_length | -0.298715 |
| lex__ttr | -0.271151 |
| syll__cvcc_density | -0.260007 |
| lex__corr_ttr | -0.238400 |
| syll__cv_density | -0.211268 |
| syll__consonant_cluster | -0.184323 |
| syll__v_density | -0.118809 |
| trad__ave_syllable_count_of_word | -0.118809 |
| syll__vc_density | -0.111141 |
| lex__compound_tr | -0.109257 |
| syll__vcc_density | -0.102489 |
| morph__total_affix_derived_ratio | -0.102378 |
| morph__participle_verb_ratio | -0.096633 |
| morph__perfective_verb_ratio | -0.093181 |
| sw__count_stopwords | -0.063949 |
| trad__word_count_per_sentence | -0.063489 |
| trad__polysyll_count | -0.052793 |
| lex__foreign_tr | -0.037724 |
| morph__total_affix_token_ratio | -0.027490 |
| morph__contemplative_verb_ratio | -0.025192 |
| morph__suffix_token_ratio | -0.017891 |
| morph__prefix_token_ratio | -0.009599 |
| morph__object_focus_ratio | -0.007816 |
| morph__referential_focus_ratio | -0.003299 |
| syll__ccvcc_density | -0.002273 |
| morph__aux_verb_ratio | -0.001999 |
| morph__locative_focus_ratio | -0.000660 |
| morph__recent_past_verb_ratio | 0.000000 |
| morph__instrumental_focus_ratio | 0.000000 |
| morph__benefactive_focus_ratio | 0.002848 |
| trad__word_count | 0.014996 |
| syll__ccvccc_density | 0.018048 |
| morph__imperfective_verb_ratio | 0.021741 |
| lex__verb_tr | 0.036398 |
| oov__count_oov_words | 0.045380 |
| trad__sentence_count | 0.061279 |
| morph__infinitive_verb_ratio | 0.072211 |
| morph__actor_focus_ratio | 0.097601 |
| read__readability_score | 0.151202 |
| morph__suffix_derived_ratio | 0.363213 |
| lex__noun_tr | 0.465511 |
| lex__lexical_density | 0.500297 |

## Table 10 replacement (top vectorizer predictors)

| Feature | Coefficient |
| --- | --- |
| vectorizers__bow__upang | -0.959123 |
| vectorizers__bow__ngunit | -0.905210 |
| vectorizers__bow__noong | -0.773495 |
| vectorizers__bow__sinabi | -0.769751 |
| vectorizers__bow__dakong | -0.761375 |
| vectorizers__bow__nagsasabing | -0.758867 |
| vectorizers__bow__ferdinand | -0.680611 |
| vectorizers__bow__idinagdag | -0.655430 |
| vectorizers__bow__enero | -0.645895 |
| vectorizers__bow__kumakalat | -0.615180 |
| vectorizers__bow__source | 2.487863 |
| vectorizers__bow__below | 1.411136 |
| vectorizers__bow__gma | 1.339239 |
| vectorizers__bow__philippines | 1.252139 |
| vectorizers__bow__frj | 1.167990 |
| vectorizers__bow__news | 1.049512 |
| vectorizers__bow__ofw | 1.011598 |
| vectorizers__bow__panoorin | 0.976942 |
| vectorizers__bow__kaniyang | 0.946027 |
| vectorizers__bow__ptv | 0.927673 |

## Tables 13-14 replacement (misclassification analysis)

Classification outcomes:
| article_id | gold_label | predicted_label | previous_outcome | new_outcome | changed |
| --- | --- | --- | --- | --- | --- |
| false_positive_article | 0 | 0 | FP | TP | True |
| true_positive_article | 0 | 0 | TP | TP | False |
| false_negative_article | 1 | 0 | FN | FN | False |
| true_negative_article | 1 | 1 | TN | TN | False |

Linguistic feature values:
| article_id | word_count | sentence_count | polysyll_count | ave_word_length | ave_phrase_count | ave_syllable_count_of_word | word_count_per_sentence | consonant_cluster | v_density | cv_density | vc_density | cvc_density | vcc_density | cvcc_density | ccvcc_density | ccvccc_density | count_oov_words | count_stopwords | readability_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| false_positive_article | 129 | 8 | 4 | 5.186047 | 2.125000 | 2.131783 | 16.125000 | 0.596899 | 2.131783 | 1.806202 | 1.596899 | 0.930233 | 0.480620 | 0.348837 | 0.062016 | 0.000000 | 16 | 44 | 14.028000 |
| true_positive_article | 132 | 9 | 5 | 5.371212 | 3.000000 | 2.098485 | 14.666667 | 0.719697 | 2.098485 | 1.795455 | 1.659091 | 1.022727 | 0.553030 | 0.431818 | 0.075758 | 0.015152 | 24 | 40 | 13.937667 |
| false_negative_article | 110 | 5 | 6 | 5.472727 | 1.600000 | 2.118182 | 22.000000 | 0.881818 | 2.118182 | 1.763636 | 1.554545 | 0.972727 | 0.718182 | 0.490909 | 0.136364 | 0.009091 | 8 | 47 | 13.659000 |
| true_negative_article | 142 | 9 | 5 | 4.845070 | 1.777778 | 1.978873 | 15.777778 | 0.542254 | 1.978873 | 1.640845 | 1.577465 | 0.908451 | 0.443662 | 0.295775 | 0.028169 | 0.000000 | 18 | 56 | 15.308778 |

Vectorizer predictors:
| article_id | direction | feature | coefficient |
| --- | --- | --- | --- |
| false_positive_article | lowest | bow__supt | -0.518008 |
| false_positive_article | lowest | bow__iba | -0.463612 |
| false_positive_article | lowest | bow__director | -0.358003 |
| false_positive_article | lowest | bow__senior | -0.307914 |
| false_positive_article | highest | bow__ulat | 0.338630 |
| false_positive_article | highest | bow__pumasok | 0.359647 |
| false_positive_article | highest | bow__reklamo | 0.387851 |
| false_positive_article | highest | bow__office | 0.540150 |
| true_positive_article | lowest | bow__noong | -0.785753 |
| true_positive_article | lowest | bow__supt | -0.518008 |
| true_positive_article | lowest | bow__iba | -0.463612 |
| true_positive_article | lowest | bow__sina | -0.360117 |
| true_positive_article | highest | bow__suspek | 0.342199 |
| true_positive_article | highest | bow__madaling | 0.410671 |
| true_positive_article | highest | bow__driver | 0.415596 |
| true_positive_article | highest | bow__office | 0.540150 |
| false_negative_article | lowest | bow__noong | -0.785753 |
| false_negative_article | lowest | bow__board | -0.389138 |
| false_negative_article | lowest | bow__about | -0.350795 |
| false_negative_article | lowest | bow__kanyang | -0.305296 |
| false_negative_article | highest | bow__taon | 0.313104 |
| false_negative_article | highest | bow__para | 0.317631 |
| false_negative_article | highest | bow__pagbibigay | 0.399480 |
| false_negative_article | highest | bow__2018 | 0.450075 |
| true_negative_article | lowest | bow__mayo | -0.341020 |
| true_negative_article | lowest | bow__de | -0.288649 |
| true_negative_article | lowest | bow__umaga | -0.276231 |
| true_negative_article | lowest | bow__cbn | -0.272249 |
| true_negative_article | highest | bow__patay | 0.429375 |
| true_negative_article | highest | bow__anyos | 0.502501 |
| true_negative_article | highest | bow__kaniyang | 0.932441 |
| true_negative_article | highest | bow__news | 1.113835 |

## Table 8 replacement (inference benchmark)

| Article length | Mean ms | Median ms | SD ms |
| --- | --- | --- | --- |
| Short article (~50 words) | 16.391 | 16.587 | 0.552 |
| Medium article (~100 words) | 17.746 | 17.494 | 0.568 |
| Long article (~200 words) | 17.824 | 17.882 | 0.288 |
| Overall | 17.320 | 17.475 | 0.817 |

## Deployment model

| CV mean accuracy | CV SD | Model file size bytes | Model file size MB | Parameter count |
| --- | --- | --- | --- | --- |
| 0.923668 | 0.008531 | 69243399 | 66.036 | 1451808 |

## Table 7 replacement (tuned ablation, ALL three datasets)

### Fake News Filipino 2020

| Feature Set | MNB | LR | RF | SVC |
| --- | --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- |
| Vectorizers (TF-IDF + BOW) | 0.7864 | 0.9233 | 0.8941 | 0.9228 |
| + Readability (READ) | 0.7843 | 0.9225 | 0.8939 | 0.9226 |
| + Out-of-vocabulary (OOV) | 0.7801 | 0.9241 | 0.8936 | 0.9237 |
| + Stop words (SW) | 0.7708 | 0.9242 | 0.8920 | 0.9236 |
| + Traditional features (TRAD) | 0.8473 | 0.9237 | 0.8914 | 0.9236 |
| + Syllabic features (SYLL) | 0.8523 | 0.9237 | 0.8918 | 0.9236 |
| + Lexical features (LEX) | 0.8564 | 0.9233 | 0.8907 | 0.9232 |
| + Morphological features (MORPH) [full set] | 0.8563 | 0.9231 | 0.8905 | 0.9227 |

## Changes summary

- Step 0, Cruz SwFeatures.csv: diff present -> no diff; changed rows 0.
- Step 0, Lupac SwFeatures.csv: diff present -> no diff; changed rows 0.
- Server `StopWordsExtractor.from_csv`: `count_oov_words` source -> `count_stopwords` source.
- Table 2, MNB, confusion matrix Fake->Real: 537 -> 538.
- Table 2, LR, confusion matrix Real->Fake: 51 -> 50.
- Table 2, RF, confusion matrix Fake->Real: 66 -> 62; Real->Fake: 59 -> 49; accuracy approximately 0.90 -> 0.913.
- Table 2, SVC, confusion matrix Fake->Real: 169 -> 158; Real->Fake: 70 -> 71; accuracy approximately 0.81 -> 0.822.
- Table 3, MNB, best alpha in manuscript text: 0.01 -> 0.1; holdout accuracy 0.865 -> 0.861.
- Table 3, RF, best params: max_depth=20 -> max_depth=20, min_samples_split=2, n_estimators=100; holdout accuracy 0.889 -> 0.892.
- Table 3, SVC, confusion matrix Fake->Real: 42 -> 41; Real->Fake: 51 -> 53; holdout accuracy 0.928 -> 0.927.
- Table 6, LR, FNF2020 accuracy: 0.951 -> 0.952288.
- Table 6, LR, FNF2024 accuracy: 0.947 -> 0.945462.
- Table 6, LR, Combined accuracy: 0.924 -> 0.923084.
- Table 6, MNB, FNF2020 accuracy: 0.923 -> 0.920504.
- Table 6, MNB, FNF2024 accuracy: 0.885 -> 0.881371.
- Table 6, MNB, Combined accuracy: 0.860 -> 0.856337.
- Table 6, RF, Combined accuracy: 0.888 -> 0.890523.
- Table 6, SVC, Combined accuracy: 0.922 -> 0.922726.
- Table 12, MNB ROC-AUC: 0.925 +/- 0.010 -> 0.921877 +/- 0.009800.
- Table 12, LR ROC-AUC: 0.976 +/- 0.005 -> 0.975503 +/- 0.004597.
- Table 12, RF ROC-AUC: 0.960 +/- 0.006 -> 0.960274 +/- 0.005898.
- Table 12, SVC ROC-AUC: 0.973 +/- 0.005 -> 0.973595 +/- 0.005171.
- Table 9, count-stopwords coefficient: 0.025210022216103887 -> -0.063949.
- Table 9, count-oov-words coefficient: 0.025210022216103887 -> 0.045380.
- Table 13, false positive article outcome: FP -> TP.
- Table 13, false positive readability score: 14.673 -> 14.028.
- Table 13, false negative word_count: 109 -> 110; count_oov_words 10 -> 8.
- Table 8/Deployment model, CV accuracy: 0.92 -> 0.923668 +/- 0.008531.
- Deployment model size: 55.64 MB -> 66.036 MB.
- Deployment parameter count: 1,200,000 -> 1,451,808.
- Table 7, ablation protocol: default-hyperparameter joint-only table -> tuned-hyperparameter tables for FNF2020, FNF2024, and joint corpus.

## Raw console output

### training/results/stopwords_rerun/step0_recache_sw.log

```text
# Step 0: Re-cache SW Features
Started: 2026-06-03_22-02-31
Output directory: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31
Only SwFeatures.csv files will be regenerated.

=== Fake News Filipino 2020 (Cruz) ===
Article CSV: D:\Creative Corner\Projects\Software\Fake\training\root\datasets\Cruz\FakeNewsFilipino_Cruz2020.csv
SW CSV:      D:\Creative Corner\Projects\Software\Fake\training\root\datasets\Cruz\SwFeatures.csv
OOV CSV:     D:\Creative Corner\Projects\Software\Fake\training\root\datasets\Cruz\OovFeatures.csv
Rows processed: 3206
Changed SW rows after regeneration: 0
Regenerated SW matches original CSV: True
Regenerated SW column identical to OOV column: False
Unified diff contains changes: False
Before copy: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Cruz\SwFeatures.before.csv
After copy:  D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Cruz\SwFeatures.after.csv
Unified diff artifact: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Cruz\SwFeatures.before_vs_after.diff
Row diff artifact:     D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Cruz\SwFeatures.changed_rows.csv

First 10 rows, SW and OOV side-by-side:
 row_index  count_stopwords  count_oov_words
         0              117               20
         1               60                7
         2               11                2
         3               28                2
         4                9                2
         5               17                7
         6               14                4
         7               30                1
         8               20                2
         9               85                5

=== Fake News Filipino 2024 (Lupac) ===
Article CSV: D:\Creative Corner\Projects\Software\Fake\training\root\datasets\Lupac\FakeNewsPhilippines2024_Lupac.csv
SW CSV:      D:\Creative Corner\Projects\Software\Fake\training\root\datasets\Lupac\SwFeatures.csv
OOV CSV:     D:\Creative Corner\Projects\Software\Fake\training\root\datasets\Lupac\OovFeatures.csv
Rows processed: 3206
Changed SW rows after regeneration: 0
Regenerated SW matches original CSV: True
Regenerated SW column identical to OOV column: False
Unified diff contains changes: False
Before copy: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Lupac\SwFeatures.before.csv
After copy:  D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Lupac\SwFeatures.after.csv
Unified diff artifact: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Lupac\SwFeatures.before_vs_after.diff
Row diff artifact:     D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\2026-06-03_22-02-31\Lupac\SwFeatures.changed_rows.csv

First 10 rows, SW and OOV side-by-side:
 row_index  count_stopwords  count_oov_words
         0              132               17
         1              259               43
         2              130               16
         3              154               29
         4              209               29
         5               46               12
         6              138               30
         7              151               21
         8               93               24
         9               46               11

Latest output copy: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_rerun\latest
Step 0 completed.
```

### training/results/stopwords_rerun/step1_verify_sw_extractor.log

```text
# Step 1: Verify StopWordsExtractor(from_csv=True)

=== Fake News Filipino 2020 (Cruz) ===
 row_index  extractor_from_csv  csv_count_stopwords  csv_count_oov_words  matches_sw_csv  matches_oov_csv
         0                 117                  117                   20            True            False
         1                  60                   60                    7            True            False
         2                  11                   11                    2            True            False
         3                  28                   28                    2            True            False
         4                   9                    9                    2            True            False
         5                  17                   17                    7            True            False
         6                  14                   14                    4            True            False
         7                  30                   30                    1            True            False
         8                  20                   20                    2            True            False
         9                  85                   85                    5            True            False
All sample extractor outputs match SW CSV: True
All sample extractor outputs match OOV CSV: False

=== Fake News Filipino 2024 (Lupac) ===
 row_index  extractor_from_csv  csv_count_stopwords  csv_count_oov_words  matches_sw_csv  matches_oov_csv
         0                 132                  132                   17            True            False
         1                 259                  259                   43            True            False
         2                 130                  130                   16            True            False
         3                 154                  154                   29            True            False
         4                 209                  209                   29            True            False
         5                  46                   46                   12            True            False
         6                 138                  138                   30            True            False
         7                 151                  151                   21            True            False
         8                  93                   93                   24            True            False
         9                  46                   46                   11            True            False
All sample extractor outputs match SW CSV: True
All sample extractor outputs match OOV CSV: False

Step 1 completed.
```

### training/results/stopwords_fix_rerun/03_04_descriptives_mannwhitney.log

```text
# Steps 3-4: Descriptive Statistics and Mann-Whitney U Tests
Dataset comparison orientation: Fake News Filipino 2024 versus Fake News Filipino 2020
Bonferroni alpha for three planned tests: 0.0167

Descriptive statistics:
                dataset  mean_oov  mean_readability  mean_stopwords
Fake News Filipino 2020 17.601996         20.514714       77.796319
Fake News Filipino 2024 22.217093         19.664522       74.547723

Mann-Whitney U tests:
          feature         U            p  bonferroni_corrected_p  rank_biserial_r  significant_at_0.0167
        OOV count 5814710.5 7.708579e-20            2.312574e-19         0.131439                   True
Readability index 4745072.0 1.048735e-07            3.146206e-07        -0.076694                   True
  Stop word count 4759872.0 3.079568e-07            9.238703e-07        -0.073814                   True

Saved JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\descriptives_mannwhitney.json
```

### training/results/stopwords_fix_rerun/05_train_default.err.log

```text

```

### training/results/stopwords_fix_rerun/05_train_default.log

```text
# Step 5: Joint-Corpus Training Without Hyperparameter Tuning
Feature set: vectorizers + READ + OOV + SW + TRAD + SYLL + LEX + MORPH
Split: train_test_split(test_size=0.2, stratify=y, random_state=42)
Dataset: Joint corpus
Rows: total=6412, train=5129, test=1283

Training Model: Multinomial Naive Bayes (MNB)
Elapsed seconds: 12.92
Accuracy: 0.580670304
Classification Report:
              precision    recall  f1-score   support

        Fake       1.00      0.16      0.28       642
        Real       0.54      1.00      0.70       641

    accuracy                           0.58      1283
   macro avg       0.77      0.58      0.49      1283
weighted avg       0.77      0.58      0.49      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[104 538]
 [  0 641]]

Training Model: Logistic Regression (LR)
Elapsed seconds: 221.68
Accuracy: 0.928293063
Classification Report:
              precision    recall  f1-score   support

        Fake       0.92      0.93      0.93       642
        Real       0.93      0.92      0.93       641

    accuracy                           0.93      1283
   macro avg       0.93      0.93      0.93      1283
weighted avg       0.93      0.93      0.93      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[600  42]
 [ 50 591]]

Training Model: Random Forest (RF)
Elapsed seconds: 19.63
Accuracy: 0.913484022
Classification Report:
              precision    recall  f1-score   support

        Fake       0.92      0.90      0.91       642
        Real       0.91      0.92      0.91       641

    accuracy                           0.91      1283
   macro avg       0.91      0.91      0.91      1283
weighted avg       0.91      0.91      0.91      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[580  62]
 [ 49 592]]

Training Model: Support Vector Classifier (SVC)
Elapsed seconds: 63.03
Accuracy: 0.821512081
Classification Report:
              precision    recall  f1-score   support

        Fake       0.87      0.75      0.81       642
        Real       0.78      0.89      0.83       641

    accuracy                           0.82      1283
   macro avg       0.83      0.82      0.82      1283
weighted avg       0.83      0.82      0.82      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[484 158]
 [ 71 570]]

Saved JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\train_default_joint.json
```

### training/results/stopwords_fix_rerun/06_train_tuned.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
```

### training/results/stopwords_fix_rerun/06_train_tuned.log

```text
# Step 6: Joint-Corpus Training With Hyperparameter Tuning
Feature set: vectorizers + READ + OOV + SW + TRAD + SYLL + LEX + MORPH
Grid search: five-fold CV, scoring=accuracy
Split: train_test_split(test_size=0.2, stratify=y, random_state=42)
Dataset: Joint corpus
Rows: total=6412, train=5129, test=1283

Grid search: Multinomial Naive Bayes (MNB)
Search space: {'classifier__alpha': [0.1, 1.0, 10.0]}
Fitting 5 folds for each of 3 candidates, totalling 15 fits
Elapsed seconds: 55.29
Best params: {'alpha': 0.1}
Best CV accuracy: 0.857863738
Holdout accuracy: 0.861262666
Classification Report:
              precision    recall  f1-score   support

        Fake       0.96      0.75      0.84       642
        Real       0.80      0.97      0.87       641

    accuracy                           0.86      1283
   macro avg       0.88      0.86      0.86      1283
weighted avg       0.88      0.86      0.86      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[482 160]
 [ 18 623]]
Differs from previous best {'alpha': 0.1}: False

Grid search: Logistic Regression (LR)
Search space: {'classifier__C': [0.1, 1.0, 10.0]}
Fitting 5 folds for each of 3 candidates, totalling 15 fits
[CV] END ..............................classifier__alpha=0.1; total time=  11.8s
[CV] END ..............................classifier__alpha=1.0; total time=  11.7s
[CV] END ..............................classifier__alpha=1.0; total time=  10.2s
[CV] END .............................classifier__alpha=10.0; total time=   9.3s
[CV] END ..................................classifier__C=0.1; total time= 5.6min
[CV] END ..................................classifier__C=1.0; total time= 5.3min
[CV] END .................................classifier__C=10.0; total time= 7.1min
[CV] END ..............................classifier__alpha=0.1; total time=  11.3s
[CV] END ..............................classifier__alpha=0.1; total time=  12.3s
[CV] END .............................classifier__alpha=10.0; total time=  10.2s
[CV] END .............................classifier__alpha=10.0; total time=   9.3s
[CV] END ..................................classifier__C=0.1; total time= 3.4min
[CV] END ..................................classifier__C=0.1; total time= 5.4min
[CV] END ..................................classifier__C=1.0; total time= 4.6min
[CV] END .................................classifier__C=10.0; total time= 6.7min
[CV] END ..............................classifier__alpha=0.1; total time=  11.4s
[CV] END ..............................classifier__alpha=1.0; total time=  11.7s
[CV] END ..............................classifier__alpha=1.0; total time=  10.1s
[CV] END .............................classifier__alpha=10.0; total time=   9.3s
[CV] END ..................................classifier__C=0.1; total time= 5.3min
[CV] END ..................................classifier__C=1.0; total time= 4.4min
[CV] END ..................................classifier__C=1.0; total time= 5.1min
[CV] END .................................classifier__C=10.0; total time= 6.1min
[CV] END ..............................classifier__alpha=0.1; total time=  11.7s
[CV] END ..............................classifier__alpha=1.0; total time=  12.0s
[CV] END .............................classifier__alpha=10.0; total time=  10.3s
[CV] END ..................................classifier__C=0.1; total time= 5.8min
[CV] END ..................................classifier__C=1.0; total time= 4.9min
[CV] END .................................classifier__C=10.0; total time= 5.9min
[CV] END .................................classifier__C=10.0; total time= 4.8min
Elapsed seconds: 1440.46
Best params: {'C': 1.0}
Best CV accuracy: 0.923375077
Holdout accuracy: 0.928293063
Classification Report:
              precision    recall  f1-score   support

        Fake       0.92      0.93      0.93       642
        Real       0.93      0.92      0.93       641

    accuracy                           0.93      1283
   macro avg       0.93      0.93      0.93      1283
weighted avg       0.93      0.93      0.93      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[600  42]
 [ 50 591]]
Differs from previous best {'C': 1.0}: False

Grid search: Random Forest (RF)
Search space: {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [10, 20], 'classifier__min_samples_split': [2, 5, 10]}
Fitting 5 folds for each of 12 candidates, totalling 60 fits
Elapsed seconds: 284.13
Best params: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100}
Best CV accuracy: 0.890426092
Holdout accuracy: 0.891660171
Classification Report:
              precision    recall  f1-score   support

        Fake       0.89      0.89      0.89       642
        Real       0.89      0.89      0.89       641

    accuracy                           0.89      1283
   macro avg       0.89      0.89      0.89      1283
weighted avg       0.89      0.89      0.89      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[571  71]
 [ 68 573]]
Differs from previous best {'max_depth': 20}: True

Grid search: Support Vector Classifier (SVC)
Search space: {'classifier__C': [0.1, 1.0, 10.0], 'classifier__kernel': ['linear', 'rbf']}
Fitting 5 folds for each of 6 candidates, totalling 30 fits
Elapsed seconds: 338.75
Best params: {'C': 0.1, 'kernel': 'linear'}
Best CV accuracy: 0.923960063
Holdout accuracy: 0.926734217
Classification Report:
              precision    recall  f1-score   support

        Fake       0.92      0.94      0.93       642
        Real       0.93      0.92      0.93       641

    accuracy                           0.93      1283
   macro avg       0.93      0.93      0.93      1283
weighted avg       0.93      0.93      0.93      1283

Confusion Matrix [rows actual Fake, Real; columns predicted Fake, Real]:
[[601  41]
 [ 53 588]]
Differs from previous best {'C': 0.1, 'kernel': 'linear'}: False

Saved JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\train_tuned_joint.json
Saved best params JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\tuned_best_params.json
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  12.0s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  14.9s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  11.7s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  12.2s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  14.5s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  11.7s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  14.0s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  17.9s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  17.6s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  24.9s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  16.5s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  24.4s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  16.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  16.6s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  22.7s
[CV] END .......classifier__C=0.1, classifier__kernel=linear; total time=  34.9s
[CV] END ..........classifier__C=0.1, classifier__kernel=rbf; total time=  43.4s
[CV] END ..........classifier__C=0.1, classifier__kernel=rbf; total time=  42.9s
[CV] END ..........classifier__C=1.0, classifier__kernel=rbf; total time=  41.3s
[CV] END ..........classifier__C=1.0, classifier__kernel=rbf; total time=  40.9s
[CV] END ......classifier__C=10.0, classifier__kernel=linear; total time=  34.7s
[CV] END .........classifier__C=10.0, classifier__kernel=rbf; total time=  38.0s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  12.0s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  14.2s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  14.8s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  14.0s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  14.3s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  11.3s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  14.0s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  16.8s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  24.2s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  24.9s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  23.7s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  23.8s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  15.5s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  21.7s
[CV] END .......classifier__C=0.1, classifier__kernel=linear; total time=  34.6s
[CV] END .......classifier__C=0.1, classifier__kernel=linear; total time=  35.1s
[CV] END ..........classifier__C=0.1, classifier__kernel=rbf; total time=  43.4s
[CV] END .......classifier__C=1.0, classifier__kernel=linear; total time=  34.9s
[CV] END ..........classifier__C=1.0, classifier__kernel=rbf; total time=  40.5s
[CV] END ......classifier__C=10.0, classifier__kernel=linear; total time=  35.5s
[CV] END ......classifier__C=10.0, classifier__kernel=linear; total time=  35.2s
[CV] END .........classifier__C=10.0, classifier__kernel=rbf; total time=  36.3s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  12.0s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  14.4s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  11.7s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  11.7s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  14.3s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  11.4s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  13.9s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  14.2s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  16.3s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  24.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  16.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  16.7s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  22.8s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  15.3s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  22.1s
[CV] END .......classifier__C=0.1, classifier__kernel=linear; total time=  35.8s
[CV] END ..........classifier__C=0.1, classifier__kernel=rbf; total time=  43.9s
[CV] END .......classifier__C=1.0, classifier__kernel=linear; total time=  34.8s
[CV] END .......classifier__C=1.0, classifier__kernel=linear; total time=  35.2s
[CV] END ..........classifier__C=1.0, classifier__kernel=rbf; total time=  41.0s
[CV] END ......classifier__C=10.0, classifier__kernel=linear; total time=  34.8s
[CV] END .........classifier__C=10.0, classifier__kernel=rbf; total time=  37.6s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  11.6s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  11.9s
[CV] END classifier__max_depth=10, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  14.1s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  11.7s
[CV] END classifier__max_depth=10, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  13.9s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  11.5s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  11.7s
[CV] END classifier__max_depth=10, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  13.7s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=50; total time=  16.8s
[CV] END classifier__max_depth=20, classifier__min_samples_split=2, classifier__n_estimators=100; total time=  23.9s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  16.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=  16.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=  23.0s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=50; total time=  15.5s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  22.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=10, classifier__n_estimators=100; total time=  20.6s
[CV] END .......classifier__C=0.1, classifier__kernel=linear; total time=  35.3s
[CV] END ..........classifier__C=0.1, classifier__kernel=rbf; total time=  43.1s
[CV] END .......classifier__C=1.0, classifier__kernel=linear; total time=  35.4s
[CV] END .......classifier__C=1.0, classifier__kernel=linear; total time=  34.6s
[CV] END ..........classifier__C=1.0, classifier__kernel=rbf; total time=  41.1s
[CV] END ......classifier__C=10.0, classifier__kernel=linear; total time=  34.7s
[CV] END .........classifier__C=10.0, classifier__kernel=rbf; total time=  37.9s
[CV] END .........classifier__C=10.0, classifier__kernel=rbf; total time=  36.1s
```

### training/results/stopwords_fix_rerun/07_cross_dataset_eval.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
```

### training/results/stopwords_fix_rerun/07_cross_dataset_eval.log

```text
# Step 7: Cross-Dataset Evaluation, 30 Runs
Protocol: stratified 80% train partition, then RepeatedKFold(n_splits=5, n_repeats=6, random_state=42) on the train partition
Loading tuned best params from: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\tuned_best_params.json

CV condition: dataset=Fake News Filipino 2020, classifier=MNB, feature_set=full
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=1 | fold=1 | accuracy=0.925925926 | elapsed_seconds=4.79
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=1 | fold=2 | accuracy=0.914230019 | elapsed_seconds=4.82
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=1 | fold=3 | accuracy=0.931773879 | elapsed_seconds=4.81
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=1 | fold=4 | accuracy=0.912280702 | elapsed_seconds=4.86
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=1 | fold=5 | accuracy=0.923828125 | elapsed_seconds=4.61
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=2 | fold=1 | accuracy=0.929824561 | elapsed_seconds=4.55
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=2 | fold=2 | accuracy=0.906432749 | elapsed_seconds=4.63
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=2 | fold=3 | accuracy=0.912280702 | elapsed_seconds=4.72
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=2 | fold=4 | accuracy=0.912280702 | elapsed_seconds=5.00
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=2 | fold=5 | accuracy=0.933593750 | elapsed_seconds=5.19
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=3 | fold=1 | accuracy=0.908382066 | elapsed_seconds=4.98
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=3 | fold=2 | accuracy=0.935672515 | elapsed_seconds=5.09
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=3 | fold=3 | accuracy=0.925925926 | elapsed_seconds=4.78
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=3 | fold=4 | accuracy=0.918128655 | elapsed_seconds=4.72
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=3 | fold=5 | accuracy=0.921875000 | elapsed_seconds=4.75
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=4 | fold=1 | accuracy=0.904483431 | elapsed_seconds=4.77
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=4 | fold=2 | accuracy=0.916179337 | elapsed_seconds=4.61
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=4 | fold=3 | accuracy=0.937621832 | elapsed_seconds=4.62
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=4 | fold=4 | accuracy=0.912280702 | elapsed_seconds=4.62
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=4 | fold=5 | accuracy=0.927734375 | elapsed_seconds=4.64
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=5 | fold=1 | accuracy=0.900584795 | elapsed_seconds=4.59
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=5 | fold=2 | accuracy=0.916179337 | elapsed_seconds=4.61
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=5 | fold=3 | accuracy=0.922027290 | elapsed_seconds=4.59
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=5 | fold=4 | accuracy=0.925925926 | elapsed_seconds=4.56
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=5 | fold=5 | accuracy=0.937500000 | elapsed_seconds=4.61
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=6 | fold=1 | accuracy=0.941520468 | elapsed_seconds=4.59
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=6 | fold=2 | accuracy=0.933723197 | elapsed_seconds=4.55
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=6 | fold=3 | accuracy=0.906432749 | elapsed_seconds=4.66
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=6 | fold=4 | accuracy=0.906432749 | elapsed_seconds=4.06
Fold metric | dataset=Fake News Filipino 2020 | classifier=MNB | feature_set=full | repeat=6 | fold=5 | accuracy=0.914062500 | elapsed_seconds=4.00

CV condition: dataset=Fake News Filipino 2020, classifier=LR, feature_set=full
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=1 | fold=1 | accuracy=0.968810916 | elapsed_seconds=144.59
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=1 | fold=2 | accuracy=0.945419103 | elapsed_seconds=93.87
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=1 | fold=3 | accuracy=0.953216374 | elapsed_seconds=145.51
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=1 | fold=4 | accuracy=0.939571150 | elapsed_seconds=149.75
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=1 | fold=5 | accuracy=0.955078125 | elapsed_seconds=136.88
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=2 | fold=1 | accuracy=0.955165692 | elapsed_seconds=152.91
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=2 | fold=2 | accuracy=0.939571150 | elapsed_seconds=158.97
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=2 | fold=3 | accuracy=0.957115010 | elapsed_seconds=154.43
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=2 | fold=4 | accuracy=0.949317739 | elapsed_seconds=144.98
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=2 | fold=5 | accuracy=0.957031250 | elapsed_seconds=153.85
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=3 | fold=1 | accuracy=0.927875244 | elapsed_seconds=88.19
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=3 | fold=2 | accuracy=0.968810916 | elapsed_seconds=154.69
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=3 | fold=3 | accuracy=0.962962963 | elapsed_seconds=171.20
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=3 | fold=4 | accuracy=0.959064327 | elapsed_seconds=145.54
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=3 | fold=5 | accuracy=0.941406250 | elapsed_seconds=140.84
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=4 | fold=1 | accuracy=0.941520468 | elapsed_seconds=104.37
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=4 | fold=2 | accuracy=0.962962963 | elapsed_seconds=158.25
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=4 | fold=3 | accuracy=0.955165692 | elapsed_seconds=153.32
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=4 | fold=4 | accuracy=0.953216374 | elapsed_seconds=142.01
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=4 | fold=5 | accuracy=0.951171875 | elapsed_seconds=151.91
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=5 | fold=1 | accuracy=0.949317739 | elapsed_seconds=154.10
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=5 | fold=2 | accuracy=0.953216374 | elapsed_seconds=175.82
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=5 | fold=3 | accuracy=0.949317739 | elapsed_seconds=158.34
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=5 | fold=4 | accuracy=0.949317739 | elapsed_seconds=137.35
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=5 | fold=5 | accuracy=0.957031250 | elapsed_seconds=152.06
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=6 | fold=1 | accuracy=0.964912281 | elapsed_seconds=147.31
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=6 | fold=2 | accuracy=0.951267057 | elapsed_seconds=151.22
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=6 | fold=3 | accuracy=0.945419103 | elapsed_seconds=150.00
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=6 | fold=4 | accuracy=0.957115010 | elapsed_seconds=73.45
Fold metric | dataset=Fake News Filipino 2020 | classifier=LR | feature_set=full | repeat=6 | fold=5 | accuracy=0.947265625 | elapsed_seconds=104.16

CV condition: dataset=Fake News Filipino 2020, classifier=RF, feature_set=full
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=1 | fold=1 | accuracy=0.920077973 | elapsed_seconds=11.22
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=1 | fold=2 | accuracy=0.914230019 | elapsed_seconds=11.77
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=1 | fold=3 | accuracy=0.914230019 | elapsed_seconds=11.00
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=1 | fold=4 | accuracy=0.902534113 | elapsed_seconds=11.29
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=1 | fold=5 | accuracy=0.941406250 | elapsed_seconds=11.21
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=2 | fold=1 | accuracy=0.918128655 | elapsed_seconds=11.34
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=2 | fold=2 | accuracy=0.912280702 | elapsed_seconds=11.37
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=2 | fold=3 | accuracy=0.916179337 | elapsed_seconds=11.58
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=2 | fold=4 | accuracy=0.927875244 | elapsed_seconds=11.37
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=2 | fold=5 | accuracy=0.923828125 | elapsed_seconds=11.53
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=3 | fold=1 | accuracy=0.906432749 | elapsed_seconds=10.77
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=3 | fold=2 | accuracy=0.929824561 | elapsed_seconds=11.38
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=3 | fold=3 | accuracy=0.927875244 | elapsed_seconds=11.07
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=3 | fold=4 | accuracy=0.920077973 | elapsed_seconds=11.35
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=3 | fold=5 | accuracy=0.919921875 | elapsed_seconds=11.19
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=4 | fold=1 | accuracy=0.910331384 | elapsed_seconds=11.66
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=4 | fold=2 | accuracy=0.925925926 | elapsed_seconds=11.06
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=4 | fold=3 | accuracy=0.929824561 | elapsed_seconds=10.99
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=4 | fold=4 | accuracy=0.922027290 | elapsed_seconds=11.73
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=4 | fold=5 | accuracy=0.914062500 | elapsed_seconds=11.49
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=5 | fold=1 | accuracy=0.933723197 | elapsed_seconds=11.22
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=5 | fold=2 | accuracy=0.912280702 | elapsed_seconds=11.36
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=5 | fold=3 | accuracy=0.925925926 | elapsed_seconds=11.32
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=5 | fold=4 | accuracy=0.904483431 | elapsed_seconds=11.40
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=5 | fold=5 | accuracy=0.912109375 | elapsed_seconds=11.55
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=6 | fold=1 | accuracy=0.947368421 | elapsed_seconds=11.27
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=6 | fold=2 | accuracy=0.912280702 | elapsed_seconds=11.26
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=6 | fold=3 | accuracy=0.886939571 | elapsed_seconds=11.39
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=6 | fold=4 | accuracy=0.914230019 | elapsed_seconds=10.00
Fold metric | dataset=Fake News Filipino 2020 | classifier=RF | feature_set=full | repeat=6 | fold=5 | accuracy=0.933593750 | elapsed_seconds=10.40

CV condition: dataset=Fake News Filipino 2020, classifier=SVC, feature_set=full
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=1 | fold=1 | accuracy=0.970760234 | elapsed_seconds=10.21
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=1 | fold=2 | accuracy=0.937621832 | elapsed_seconds=10.08
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=1 | fold=3 | accuracy=0.943469786 | elapsed_seconds=10.07
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=1 | fold=4 | accuracy=0.947368421 | elapsed_seconds=9.84
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=1 | fold=5 | accuracy=0.960937500 | elapsed_seconds=10.19
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=2 | fold=1 | accuracy=0.953216374 | elapsed_seconds=10.15
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=2 | fold=2 | accuracy=0.943469786 | elapsed_seconds=9.98
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=2 | fold=3 | accuracy=0.957115010 | elapsed_seconds=9.95
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=2 | fold=4 | accuracy=0.955165692 | elapsed_seconds=10.15
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=2 | fold=5 | accuracy=0.958984375 | elapsed_seconds=10.40
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=3 | fold=1 | accuracy=0.929824561 | elapsed_seconds=9.79
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=3 | fold=2 | accuracy=0.972709552 | elapsed_seconds=10.50
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=3 | fold=3 | accuracy=0.957115010 | elapsed_seconds=10.34
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=3 | fold=4 | accuracy=0.959064327 | elapsed_seconds=10.39
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=3 | fold=5 | accuracy=0.933593750 | elapsed_seconds=9.94
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=4 | fold=1 | accuracy=0.949317739 | elapsed_seconds=10.32
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=4 | fold=2 | accuracy=0.961013645 | elapsed_seconds=10.29
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=4 | fold=3 | accuracy=0.949317739 | elapsed_seconds=10.25
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=4 | fold=4 | accuracy=0.951267057 | elapsed_seconds=10.14
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=4 | fold=5 | accuracy=0.953125000 | elapsed_seconds=9.88
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=5 | fold=1 | accuracy=0.957115010 | elapsed_seconds=10.08
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=5 | fold=2 | accuracy=0.957115010 | elapsed_seconds=10.05
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=5 | fold=3 | accuracy=0.949317739 | elapsed_seconds=10.25
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=5 | fold=4 | accuracy=0.943469786 | elapsed_seconds=10.02
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=5 | fold=5 | accuracy=0.958984375 | elapsed_seconds=9.95
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=6 | fold=1 | accuracy=0.955165692 | elapsed_seconds=10.42
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=6 | fold=2 | accuracy=0.943469786 | elapsed_seconds=10.24
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=6 | fold=3 | accuracy=0.947368421 | elapsed_seconds=9.93
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=6 | fold=4 | accuracy=0.953216374 | elapsed_seconds=9.36
Fold metric | dataset=Fake News Filipino 2020 | classifier=SVC | feature_set=full | repeat=6 | fold=5 | accuracy=0.947265625 | elapsed_seconds=9.34

CV condition: dataset=Fake News Filipino 2024, classifier=MNB, feature_set=full
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=1 | fold=1 | accuracy=0.863547758 | elapsed_seconds=4.75
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=1 | fold=2 | accuracy=0.888888889 | elapsed_seconds=4.69
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=1 | fold=3 | accuracy=0.883040936 | elapsed_seconds=4.64
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=1 | fold=4 | accuracy=0.894736842 | elapsed_seconds=4.77
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=1 | fold=5 | accuracy=0.863281250 | elapsed_seconds=4.67
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=2 | fold=1 | accuracy=0.906432749 | elapsed_seconds=4.63
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=2 | fold=2 | accuracy=0.873294347 | elapsed_seconds=4.75
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=2 | fold=3 | accuracy=0.861598441 | elapsed_seconds=4.64
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=2 | fold=4 | accuracy=0.871345029 | elapsed_seconds=4.77
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=2 | fold=5 | accuracy=0.878906250 | elapsed_seconds=4.69
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=3 | fold=1 | accuracy=0.881091618 | elapsed_seconds=4.79
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=3 | fold=2 | accuracy=0.863547758 | elapsed_seconds=4.74
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=3 | fold=3 | accuracy=0.908382066 | elapsed_seconds=4.69
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=3 | fold=4 | accuracy=0.884990253 | elapsed_seconds=4.70
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=3 | fold=5 | accuracy=0.884765625 | elapsed_seconds=4.79
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=4 | fold=1 | accuracy=0.869395712 | elapsed_seconds=4.72
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=4 | fold=2 | accuracy=0.904483431 | elapsed_seconds=4.70
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=4 | fold=3 | accuracy=0.884990253 | elapsed_seconds=4.59
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=4 | fold=4 | accuracy=0.877192982 | elapsed_seconds=4.65
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=4 | fold=5 | accuracy=0.873046875 | elapsed_seconds=4.72
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=5 | fold=1 | accuracy=0.877192982 | elapsed_seconds=4.73
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=5 | fold=2 | accuracy=0.879142300 | elapsed_seconds=4.69
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=5 | fold=3 | accuracy=0.865497076 | elapsed_seconds=4.75
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=5 | fold=4 | accuracy=0.910331384 | elapsed_seconds=4.72
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=5 | fold=5 | accuracy=0.886718750 | elapsed_seconds=4.71
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=6 | fold=1 | accuracy=0.863547758 | elapsed_seconds=4.72
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=6 | fold=2 | accuracy=0.898635478 | elapsed_seconds=4.66
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=6 | fold=3 | accuracy=0.855750487 | elapsed_seconds=4.67
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=6 | fold=4 | accuracy=0.879142300 | elapsed_seconds=4.16
Fold metric | dataset=Fake News Filipino 2024 | classifier=MNB | feature_set=full | repeat=6 | fold=5 | accuracy=0.908203125 | elapsed_seconds=4.10

CV condition: dataset=Fake News Filipino 2024, classifier=LR, feature_set=full
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=1 | fold=1 | accuracy=0.931773879 | elapsed_seconds=145.67
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=1 | fold=2 | accuracy=0.935672515 | elapsed_seconds=153.77
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=1 | fold=3 | accuracy=0.961013645 | elapsed_seconds=160.66
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=1 | fold=4 | accuracy=0.945419103 | elapsed_seconds=159.77
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=1 | fold=5 | accuracy=0.941406250 | elapsed_seconds=186.40
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=2 | fold=1 | accuracy=0.959064327 | elapsed_seconds=161.14
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=2 | fold=2 | accuracy=0.941520468 | elapsed_seconds=146.88
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=2 | fold=3 | accuracy=0.945419103 | elapsed_seconds=144.23
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=2 | fold=4 | accuracy=0.929824561 | elapsed_seconds=167.59
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=2 | fold=5 | accuracy=0.939453125 | elapsed_seconds=138.75
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=3 | fold=1 | accuracy=0.961013645 | elapsed_seconds=139.63
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=3 | fold=2 | accuracy=0.933723197 | elapsed_seconds=143.26
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=3 | fold=3 | accuracy=0.947368421 | elapsed_seconds=142.74
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=3 | fold=4 | accuracy=0.955165692 | elapsed_seconds=148.35
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=3 | fold=5 | accuracy=0.949218750 | elapsed_seconds=141.98
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=4 | fold=1 | accuracy=0.949317739 | elapsed_seconds=134.84
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=4 | fold=2 | accuracy=0.953216374 | elapsed_seconds=143.31
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=4 | fold=3 | accuracy=0.957115010 | elapsed_seconds=153.35
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=4 | fold=4 | accuracy=0.929824561 | elapsed_seconds=149.58
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=4 | fold=5 | accuracy=0.935546875 | elapsed_seconds=148.39
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=5 | fold=1 | accuracy=0.949317739 | elapsed_seconds=138.36
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=5 | fold=2 | accuracy=0.951267057 | elapsed_seconds=155.80
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=5 | fold=3 | accuracy=0.943469786 | elapsed_seconds=142.84
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=5 | fold=4 | accuracy=0.955165692 | elapsed_seconds=150.60
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=5 | fold=5 | accuracy=0.935546875 | elapsed_seconds=149.33
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=6 | fold=1 | accuracy=0.957115010 | elapsed_seconds=154.82
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=6 | fold=2 | accuracy=0.945419103 | elapsed_seconds=153.47
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=6 | fold=3 | accuracy=0.923976608 | elapsed_seconds=156.53
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=6 | fold=4 | accuracy=0.941520468 | elapsed_seconds=123.70
Fold metric | dataset=Fake News Filipino 2024 | classifier=LR | feature_set=full | repeat=6 | fold=5 | accuracy=0.958984375 | elapsed_seconds=86.85

CV condition: dataset=Fake News Filipino 2024, classifier=RF, feature_set=full
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=1 | fold=1 | accuracy=0.923976608 | elapsed_seconds=11.90
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=1 | fold=2 | accuracy=0.912280702 | elapsed_seconds=11.92
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=1 | fold=3 | accuracy=0.941520468 | elapsed_seconds=12.40
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=1 | fold=4 | accuracy=0.925925926 | elapsed_seconds=12.41
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=1 | fold=5 | accuracy=0.931640625 | elapsed_seconds=12.65
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=2 | fold=1 | accuracy=0.935672515 | elapsed_seconds=12.25
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=2 | fold=2 | accuracy=0.908382066 | elapsed_seconds=11.90
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=2 | fold=3 | accuracy=0.927875244 | elapsed_seconds=11.90
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=2 | fold=4 | accuracy=0.922027290 | elapsed_seconds=12.48
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=2 | fold=5 | accuracy=0.914062500 | elapsed_seconds=11.86
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=3 | fold=1 | accuracy=0.931773879 | elapsed_seconds=12.27
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=3 | fold=2 | accuracy=0.914230019 | elapsed_seconds=11.90
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=3 | fold=3 | accuracy=0.933723197 | elapsed_seconds=12.37
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=3 | fold=4 | accuracy=0.920077973 | elapsed_seconds=11.78
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=3 | fold=5 | accuracy=0.935546875 | elapsed_seconds=12.26
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=4 | fold=1 | accuracy=0.923976608 | elapsed_seconds=12.14
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=4 | fold=2 | accuracy=0.937621832 | elapsed_seconds=12.12
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=4 | fold=3 | accuracy=0.947368421 | elapsed_seconds=12.06
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=4 | fold=4 | accuracy=0.908382066 | elapsed_seconds=11.96
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=4 | fold=5 | accuracy=0.914062500 | elapsed_seconds=12.01
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=5 | fold=1 | accuracy=0.941520468 | elapsed_seconds=11.95
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=5 | fold=2 | accuracy=0.923976608 | elapsed_seconds=11.77
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=5 | fold=3 | accuracy=0.906432749 | elapsed_seconds=12.32
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=5 | fold=4 | accuracy=0.927875244 | elapsed_seconds=12.01
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=5 | fold=5 | accuracy=0.931640625 | elapsed_seconds=12.27
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=6 | fold=1 | accuracy=0.918128655 | elapsed_seconds=12.43
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=6 | fold=2 | accuracy=0.925925926 | elapsed_seconds=11.93
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=6 | fold=3 | accuracy=0.927875244 | elapsed_seconds=12.00
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=6 | fold=4 | accuracy=0.922027290 | elapsed_seconds=11.57
Fold metric | dataset=Fake News Filipino 2024 | classifier=RF | feature_set=full | repeat=6 | fold=5 | accuracy=0.937500000 | elapsed_seconds=10.47

CV condition: dataset=Fake News Filipino 2024, classifier=SVC, feature_set=full
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=1 | fold=1 | accuracy=0.937621832 | elapsed_seconds=10.04
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=1 | fold=2 | accuracy=0.939571150 | elapsed_seconds=10.09
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=1 | fold=3 | accuracy=0.957115010 | elapsed_seconds=10.28
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=1 | fold=4 | accuracy=0.947368421 | elapsed_seconds=10.14
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=1 | fold=5 | accuracy=0.941406250 | elapsed_seconds=10.23
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=2 | fold=1 | accuracy=0.951267057 | elapsed_seconds=10.24
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=2 | fold=2 | accuracy=0.953216374 | elapsed_seconds=10.32
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=2 | fold=3 | accuracy=0.947368421 | elapsed_seconds=10.19
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=2 | fold=4 | accuracy=0.933723197 | elapsed_seconds=10.09
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=2 | fold=5 | accuracy=0.939453125 | elapsed_seconds=10.19
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=3 | fold=1 | accuracy=0.955165692 | elapsed_seconds=10.38
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=3 | fold=2 | accuracy=0.937621832 | elapsed_seconds=10.06
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=3 | fold=3 | accuracy=0.943469786 | elapsed_seconds=10.39
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=3 | fold=4 | accuracy=0.951267057 | elapsed_seconds=10.11
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=3 | fold=5 | accuracy=0.941406250 | elapsed_seconds=10.29
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=4 | fold=1 | accuracy=0.947368421 | elapsed_seconds=10.31
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=4 | fold=2 | accuracy=0.959064327 | elapsed_seconds=10.26
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=4 | fold=3 | accuracy=0.964912281 | elapsed_seconds=10.13
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=4 | fold=4 | accuracy=0.931773879 | elapsed_seconds=10.02
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=4 | fold=5 | accuracy=0.937500000 | elapsed_seconds=10.25
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=5 | fold=1 | accuracy=0.957115010 | elapsed_seconds=10.06
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=5 | fold=2 | accuracy=0.957115010 | elapsed_seconds=10.26
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=5 | fold=3 | accuracy=0.939571150 | elapsed_seconds=9.81
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=5 | fold=4 | accuracy=0.964912281 | elapsed_seconds=10.27
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=5 | fold=5 | accuracy=0.947265625 | elapsed_seconds=10.26
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=6 | fold=1 | accuracy=0.953216374 | elapsed_seconds=10.31
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=6 | fold=2 | accuracy=0.945419103 | elapsed_seconds=10.12
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=6 | fold=3 | accuracy=0.923976608 | elapsed_seconds=9.94
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=6 | fold=4 | accuracy=0.945419103 | elapsed_seconds=9.45
Fold metric | dataset=Fake News Filipino 2024 | classifier=SVC | feature_set=full | repeat=6 | fold=5 | accuracy=0.958984375 | elapsed_seconds=9.59

CV condition: dataset=Joint corpus, classifier=MNB, feature_set=full
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=1 | fold=1 | accuracy=0.864522417 | elapsed_seconds=9.23
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=1 | fold=2 | accuracy=0.841130604 | elapsed_seconds=9.47
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=1 | fold=3 | accuracy=0.877192982 | elapsed_seconds=9.44
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=1 | fold=4 | accuracy=0.868421053 | elapsed_seconds=9.36
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=1 | fold=5 | accuracy=0.831219512 | elapsed_seconds=9.20
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=2 | fold=1 | accuracy=0.850877193 | elapsed_seconds=9.28
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=2 | fold=2 | accuracy=0.860623782 | elapsed_seconds=9.20
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=2 | fold=3 | accuracy=0.883040936 | elapsed_seconds=9.44
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=2 | fold=4 | accuracy=0.849902534 | elapsed_seconds=9.36
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=2 | fold=5 | accuracy=0.840000000 | elapsed_seconds=9.36
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=3 | fold=1 | accuracy=0.845029240 | elapsed_seconds=9.21
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=3 | fold=2 | accuracy=0.848927875 | elapsed_seconds=9.40
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=3 | fold=3 | accuracy=0.877192982 | elapsed_seconds=9.34
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=3 | fold=4 | accuracy=0.858674464 | elapsed_seconds=9.24
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=3 | fold=5 | accuracy=0.848780488 | elapsed_seconds=9.34
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=4 | fold=1 | accuracy=0.860623782 | elapsed_seconds=9.29
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=4 | fold=2 | accuracy=0.863547758 | elapsed_seconds=9.32
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=4 | fold=3 | accuracy=0.850877193 | elapsed_seconds=9.41
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=4 | fold=4 | accuracy=0.851851852 | elapsed_seconds=9.34
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=4 | fold=5 | accuracy=0.859512195 | elapsed_seconds=9.39
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=5 | fold=1 | accuracy=0.875243665 | elapsed_seconds=9.24
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=5 | fold=2 | accuracy=0.859649123 | elapsed_seconds=9.42
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=5 | fold=3 | accuracy=0.852826511 | elapsed_seconds=9.25
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=5 | fold=4 | accuracy=0.849902534 | elapsed_seconds=9.39
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=5 | fold=5 | accuracy=0.843902439 | elapsed_seconds=9.27
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=6 | fold=1 | accuracy=0.862573099 | elapsed_seconds=9.30
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=6 | fold=2 | accuracy=0.879142300 | elapsed_seconds=9.32
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=6 | fold=3 | accuracy=0.860623782 | elapsed_seconds=9.32
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=6 | fold=4 | accuracy=0.850877193 | elapsed_seconds=8.36
Fold metric | dataset=Joint corpus | classifier=MNB | feature_set=full | repeat=6 | fold=5 | accuracy=0.823414634 | elapsed_seconds=8.33

CV condition: dataset=Joint corpus, classifier=LR, feature_set=full
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=1 | fold=1 | accuracy=0.920077973 | elapsed_seconds=425.98
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=1 | fold=2 | accuracy=0.902534113 | elapsed_seconds=316.67
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=1 | fold=3 | accuracy=0.936647173 | elapsed_seconds=293.63
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=1 | fold=4 | accuracy=0.928849903 | elapsed_seconds=313.22
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=1 | fold=5 | accuracy=0.924878049 | elapsed_seconds=290.27
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=2 | fold=1 | accuracy=0.923001949 | elapsed_seconds=316.18
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=2 | fold=2 | accuracy=0.918128655 | elapsed_seconds=305.90
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=2 | fold=3 | accuracy=0.928849903 | elapsed_seconds=266.69
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=2 | fold=4 | accuracy=0.927875244 | elapsed_seconds=292.29
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=2 | fold=5 | accuracy=0.920975610 | elapsed_seconds=283.31
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=3 | fold=1 | accuracy=0.924951267 | elapsed_seconds=263.45
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=3 | fold=2 | accuracy=0.923976608 | elapsed_seconds=261.46
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=3 | fold=3 | accuracy=0.930799220 | elapsed_seconds=283.25
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=3 | fold=4 | accuracy=0.916179337 | elapsed_seconds=295.48
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=3 | fold=5 | accuracy=0.911219512 | elapsed_seconds=428.39
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=4 | fold=1 | accuracy=0.926900585 | elapsed_seconds=438.12
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=4 | fold=2 | accuracy=0.923001949 | elapsed_seconds=292.98
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=4 | fold=3 | accuracy=0.940545809 | elapsed_seconds=439.00
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=4 | fold=4 | accuracy=0.905458090 | elapsed_seconds=317.43
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=4 | fold=5 | accuracy=0.922926829 | elapsed_seconds=430.97
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=5 | fold=1 | accuracy=0.946393762 | elapsed_seconds=308.58
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=5 | fold=2 | accuracy=0.926900585 | elapsed_seconds=310.36
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=5 | fold=3 | accuracy=0.916179337 | elapsed_seconds=275.87
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=5 | fold=4 | accuracy=0.913255361 | elapsed_seconds=294.96
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=5 | fold=5 | accuracy=0.911219512 | elapsed_seconds=278.57
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=6 | fold=1 | accuracy=0.920077973 | elapsed_seconds=429.32
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=6 | fold=2 | accuracy=0.926900585 | elapsed_seconds=299.20
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=6 | fold=3 | accuracy=0.920077973 | elapsed_seconds=404.41
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=6 | fold=4 | accuracy=0.934697856 | elapsed_seconds=389.76
Fold metric | dataset=Joint corpus | classifier=LR | feature_set=full | repeat=6 | fold=5 | accuracy=0.919024390 | elapsed_seconds=264.28

CV condition: dataset=Joint corpus, classifier=RF, feature_set=full
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=1 | fold=1 | accuracy=0.891812865 | elapsed_seconds=25.18
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=1 | fold=2 | accuracy=0.872319688 | elapsed_seconds=25.39
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=1 | fold=3 | accuracy=0.907407407 | elapsed_seconds=24.11
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=1 | fold=4 | accuracy=0.911306043 | elapsed_seconds=24.45
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=1 | fold=5 | accuracy=0.868292683 | elapsed_seconds=23.93
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=2 | fold=1 | accuracy=0.884990253 | elapsed_seconds=24.83
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=2 | fold=2 | accuracy=0.891812865 | elapsed_seconds=25.07
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=2 | fold=3 | accuracy=0.895711501 | elapsed_seconds=24.49
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=2 | fold=4 | accuracy=0.885964912 | elapsed_seconds=25.20
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=2 | fold=5 | accuracy=0.889756098 | elapsed_seconds=24.66
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=3 | fold=1 | accuracy=0.882066277 | elapsed_seconds=24.98
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=3 | fold=2 | accuracy=0.890838207 | elapsed_seconds=25.04
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=3 | fold=3 | accuracy=0.898635478 | elapsed_seconds=23.89
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=3 | fold=4 | accuracy=0.886939571 | elapsed_seconds=24.47
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=3 | fold=5 | accuracy=0.902439024 | elapsed_seconds=24.77
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=4 | fold=1 | accuracy=0.903508772 | elapsed_seconds=25.12
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=4 | fold=2 | accuracy=0.884015595 | elapsed_seconds=24.61
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=4 | fold=3 | accuracy=0.897660819 | elapsed_seconds=24.55
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=4 | fold=4 | accuracy=0.869395712 | elapsed_seconds=24.70
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=4 | fold=5 | accuracy=0.886829268 | elapsed_seconds=25.08
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=5 | fold=1 | accuracy=0.909356725 | elapsed_seconds=24.66
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=5 | fold=2 | accuracy=0.905458090 | elapsed_seconds=24.58
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=5 | fold=3 | accuracy=0.885964912 | elapsed_seconds=24.45
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=5 | fold=4 | accuracy=0.886939571 | elapsed_seconds=24.58
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=5 | fold=5 | accuracy=0.886829268 | elapsed_seconds=24.50
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=6 | fold=1 | accuracy=0.884990253 | elapsed_seconds=24.67
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=6 | fold=2 | accuracy=0.894736842 | elapsed_seconds=25.04
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=6 | fold=3 | accuracy=0.890838207 | elapsed_seconds=25.02
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=6 | fold=4 | accuracy=0.901559454 | elapsed_seconds=23.31
Fold metric | dataset=Joint corpus | classifier=RF | feature_set=full | repeat=6 | fold=5 | accuracy=0.867317073 | elapsed_seconds=22.52

CV condition: dataset=Joint corpus, classifier=SVC, feature_set=full
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=1 | fold=1 | accuracy=0.916179337 | elapsed_seconds=35.06
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=1 | fold=2 | accuracy=0.910331384 | elapsed_seconds=34.95
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=1 | fold=3 | accuracy=0.924951267 | elapsed_seconds=35.59
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=1 | fold=4 | accuracy=0.926900585 | elapsed_seconds=35.01
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=1 | fold=5 | accuracy=0.929756098 | elapsed_seconds=34.95
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=2 | fold=1 | accuracy=0.927875244 | elapsed_seconds=35.91
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=2 | fold=2 | accuracy=0.919103314 | elapsed_seconds=35.10
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=2 | fold=3 | accuracy=0.931773879 | elapsed_seconds=35.68
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=2 | fold=4 | accuracy=0.927875244 | elapsed_seconds=34.85
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=2 | fold=5 | accuracy=0.920975610 | elapsed_seconds=34.74
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=3 | fold=1 | accuracy=0.921052632 | elapsed_seconds=35.37
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=3 | fold=2 | accuracy=0.920077973 | elapsed_seconds=35.25
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=3 | fold=3 | accuracy=0.923976608 | elapsed_seconds=34.82
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=3 | fold=4 | accuracy=0.920077973 | elapsed_seconds=35.24
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=3 | fold=5 | accuracy=0.921951220 | elapsed_seconds=35.31
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=4 | fold=1 | accuracy=0.923976608 | elapsed_seconds=35.43
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=4 | fold=2 | accuracy=0.924951267 | elapsed_seconds=35.81
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=4 | fold=3 | accuracy=0.935672515 | elapsed_seconds=35.36
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=4 | fold=4 | accuracy=0.911306043 | elapsed_seconds=34.64
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=4 | fold=5 | accuracy=0.918048780 | elapsed_seconds=35.03
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=5 | fold=1 | accuracy=0.949317739 | elapsed_seconds=35.68
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=5 | fold=2 | accuracy=0.924951267 | elapsed_seconds=34.86
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=5 | fold=3 | accuracy=0.915204678 | elapsed_seconds=35.33
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=5 | fold=4 | accuracy=0.917153996 | elapsed_seconds=35.80
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=5 | fold=5 | accuracy=0.908292683 | elapsed_seconds=35.40
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=6 | fold=1 | accuracy=0.920077973 | elapsed_seconds=35.51
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=6 | fold=2 | accuracy=0.926900585 | elapsed_seconds=35.68
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=6 | fold=3 | accuracy=0.908382066 | elapsed_seconds=34.90
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=6 | fold=4 | accuracy=0.935672515 | elapsed_seconds=33.30
Fold metric | dataset=Joint corpus | classifier=SVC | feature_set=full | repeat=6 | fold=5 | accuracy=0.919024390 | elapsed_seconds=33.00

Mean accuracy summary:
                dataset classifier           classifier_name feature_set  mean_accuracy  sd_accuracy
Fake News Filipino 2020         LR       Logistic Regression        full       0.952288     0.009114
Fake News Filipino 2020        MNB   Multinomial Naive Bayes        full       0.920504     0.011384
Fake News Filipino 2020         RF             Random Forest        full       0.919334     0.012154
Fake News Filipino 2020        SVC Support Vector Classifier        full       0.951898     0.009530
Fake News Filipino 2024         LR       Logistic Regression        full       0.945462     0.010354
Fake News Filipino 2024        MNB   Multinomial Naive Bayes        full       0.881371     0.015605
Fake News Filipino 2024         RF             Random Forest        full       0.925768     0.010687
Fake News Filipino 2024        SVC Support Vector Classifier        full       0.947022     0.009949
           Joint corpus         LR       Logistic Regression        full       0.923084     0.009553
           Joint corpus        MNB   Multinomial Naive Bayes        full       0.856337     0.013968
           Joint corpus         RF             Random Forest        full       0.890523     0.011704
           Joint corpus        SVC Support Vector Classifier        full       0.922726     0.008679

Saved raw CSV: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\cross_dataset_30run_raw.csv
Saved summary JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\cross_dataset_30run_summary.json
```

### training/results/stopwords_fix_rerun/08_anova.log

```text
# Step 8: ANOVA Pipeline
Input CSV: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\cross_dataset_30run_raw.csv

Shapiro-Wilk tests:
                dataset classifier        W        p
Fake News Filipino 2020        MNB 0.960576 0.320437
Fake News Filipino 2020         LR 0.973468 0.637729
Fake News Filipino 2020         RF 0.976488 0.726482
Fake News Filipino 2020        SVC 0.971863 0.591321
Fake News Filipino 2024        MNB 0.941648 0.100748
Fake News Filipino 2024         LR 0.962666 0.361719
Fake News Filipino 2024         RF 0.978086 0.772686
Fake News Filipino 2024        SVC 0.979374 0.808671
           Joint corpus        MNB 0.972845 0.619572
           Joint corpus         LR 0.982882 0.895883
           Joint corpus         RF 0.956926 0.257988
           Joint corpus        SVC 0.948676 0.155791

Levene's test:
       W        p
2.096062 0.020043

Two-way ANOVA for trimmed means:
              factor   statistic     p
             dataset  721.356750 0.001
          classifier 1142.943020 0.001
dataset * classifier  124.354493 0.001

Standard two-way ANOVA partial eta-squared:
              effect  df_effect  df_error          F             p  partial_eta_squared
             dataset          2       348 359.880231  1.906958e-85             0.674084
          classifier          3       348 487.594495 3.116224e-124             0.807818
dataset * classifier          6       348  28.735307  6.735318e-28             0.331299

Bonferroni post-hoc classifier pairs within each dataset:
                dataset classifier_a classifier_b        raw_p  bonferroni_p
Fake News Filipino 2020          MNB           LR 6.101889e-17  3.661134e-16
Fake News Filipino 2020          MNB           RF 7.016625e-01  1.000000e+00
Fake News Filipino 2020          MNB          SVC 1.597996e-16  9.587976e-16
Fake News Filipino 2020           LR           RF 1.153796e-16  6.922776e-16
Fake News Filipino 2020           LR          SVC 8.720009e-01  1.000000e+00
Fake News Filipino 2020           RF          SVC 2.588863e-16  1.553318e-15
Fake News Filipino 2024          MNB           LR 2.294570e-24  1.376742e-23
Fake News Filipino 2024          MNB           RF 1.137838e-17  6.827028e-17
Fake News Filipino 2024          MNB          SVC 9.789445e-25  5.873667e-24
Fake News Filipino 2024           LR           RF 1.122621e-09  6.735723e-09
Fake News Filipino 2024           LR          SVC 5.541574e-01  1.000000e+00
Fake News Filipino 2024           RF          SVC 7.030532e-11  4.218319e-10
           Joint corpus          MNB           LR 2.093883e-27  1.256330e-26
           Joint corpus          MNB           RF 1.600837e-14  9.605023e-14
           Joint corpus          MNB          SVC 5.617426e-27  3.370456e-26
           Joint corpus           LR           RF 8.542102e-17  5.125261e-16
           Joint corpus           LR          SVC 8.800679e-01  1.000000e+00
           Joint corpus           RF          SVC 6.039969e-17  3.623982e-16

Bonferroni post-hoc dataset pairs within each classifier:
classifier               dataset_a               dataset_b        raw_p  bonferroni_p
       MNB Fake News Filipino 2020 Fake News Filipino 2024 1.925111e-15  5.775334e-15
       MNB Fake News Filipino 2020            Joint corpus 1.434120e-26  4.302360e-26
       MNB Fake News Filipino 2024            Joint corpus 1.764995e-08  5.294985e-08
        LR Fake News Filipino 2020 Fake News Filipino 2024 8.857894e-03  2.657368e-02
        LR Fake News Filipino 2020            Joint corpus 1.640475e-17  4.921425e-17
        LR Fake News Filipino 2024            Joint corpus 4.366072e-12  1.309822e-11
        RF Fake News Filipino 2020 Fake News Filipino 2024 3.359607e-02  1.007882e-01
        RF Fake News Filipino 2020            Joint corpus 3.545076e-13  1.063523e-12
        RF Fake News Filipino 2024            Joint corpus 1.450236e-17  4.350707e-17
       SVC Fake News Filipino 2020 Fake News Filipino 2024 5.741956e-02  1.722587e-01
       SVC Fake News Filipino 2020            Joint corpus 6.986809e-18  2.096043e-17
       SVC Fake News Filipino 2024            Joint corpus 2.838297e-14  8.514890e-14

Saved JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\anova_pipeline.json
```

### training/results/stopwords_fix_rerun/09_roc_auc.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
```

### training/results/stopwords_fix_rerun/09_roc_auc.log

```text
# Step 9: ROC-AUC on Joint Corpus
Protocol: stratified 80% train partition, then RepeatedKFold(n_splits=5, n_repeats=6, random_state=42) on the train partition
Loading tuned best params from: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\tuned_best_params.json

ROC-AUC condition: dataset=Joint corpus, classifier=MNB
AUC metric | classifier=MNB | repeat=1 | fold=1 | auc=0.925272457 | elapsed_seconds=9.48
AUC metric | classifier=MNB | repeat=1 | fold=2 | auc=0.901635077 | elapsed_seconds=9.96
AUC metric | classifier=MNB | repeat=1 | fold=3 | auc=0.937137081 | elapsed_seconds=9.87
AUC metric | classifier=MNB | repeat=1 | fold=4 | auc=0.935070240 | elapsed_seconds=9.70
AUC metric | classifier=MNB | repeat=1 | fold=5 | auc=0.906466548 | elapsed_seconds=9.72
AUC metric | classifier=MNB | repeat=2 | fold=1 | auc=0.915225663 | elapsed_seconds=9.83
AUC metric | classifier=MNB | repeat=2 | fold=2 | auc=0.929592504 | elapsed_seconds=9.73
AUC metric | classifier=MNB | repeat=2 | fold=3 | auc=0.929671444 | elapsed_seconds=9.86
AUC metric | classifier=MNB | repeat=2 | fold=4 | auc=0.917569901 | elapsed_seconds=9.29
AUC metric | classifier=MNB | repeat=2 | fold=5 | auc=0.916433935 | elapsed_seconds=9.36
AUC metric | classifier=MNB | repeat=3 | fold=1 | auc=0.914675483 | elapsed_seconds=9.23
AUC metric | classifier=MNB | repeat=3 | fold=2 | auc=0.912066803 | elapsed_seconds=9.62
AUC metric | classifier=MNB | repeat=3 | fold=3 | auc=0.932189220 | elapsed_seconds=9.26
AUC metric | classifier=MNB | repeat=3 | fold=4 | auc=0.925673159 | elapsed_seconds=9.32
AUC metric | classifier=MNB | repeat=3 | fold=5 | auc=0.917683252 | elapsed_seconds=9.33
AUC metric | classifier=MNB | repeat=4 | fold=1 | auc=0.925559129 | elapsed_seconds=9.42
AUC metric | classifier=MNB | repeat=4 | fold=2 | auc=0.920173677 | elapsed_seconds=9.26
AUC metric | classifier=MNB | repeat=4 | fold=3 | auc=0.931883926 | elapsed_seconds=9.23
AUC metric | classifier=MNB | repeat=4 | fold=4 | auc=0.912464602 | elapsed_seconds=9.31
AUC metric | classifier=MNB | repeat=4 | fold=5 | auc=0.921941531 | elapsed_seconds=9.42
AUC metric | classifier=MNB | repeat=5 | fold=1 | auc=0.938067441 | elapsed_seconds=9.43
AUC metric | classifier=MNB | repeat=5 | fold=2 | auc=0.922608372 | elapsed_seconds=9.28
AUC metric | classifier=MNB | repeat=5 | fold=3 | auc=0.918915324 | elapsed_seconds=9.16
AUC metric | classifier=MNB | repeat=5 | fold=4 | auc=0.920178799 | elapsed_seconds=9.33
AUC metric | classifier=MNB | repeat=5 | fold=5 | auc=0.911227124 | elapsed_seconds=9.30
AUC metric | classifier=MNB | repeat=6 | fold=1 | auc=0.927794857 | elapsed_seconds=9.25
AUC metric | classifier=MNB | repeat=6 | fold=2 | auc=0.932094913 | elapsed_seconds=9.32
AUC metric | classifier=MNB | repeat=6 | fold=3 | auc=0.921533894 | elapsed_seconds=9.42
AUC metric | classifier=MNB | repeat=6 | fold=4 | auc=0.932813257 | elapsed_seconds=8.29
AUC metric | classifier=MNB | repeat=6 | fold=5 | auc=0.902682903 | elapsed_seconds=8.39

ROC-AUC condition: dataset=Joint corpus, classifier=LR
AUC metric | classifier=LR | repeat=1 | fold=1 | auc=0.969437676 | elapsed_seconds=422.13
AUC metric | classifier=LR | repeat=1 | fold=2 | auc=0.974381384 | elapsed_seconds=316.14
AUC metric | classifier=LR | repeat=1 | fold=3 | auc=0.976947983 | elapsed_seconds=287.95
AUC metric | classifier=LR | repeat=1 | fold=4 | auc=0.979822851 | elapsed_seconds=309.14
AUC metric | classifier=LR | repeat=1 | fold=5 | auc=0.976195915 | elapsed_seconds=289.83
AUC metric | classifier=LR | repeat=2 | fold=1 | auc=0.974603621 | elapsed_seconds=316.96
AUC metric | classifier=LR | repeat=2 | fold=2 | auc=0.974206590 | elapsed_seconds=308.61
AUC metric | classifier=LR | repeat=2 | fold=3 | auc=0.977675593 | elapsed_seconds=266.74
AUC metric | classifier=LR | repeat=2 | fold=4 | auc=0.976698461 | elapsed_seconds=293.91
AUC metric | classifier=LR | repeat=2 | fold=5 | auc=0.975193053 | elapsed_seconds=283.60
AUC metric | classifier=LR | repeat=3 | fold=1 | auc=0.975269798 | elapsed_seconds=264.02
AUC metric | classifier=LR | repeat=3 | fold=2 | auc=0.976820185 | elapsed_seconds=261.67
AUC metric | classifier=LR | repeat=3 | fold=3 | auc=0.975469705 | elapsed_seconds=281.76
AUC metric | classifier=LR | repeat=3 | fold=4 | auc=0.976068844 | elapsed_seconds=294.12
AUC metric | classifier=LR | repeat=3 | fold=5 | auc=0.975320205 | elapsed_seconds=431.43
AUC metric | classifier=LR | repeat=4 | fold=1 | auc=0.979964879 | elapsed_seconds=437.20
AUC metric | classifier=LR | repeat=4 | fold=2 | auc=0.978148849 | elapsed_seconds=294.06
AUC metric | classifier=LR | repeat=4 | fold=3 | auc=0.982647453 | elapsed_seconds=441.32
AUC metric | classifier=LR | repeat=4 | fold=4 | auc=0.960993880 | elapsed_seconds=318.60
AUC metric | classifier=LR | repeat=4 | fold=5 | auc=0.973709467 | elapsed_seconds=428.38
AUC metric | classifier=LR | repeat=5 | fold=1 | auc=0.984255111 | elapsed_seconds=309.98
AUC metric | classifier=LR | repeat=5 | fold=2 | auc=0.975544139 | elapsed_seconds=313.47
AUC metric | classifier=LR | repeat=5 | fold=3 | auc=0.974885984 | elapsed_seconds=277.41
AUC metric | classifier=LR | repeat=5 | fold=4 | auc=0.973544974 | elapsed_seconds=290.45
AUC metric | classifier=LR | repeat=5 | fold=5 | auc=0.969674122 | elapsed_seconds=274.30
AUC metric | classifier=LR | repeat=6 | fold=1 | auc=0.969816039 | elapsed_seconds=431.46
AUC metric | classifier=LR | repeat=6 | fold=2 | auc=0.978980853 | elapsed_seconds=295.91
AUC metric | classifier=LR | repeat=6 | fold=3 | auc=0.973645698 | elapsed_seconds=408.67
AUC metric | classifier=LR | repeat=6 | fold=4 | auc=0.983902024 | elapsed_seconds=395.95
AUC metric | classifier=LR | repeat=6 | fold=5 | auc=0.971274741 | elapsed_seconds=264.77

ROC-AUC condition: dataset=Joint corpus, classifier=RF
AUC metric | classifier=RF | repeat=1 | fold=1 | auc=0.962401499 | elapsed_seconds=25.28
AUC metric | classifier=RF | repeat=1 | fold=2 | auc=0.946064871 | elapsed_seconds=24.97
AUC metric | classifier=RF | repeat=1 | fold=3 | auc=0.967021859 | elapsed_seconds=24.40
AUC metric | classifier=RF | repeat=1 | fold=4 | auc=0.971482203 | elapsed_seconds=24.42
AUC metric | classifier=RF | repeat=1 | fold=5 | auc=0.953949072 | elapsed_seconds=24.44
AUC metric | classifier=RF | repeat=2 | fold=1 | auc=0.954859752 | elapsed_seconds=24.98
AUC metric | classifier=RF | repeat=2 | fold=2 | auc=0.962586637 | elapsed_seconds=24.13
AUC metric | classifier=RF | repeat=2 | fold=3 | auc=0.958515875 | elapsed_seconds=24.75
AUC metric | classifier=RF | repeat=2 | fold=4 | auc=0.958177407 | elapsed_seconds=25.72
AUC metric | classifier=RF | repeat=2 | fold=5 | auc=0.962762873 | elapsed_seconds=24.31
AUC metric | classifier=RF | repeat=3 | fold=1 | auc=0.960552516 | elapsed_seconds=24.84
AUC metric | classifier=RF | repeat=3 | fold=2 | auc=0.958971728 | elapsed_seconds=24.76
AUC metric | classifier=RF | repeat=3 | fold=3 | auc=0.966786246 | elapsed_seconds=24.26
AUC metric | classifier=RF | repeat=3 | fold=4 | auc=0.958903485 | elapsed_seconds=24.23
AUC metric | classifier=RF | repeat=3 | fold=5 | auc=0.962218821 | elapsed_seconds=24.71
AUC metric | classifier=RF | repeat=4 | fold=1 | auc=0.967573588 | elapsed_seconds=24.90
AUC metric | classifier=RF | repeat=4 | fold=2 | auc=0.958746007 | elapsed_seconds=24.66
AUC metric | classifier=RF | repeat=4 | fold=3 | auc=0.963977954 | elapsed_seconds=24.57
AUC metric | classifier=RF | repeat=4 | fold=4 | auc=0.949381109 | elapsed_seconds=24.61
AUC metric | classifier=RF | repeat=4 | fold=5 | auc=0.955355987 | elapsed_seconds=24.89
AUC metric | classifier=RF | repeat=5 | fold=1 | auc=0.969817545 | elapsed_seconds=24.46
AUC metric | classifier=RF | repeat=5 | fold=2 | auc=0.966690479 | elapsed_seconds=24.58
AUC metric | classifier=RF | repeat=5 | fold=3 | auc=0.958414412 | elapsed_seconds=24.55
AUC metric | classifier=RF | repeat=5 | fold=4 | auc=0.959112540 | elapsed_seconds=24.74
AUC metric | classifier=RF | repeat=5 | fold=5 | auc=0.958014424 | elapsed_seconds=24.51
AUC metric | classifier=RF | repeat=6 | fold=1 | auc=0.958871835 | elapsed_seconds=25.14
AUC metric | classifier=RF | repeat=6 | fold=2 | auc=0.962991994 | elapsed_seconds=25.24
AUC metric | classifier=RF | repeat=6 | fold=3 | auc=0.956327047 | elapsed_seconds=24.87
AUC metric | classifier=RF | repeat=6 | fold=4 | auc=0.966263645 | elapsed_seconds=23.19
AUC metric | classifier=RF | repeat=6 | fold=5 | auc=0.951440451 | elapsed_seconds=22.68

ROC-AUC condition: dataset=Joint corpus, classifier=SVC
AUC metric | classifier=SVC | repeat=1 | fold=1 | auc=0.967746105 | elapsed_seconds=127.34
AUC metric | classifier=SVC | repeat=1 | fold=2 | auc=0.974540978 | elapsed_seconds=126.79
AUC metric | classifier=SVC | repeat=1 | fold=3 | auc=0.975028882 | elapsed_seconds=127.95
AUC metric | classifier=SVC | repeat=1 | fold=4 | auc=0.977185763 | elapsed_seconds=125.63
AUC metric | classifier=SVC | repeat=1 | fold=5 | auc=0.974634867 | elapsed_seconds=125.57
AUC metric | classifier=SVC | repeat=2 | fold=1 | auc=0.974143666 | elapsed_seconds=127.68
AUC metric | classifier=SVC | repeat=2 | fold=2 | auc=0.970984314 | elapsed_seconds=126.76
AUC metric | classifier=SVC | repeat=2 | fold=3 | auc=0.975841598 | elapsed_seconds=127.39
AUC metric | classifier=SVC | repeat=2 | fold=4 | auc=0.975110670 | elapsed_seconds=125.64
AUC metric | classifier=SVC | repeat=2 | fold=5 | auc=0.972507096 | elapsed_seconds=126.11
AUC metric | classifier=SVC | repeat=3 | fold=1 | auc=0.972830217 | elapsed_seconds=126.50
AUC metric | classifier=SVC | repeat=3 | fold=2 | auc=0.976223590 | elapsed_seconds=126.00
AUC metric | classifier=SVC | repeat=3 | fold=3 | auc=0.974424650 | elapsed_seconds=126.15
AUC metric | classifier=SVC | repeat=3 | fold=4 | auc=0.971826917 | elapsed_seconds=126.88
AUC metric | classifier=SVC | repeat=3 | fold=5 | auc=0.974619637 | elapsed_seconds=127.33
AUC metric | classifier=SVC | repeat=4 | fold=1 | auc=0.975783768 | elapsed_seconds=127.15
AUC metric | classifier=SVC | repeat=4 | fold=2 | auc=0.976907481 | elapsed_seconds=129.25
AUC metric | classifier=SVC | repeat=4 | fold=3 | auc=0.981284827 | elapsed_seconds=128.02
AUC metric | classifier=SVC | repeat=4 | fold=4 | auc=0.958299077 | elapsed_seconds=124.77
AUC metric | classifier=SVC | repeat=4 | fold=5 | auc=0.971588096 | elapsed_seconds=126.12
AUC metric | classifier=SVC | repeat=5 | fold=1 | auc=0.983985285 | elapsed_seconds=128.59
AUC metric | classifier=SVC | repeat=5 | fold=2 | auc=0.973064734 | elapsed_seconds=127.05
AUC metric | classifier=SVC | repeat=5 | fold=3 | auc=0.973810429 | elapsed_seconds=125.74
AUC metric | classifier=SVC | repeat=5 | fold=4 | auc=0.971610260 | elapsed_seconds=127.34
AUC metric | classifier=SVC | repeat=5 | fold=5 | auc=0.966745870 | elapsed_seconds=126.61
AUC metric | classifier=SVC | repeat=6 | fold=1 | auc=0.965293954 | elapsed_seconds=125.35
AUC metric | classifier=SVC | repeat=6 | fold=2 | auc=0.978790600 | elapsed_seconds=128.20
AUC metric | classifier=SVC | repeat=6 | fold=3 | auc=0.968863682 | elapsed_seconds=126.43
AUC metric | classifier=SVC | repeat=6 | fold=4 | auc=0.983478795 | elapsed_seconds=120.99
AUC metric | classifier=SVC | repeat=6 | fold=5 | auc=0.970695971 | elapsed_seconds=118.84

ROC-AUC summary:
classifier           classifier_name  mean_auc   sd_auc
        LR       Logistic Regression  0.975503 0.004597
       MNB   Multinomial Naive Bayes  0.921877 0.009800
        RF             Random Forest  0.960274 0.005898
       SVC Support Vector Classifier  0.973595 0.005171

Saved raw CSV: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\roc_auc_30run_raw.csv
Saved summary JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\roc_auc_summary.json
```

### training/results/stopwords_fix_rerun/10_lr_coefficients.log

```text
# Step 10: Logistic Regression Coefficients
Training LR once on the full joint corpus.
Feature set: vectorizers + READ + OOV + SW + TRAD + SYLL + LEX + MORPH
LR C: 1.0, max_iter=2000
Rows trained on: 6412
Total coefficients: 1451835
Vectorizer coefficients: 1451788
Linguistic coefficients: 47

Linguistic coefficients:
                         feature  coefficient
     morph__prefix_derived_ratio    -0.465590
                    lex__log_ttr    -0.402674
               syll__cvc_density    -0.395495
                   lex__root_ttr    -0.337148
          trad__ave_phrase_count    -0.324248
           trad__ave_word_length    -0.298715
                        lex__ttr    -0.271151
              syll__cvcc_density    -0.260007
                   lex__corr_ttr    -0.238400
                syll__cv_density    -0.211268
         syll__consonant_cluster    -0.184323
                 syll__v_density    -0.118809
trad__ave_syllable_count_of_word    -0.118809
                syll__vc_density    -0.111141
                lex__compound_tr    -0.109257
               syll__vcc_density    -0.102489
morph__total_affix_derived_ratio    -0.102378
    morph__participle_verb_ratio    -0.096633
    morph__perfective_verb_ratio    -0.093181
             sw__count_stopwords    -0.063949
   trad__word_count_per_sentence    -0.063489
            trad__polysyll_count    -0.052793
                 lex__foreign_tr    -0.037724
  morph__total_affix_token_ratio    -0.027490
 morph__contemplative_verb_ratio    -0.025192
       morph__suffix_token_ratio    -0.017891
       morph__prefix_token_ratio    -0.009599
       morph__object_focus_ratio    -0.007816
  morph__referential_focus_ratio    -0.003299
             syll__ccvcc_density    -0.002273
           morph__aux_verb_ratio    -0.001999
     morph__locative_focus_ratio    -0.000660
   morph__recent_past_verb_ratio     0.000000
 morph__instrumental_focus_ratio     0.000000
  morph__benefactive_focus_ratio     0.002848
                trad__word_count     0.014996
            syll__ccvccc_density     0.018048
  morph__imperfective_verb_ratio     0.021741
                    lex__verb_tr     0.036398
            oov__count_oov_words     0.045380
            trad__sentence_count     0.061279
    morph__infinitive_verb_ratio     0.072211
        morph__actor_focus_ratio     0.097601
         read__readability_score     0.151202
     morph__suffix_derived_ratio     0.363213
                    lex__noun_tr     0.465511
            lex__lexical_density     0.500297

Top 20 vectorizer predictors for Fake (most negative):
                        feature  coefficient
        vectorizers__bow__upang    -0.959123
       vectorizers__bow__ngunit    -0.905210
        vectorizers__bow__noong    -0.773495
       vectorizers__bow__sinabi    -0.769751
       vectorizers__bow__dakong    -0.761375
  vectorizers__bow__nagsasabing    -0.758867
    vectorizers__bow__ferdinand    -0.680611
    vectorizers__bow__idinagdag    -0.655430
        vectorizers__bow__enero    -0.645895
    vectorizers__bow__kumakalat    -0.615180
      vectorizers__bow__youtube    -0.592263
         vectorizers__bow__2015    -0.584748
           vectorizers__bow__eh    -0.565660
       vectorizers__bow__ibalik    -0.546912
    vectorizers__bow__nakalagay    -0.545942
vectorizers__bow__entertainment    -0.542803
      vectorizers__bow__malapit    -0.539135
         vectorizers__bow__supt    -0.534094
         vectorizers__bow__2014    -0.527771
          vectorizers__bow__bay    -0.518277

Top 20 vectorizer predictors for Real (most positive):
                      feature  coefficient
     vectorizers__bow__source     2.487863
      vectorizers__bow__below     1.411136
        vectorizers__bow__gma     1.339239
vectorizers__bow__philippines     1.252139
        vectorizers__bow__frj     1.167990
       vectorizers__bow__news     1.049512
        vectorizers__bow__ofw     1.011598
   vectorizers__bow__panoorin     0.976942
   vectorizers__bow__kaniyang     0.946027
        vectorizers__bow__ptv     0.927673
     vectorizers__bow__manila     0.926739
 vectorizers__bow__integrated     0.912200
     vectorizers__bow__tanung     0.850617
    vectorizers__bow__general     0.841902
       vectorizers__bow__full     0.832299
  vectorizers__bow__inaasahan     0.830574
    vectorizers__bow__basahin     0.810706
  vectorizers__bow__pangulong     0.785298
       vectorizers__bow__2023     0.742988
  vectorizers__bow__kababayan     0.723906

Saved JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\lr_coefficients_full.json
Saved linguistic CSV: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\lr_linguistic_coefficients_full.csv
Saved vectorizer CSV: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\lr_vectorizer_coefficients_full.csv
```

### training/results/stopwords_fix_rerun/11_misclassification.log

```text
# Step 11: Misclassification Analysis
Model: retrained deployment LR without LEX and MORPH
Deployment model not found; training raw-text deployment LR for analysis.

Classification outcomes:
            article_id  gold_label  predicted_label previous_outcome new_outcome  classification_changed
false_positive_article           0                0               FP          TP                    True
 true_positive_article           0                0               TP          TP                   False
false_negative_article           1                0               FN          FN                   False
 true_negative_article           1                1               TN          TN                   False

Linguistic feature values:
            article_id  word_count  sentence_count  polysyll_count  ave_word_length  ave_phrase_count  ave_syllable_count_of_word  word_count_per_sentence  consonant_cluster  v_density  cv_density  vc_density  cvc_density  vcc_density  cvcc_density  ccvcc_density  ccvccc_density  count_oov_words  count_stopwords  readability_score
false_positive_article         129               8               4         5.186047          2.125000                    2.131783                16.125000           0.596899   2.131783    1.806202    1.596899     0.930233     0.480620      0.348837       0.062016        0.000000               16               44          14.028000
 true_positive_article         132               9               5         5.371212          3.000000                    2.098485                14.666667           0.719697   2.098485    1.795455    1.659091     1.022727     0.553030      0.431818       0.075758        0.015152               24               40          13.937667
false_negative_article         110               5               6         5.472727          1.600000                    2.118182                22.000000           0.881818   2.118182    1.763636    1.554545     0.972727     0.718182      0.490909       0.136364        0.009091                8               47          13.659000
 true_negative_article         142               9               5         4.845070          1.777778                    1.978873                15.777778           0.542254   1.978873    1.640845    1.577465     0.908451     0.443662      0.295775       0.028169        0.000000               18               56          15.308778

Active vectorizer predictors, four lowest and four highest coefficients per article:
            article_id direction         feature  coefficient
false_positive_article    lowest       bow__supt    -0.518008
false_positive_article    lowest        bow__iba    -0.463612
false_positive_article    lowest   bow__director    -0.358003
false_positive_article    lowest     bow__senior    -0.307914
false_positive_article   highest       bow__ulat     0.338630
false_positive_article   highest    bow__pumasok     0.359647
false_positive_article   highest    bow__reklamo     0.387851
false_positive_article   highest     bow__office     0.540150
 true_positive_article    lowest      bow__noong    -0.785753
 true_positive_article    lowest       bow__supt    -0.518008
 true_positive_article    lowest        bow__iba    -0.463612
 true_positive_article    lowest       bow__sina    -0.360117
 true_positive_article   highest     bow__suspek     0.342199
 true_positive_article   highest   bow__madaling     0.410671
 true_positive_article   highest     bow__driver     0.415596
 true_positive_article   highest     bow__office     0.540150
false_negative_article    lowest      bow__noong    -0.785753
false_negative_article    lowest      bow__board    -0.389138
false_negative_article    lowest      bow__about    -0.350795
false_negative_article    lowest    bow__kanyang    -0.305296
false_negative_article   highest       bow__taon     0.313104
false_negative_article   highest       bow__para     0.317631
false_negative_article   highest bow__pagbibigay     0.399480
false_negative_article   highest       bow__2018     0.450075
 true_negative_article    lowest       bow__mayo    -0.341020
 true_negative_article    lowest         bow__de    -0.288649
 true_negative_article    lowest      bow__umaga    -0.276231
 true_negative_article    lowest        bow__cbn    -0.272249
 true_negative_article   highest      bow__patay     0.429375
 true_negative_article   highest      bow__anyos     0.502501
 true_negative_article   highest   bow__kaniyang     0.932441
 true_negative_article   highest       bow__news     1.113835

Saved JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\misclassification_analysis.json
```

### training/results/stopwords_fix_rerun/12_deployment_model.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
```

### training/results/stopwords_fix_rerun/12_deployment_model.log

```text
# Step 12: Deployment Logistic Regression Model
Feature set: raw text vectorizers + READ + OOV + SW + TRAD + SYLL
LR C: 1.0, max_iter=2000

Deployment LR 30-run CV
Fold metric | repeat=1 | fold=1 | accuracy=0.920077973 | elapsed_seconds=306.69
Fold metric | repeat=1 | fold=2 | accuracy=0.906432749 | elapsed_seconds=326.98
Fold metric | repeat=1 | fold=3 | accuracy=0.930799220 | elapsed_seconds=434.45
Fold metric | repeat=1 | fold=4 | accuracy=0.929824561 | elapsed_seconds=306.87
Fold metric | repeat=1 | fold=5 | accuracy=0.926829268 | elapsed_seconds=265.01
Fold metric | repeat=2 | fold=1 | accuracy=0.923976608 | elapsed_seconds=290.66
Fold metric | repeat=2 | fold=2 | accuracy=0.918128655 | elapsed_seconds=294.10
Fold metric | repeat=2 | fold=3 | accuracy=0.928849903 | elapsed_seconds=253.64
Fold metric | repeat=2 | fold=4 | accuracy=0.927875244 | elapsed_seconds=307.02
Fold metric | repeat=2 | fold=5 | accuracy=0.920975610 | elapsed_seconds=301.55
Fold metric | repeat=3 | fold=1 | accuracy=0.925925926 | elapsed_seconds=289.95
Fold metric | repeat=3 | fold=2 | accuracy=0.921052632 | elapsed_seconds=439.90
Fold metric | repeat=3 | fold=3 | accuracy=0.928849903 | elapsed_seconds=306.76
Fold metric | repeat=3 | fold=4 | accuracy=0.923001949 | elapsed_seconds=318.19
Fold metric | repeat=3 | fold=5 | accuracy=0.915121951 | elapsed_seconds=293.00
Fold metric | repeat=4 | fold=1 | accuracy=0.927875244 | elapsed_seconds=282.74
Fold metric | repeat=4 | fold=2 | accuracy=0.924951267 | elapsed_seconds=239.87
Fold metric | repeat=4 | fold=3 | accuracy=0.938596491 | elapsed_seconds=277.96
Fold metric | repeat=4 | fold=4 | accuracy=0.905458090 | elapsed_seconds=289.66
Fold metric | repeat=4 | fold=5 | accuracy=0.920000000 | elapsed_seconds=428.57
Fold metric | repeat=5 | fold=1 | accuracy=0.946393762 | elapsed_seconds=424.63
Fold metric | repeat=5 | fold=2 | accuracy=0.929824561 | elapsed_seconds=312.41
Fold metric | repeat=5 | fold=3 | accuracy=0.919103314 | elapsed_seconds=291.56
Fold metric | repeat=5 | fold=4 | accuracy=0.918128655 | elapsed_seconds=264.58
Fold metric | repeat=5 | fold=5 | accuracy=0.911219512 | elapsed_seconds=285.31
Fold metric | repeat=6 | fold=1 | accuracy=0.922027290 | elapsed_seconds=433.73
Fold metric | repeat=6 | fold=2 | accuracy=0.929824561 | elapsed_seconds=258.16
Fold metric | repeat=6 | fold=3 | accuracy=0.919103314 | elapsed_seconds=287.12
Fold metric | repeat=6 | fold=4 | accuracy=0.931773879 | elapsed_seconds=276.80
Fold metric | repeat=6 | fold=5 | accuracy=0.918048780 | elapsed_seconds=249.68

Deployment CV summary:
count    30.000000
mean      0.923668
std       0.008531
min       0.905458
25%       0.919103
50%       0.923489
75%       0.928850
max       0.946394

Saved local deployment model: D:\Creative Corner\Projects\Software\Fake\training\models\LogisticRegression_stopwords_fix.pkl
Copied deployment model to server path: D:\Creative Corner\Projects\Software\Fake\server\root\models\LogisticRegression.pkl
Model file size bytes: 69243399
Model file size MB: 66.036
Parameter count: 1451808

Local inference benchmark:
Benchmark request | length=Short article (~50 words) | request=1 | prediction=0 | elapsed_ms=16.737
Benchmark request | length=Short article (~50 words) | request=2 | prediction=0 | elapsed_ms=15.820
Benchmark request | length=Short article (~50 words) | request=3 | prediction=0 | elapsed_ms=16.896
Benchmark request | length=Short article (~50 words) | request=4 | prediction=0 | elapsed_ms=16.842
Benchmark request | length=Short article (~50 words) | request=5 | prediction=0 | elapsed_ms=16.854
Benchmark request | length=Short article (~50 words) | request=6 | prediction=0 | elapsed_ms=16.045
Benchmark request | length=Short article (~50 words) | request=7 | prediction=0 | elapsed_ms=16.776
Benchmark request | length=Short article (~50 words) | request=8 | prediction=0 | elapsed_ms=15.236
Benchmark request | length=Short article (~50 words) | request=9 | prediction=0 | elapsed_ms=16.264
Benchmark request | length=Short article (~50 words) | request=10 | prediction=0 | elapsed_ms=16.437
Benchmark request | length=Medium article (~100 words) | request=1 | prediction=0 | elapsed_ms=19.235
Benchmark request | length=Medium article (~100 words) | request=2 | prediction=0 | elapsed_ms=17.747
Benchmark request | length=Medium article (~100 words) | request=3 | prediction=0 | elapsed_ms=17.475
Benchmark request | length=Medium article (~100 words) | request=4 | prediction=0 | elapsed_ms=17.397
Benchmark request | length=Medium article (~100 words) | request=5 | prediction=0 | elapsed_ms=18.005
Benchmark request | length=Medium article (~100 words) | request=6 | prediction=0 | elapsed_ms=17.475
Benchmark request | length=Medium article (~100 words) | request=7 | prediction=0 | elapsed_ms=17.456
Benchmark request | length=Medium article (~100 words) | request=8 | prediction=0 | elapsed_ms=17.303
Benchmark request | length=Medium article (~100 words) | request=9 | prediction=0 | elapsed_ms=17.514
Benchmark request | length=Medium article (~100 words) | request=10 | prediction=0 | elapsed_ms=17.848
Benchmark request | length=Long article (~200 words) | request=1 | prediction=0 | elapsed_ms=17.733
Benchmark request | length=Long article (~200 words) | request=2 | prediction=0 | elapsed_ms=17.701
Benchmark request | length=Long article (~200 words) | request=3 | prediction=0 | elapsed_ms=17.861
Benchmark request | length=Long article (~200 words) | request=4 | prediction=0 | elapsed_ms=17.903
Benchmark request | length=Long article (~200 words) | request=5 | prediction=0 | elapsed_ms=17.926
Benchmark request | length=Long article (~200 words) | request=6 | prediction=0 | elapsed_ms=17.952
Benchmark request | length=Long article (~200 words) | request=7 | prediction=0 | elapsed_ms=17.537
Benchmark request | length=Long article (~200 words) | request=8 | prediction=0 | elapsed_ms=18.096
Benchmark request | length=Long article (~200 words) | request=9 | prediction=0 | elapsed_ms=18.276
Benchmark request | length=Long article (~200 words) | request=10 | prediction=0 | elapsed_ms=17.250
             article_length   mean_ms  median_ms    sd_ms
  Short article (~50 words) 16.390760   16.58695 0.551675
Medium article (~100 words) 17.745540   17.49450 0.567549
  Long article (~200 words) 17.823560   17.88215 0.288465
                    Overall 17.319953   17.47510 0.817238

Saved JSON: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\deployment_model.json
```

### training/results/stopwords_fix_rerun/12_deployment_model_attempt1.err.log

```text
C:\Program Files\Python311\python.exe: can't open file 'D:\\Creative': [Errno 2] No such file or directory
```

### training/results/stopwords_fix_rerun/12_deployment_model_attempt1.log

```text

```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\linear_model\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation.log

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

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt1.err.log

```text
Traceback (most recent call last):
  File "D:\Creative Corner\Projects\Software\Fake\training\crossval_ablation_tuned.py", line 34, in <module>
    from root.scripts.FILTRANS import (
  File "D:\Creative Corner\Projects\Software\Fake\training\root\scripts\FILTRANS.py", line 5, in <module>
    import root.scripts.LEX as LEX
  File "D:\Creative Corner\Projects\Software\Fake\training\root\scripts\LEX.py", line 122, in <module>
    pos_tagger=StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx60G")   # Change -Xmx4G to -XmxYG as needed where Y is the heap size in Gigabytes
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\nltk\tag\stanford.py", line 159, in __init__
    super().__init__(*args, **kwargs)
  File "D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\nltk\tag\stanford.py", line 70, in __init__
    self._stanford_jar = find_jar(
                         ^^^^^^^^^
  File "D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\nltk\internals.py", line 833, in find_jar
    return next(
           ^^^^^
  File "D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\nltk\internals.py", line 719, in find_jar_iter
    raise LookupError(
LookupError: Could not find stanford-postagger.jar jar file at ./root/runtime_env/stanford-postagger-full-2020-11-17/stanford-postagger.jar
```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt1.log

```text

```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt2_interrupted.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt2_interrupted.log

```text
Tuned Feature Ablation Results (2026-06-04 04:04:01)
Output directory: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\13_tuned_feature_ablation
Raw CSV: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\13_tuned_feature_ablation\tuned_feature_ablation_raw.csv
Console log: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\13_tuned_feature_ablation\tuned_feature_ablation_console.log
Protocol: train_test_split(test_size=0.2, stratify=y, random_state=42), then RepeatedKFold(n_splits=5, n_repeats=6, random_state=42).
Tuned classifiers: MNB(alpha=0.1); LR(C=1.0, max_iter=2000); RF(n_estimators=100, max_depth=20, min_samples_split=2); SVC(C=0.1, kernel=linear).
Using existing Step 0 stop-word caches; no feature CSVs are regenerated here.

Dataset loaded | key=Cruz | name=Fake News Filipino 2020 | rows=3206 | training_rows=2564

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=3 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.902343750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=1 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=1 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=2 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=3 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.902343750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=2 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=2 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=3 | accuracy=0.966861598
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.916015625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=3 | fold=5 | accuracy=0.931640625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=3 | fold=5 | accuracy=0.921875000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.886939571
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=1 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=4 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.917968750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=4 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=4 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=5 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=5 | fold=5 | accuracy=0.912109375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=1 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.886939571
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=3 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=4 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.886718750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=LR | repeat=6 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=RF | repeat=6 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=Vectorizers (TF-IDF + BOW) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.949218750
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.903992167
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.947543682
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.924338552
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.948713907

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=1 | accuracy=0.964912281
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=3 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=4 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.896484375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=1 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=1 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.949218750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=2 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=3 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=4 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.902343750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=2 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=2 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=2 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.896686160
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=4 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.916015625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=3 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=3 | fold=5 | accuracy=0.910156250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=3 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=4 | accuracy=0.920077973
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.917968750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=4 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=4 | fold=5 | accuracy=0.919921875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=1 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=2 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=3 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=5 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=5 | fold=5 | accuracy=0.917968750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=2 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=3 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=3 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=4 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.884765625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=LR | repeat=6 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=RF | repeat=6 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Readability (READ) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.947265625
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.902692114
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.946893402
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.922192526
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.949038286

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=2 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=3 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.894531250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=1 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=1 | fold=5 | accuracy=0.935546875
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=1 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=2 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=3 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=4 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.906250000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=2 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=2 | fold=5 | accuracy=0.937500000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=1 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=2 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=2 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.898635478
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.914062500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=3 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=3 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=1 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=2 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=2 | accuracy=0.922027290
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=4 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.921875000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=4 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=4 | fold=5 | accuracy=0.927734375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.886939571
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=1 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=2 | accuracy=0.925925926
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=3 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.912109375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=5 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=5 | fold=5 | accuracy=0.912109375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=1 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=2 | accuracy=0.910331384
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=3 | accuracy=0.898635478
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.880859375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=LR | repeat=6 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=RF | repeat=6 | fold=5 | accuracy=0.945312500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Out-of-vocabulary (OOV) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.951171875
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.902756965
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.949493761
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.924404544
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.949688566

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifiers=MNB,LR,RF,SVC
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=1 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=1 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=1 | accuracy=0.968810916
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=2 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=3 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=3 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=4 | accuracy=0.888888889
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=4 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=4 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=1 | fold=5 | accuracy=0.886718750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=1 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=1 | fold=5 | accuracy=0.941406250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=1 | fold=5 | accuracy=0.953125000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=1 | accuracy=0.900584795
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=1 | accuracy=0.923976608
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=2 | accuracy=0.896686160
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=2 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=2 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=3 | accuracy=0.957115010
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=3 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=3 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=4 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=4 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=2 | fold=5 | accuracy=0.900390625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=2 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=2 | fold=5 | accuracy=0.929687500
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=2 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=1 | accuracy=0.890838207
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=1 | accuracy=0.935672515
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=1 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=1 | accuracy=0.937621832
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=2 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=2 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=2 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=2 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=3 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=3 | accuracy=0.962962963
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=3 | accuracy=0.929824561
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=3 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=4 | accuracy=0.892787524
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=4 | accuracy=0.959064327
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=4 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=3 | fold=5 | accuracy=0.912109375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=3 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=3 | fold=5 | accuracy=0.917968750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=3 | fold=5 | accuracy=0.933593750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=1 | accuracy=0.879142300
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=1 | accuracy=0.914230019
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=2 | accuracy=0.894736842
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=2 | accuracy=0.961013645
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=2 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=3 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=3 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=4 | accuracy=0.884990253
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=4 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=4 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=4 | fold=5 | accuracy=0.916015625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=4 | fold=5 | accuracy=0.955078125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=4 | fold=5 | accuracy=0.923828125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=4 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=1 | accuracy=0.873294347
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=1 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=1 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=2 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=2 | accuracy=0.945419103
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=2 | accuracy=0.916179337
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=2 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=3 | accuracy=0.883040936
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=3 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=3 | accuracy=0.933723197
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=3 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=4 | accuracy=0.918128655
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=4 | accuracy=0.947368421
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=4 | accuracy=0.908382066
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=4 | accuracy=0.949317739
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=5 | fold=5 | accuracy=0.906250000
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=5 | fold=5 | accuracy=0.958984375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=5 | fold=5 | accuracy=0.912109375
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=5 | fold=5 | accuracy=0.957031250
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=1 | accuracy=0.904483431
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=1 | accuracy=0.955165692
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=1 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=1 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=2 | accuracy=0.927875244
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=2 | accuracy=0.941520468
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=2 | accuracy=0.912280702
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=2 | accuracy=0.943469786
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=3 | accuracy=0.865497076
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=3 | accuracy=0.931773879
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=3 | accuracy=0.896686160
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=3 | accuracy=0.939571150
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=4 | accuracy=0.902534113
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=4 | accuracy=0.953216374
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=4 | accuracy=0.906432749
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=4 | accuracy=0.951267057
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=MNB | repeat=6 | fold=5 | accuracy=0.871093750
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=LR | repeat=6 | fold=5 | accuracy=0.947265625
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=RF | repeat=6 | fold=5 | accuracy=0.939453125
Fold metric | dataset=Fake News Filipino 2020 | feature_set=+ Stop words (SW) | classifier=SVC | repeat=6 | fold=5 | accuracy=0.951171875
Mean accuracy for 6 repetitions of 5-fold cross-validation with MNB: 0.895347146
Mean accuracy for 6 repetitions of 5-fold cross-validation with LR: 0.949233852
Mean accuracy for 6 repetitions of 5-fold cross-validation with RF: 0.920959354
Mean accuracy for 6 repetitions of 5-fold cross-validation with SVC: 0.949688566

Feature-set condition | dataset=Fake News Filipino 2020 | feature_set=+ Traditional features (TRAD) | classifiers=MNB,LR,RF,SVC
```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt3_interrupted.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt3_interrupted.log

```text
Tuned Feature Ablation Results (2026-06-04 04:40:45)
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
```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt4_interrupted.err.log

```text
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
D:\Creative Corner\Projects\Software\Fake\training\venv\Lib\site-packages\sklearn\feature_extraction\text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn(
```

### training/results/stopwords_fix_rerun/13_tuned_feature_ablation_attempt4_interrupted.log

```text
Tuned Feature Ablation Results (2026-06-04 04:45:10)
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
```

### training/results/stopwords_fix_rerun/14_build_report.log

```text
Wrote D:\Creative Corner\Projects\Software\Fake\rerun_results.md
Wrote D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\rerun_results.md
```
