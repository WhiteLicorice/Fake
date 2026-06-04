# Render Benchmark and Deployment Coefficients

## Table 8 replacement (Render inference benchmark)

Endpoint: https://fph-ml.onrender.com/check-news
Warmup response time: 456.260 ms (excluded)

| Article length | Mean ms | Median ms | SD ms |
| --- | --- | --- | --- |
| Short (~50 words) | 109.616 | 101.095 | 18.462 |
| Medium (~100 words) | 139.158 | 103.462 | 51.321 |
| Long (~200 words) | 190.099 | 196.781 | 37.021 |
| Overall | 146.291 | 127.458 | 49.916 |

Note: round-trip HTTP response time because the API response did not include server-side inference time; measured from the local Codex workspace/client at 2026-06-04T16:00:40+08:00.

## Table 9 replacement (deployment model linguistic coefficients)

| Feature | Coefficient |
| --- | --- |
| trad__ave_phrase_count | -0.360418 |
| syll__cvc_density | -0.358687 |
| trad__ave_word_length | -0.320748 |
| syll__cvcc_density | -0.248347 |
| syll__consonant_cluster | -0.196940 |
| syll__cv_density | -0.153433 |
| syll__vc_density | -0.152160 |
| trad__ave_syllable_count_of_word | -0.136975 |
| syll__v_density | -0.136975 |
| trad__word_count_per_sentence | -0.128265 |
| syll__vcc_density | -0.103161 |
| sw__count_stopwords | -0.072888 |
| trad__polysyll_count | -0.053378 |
| trad__word_count | -0.002800 |
| syll__ccvcc_density | 0.002761 |
| syll__ccvccc_density | 0.020476 |
| oov__count_oov_words | 0.046274 |
| trad__sentence_count | 0.061927 |
| read__readability_score | 0.376049 |

## Table 10 replacement (deployment model top vectorizer predictors)

| Feature | Coefficient |
| --- | --- |
| bow__upang | -0.955521 |
| bow__ngunit | -0.904110 |
| bow__sinabi | -0.802423 |
| bow__source | 2.456921 |
| bow__below | 1.503337 |
| bow__gma | 1.320817 |

## Raw console output

```text
# Render Benchmark and Deployment Coefficients
Started: 2026-06-04T16:00:28+08:00
Log path: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\render_benchmark_deployment_coefficients.log
Report path: D:\Creative Corner\Projects\Software\Fake\training\results\stopwords_fix_rerun\render_benchmark_deployment_coefficients.md

Selected benchmark samples:
Sample | length=Short (~50 words) | id=previous_short | words=44 | source=training.rerun_common.BENCHMARK_ARTICLES
Sample | length=Short (~50 words) | id=short_01 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1070
Sample | length=Short (~50 words) | id=short_02 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1212
Sample | length=Short (~50 words) | id=short_03 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1374
Sample | length=Short (~50 words) | id=short_04 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 139
Sample | length=Short (~50 words) | id=short_05 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1398
Sample | length=Short (~50 words) | id=short_06 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 186
Sample | length=Short (~50 words) | id=short_07 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 340
Sample | length=Short (~50 words) | id=short_08 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 378
Sample | length=Short (~50 words) | id=short_09 | words=50 | source=Cruz/FakeNewsFilipino_Cruz2020 row 38
Sample | length=Medium (~100 words) | id=previous_medium | words=111 | source=training.rerun_common.BENCHMARK_ARTICLES
Sample | length=Medium (~100 words) | id=medium_01 | words=100 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1591
Sample | length=Medium (~100 words) | id=medium_02 | words=100 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1729
Sample | length=Medium (~100 words) | id=medium_03 | words=100 | source=Lupac/FakeNewsPhilippines2024_Lupac row 1167
Sample | length=Medium (~100 words) | id=medium_04 | words=100 | source=Lupac/FakeNewsPhilippines2024_Lupac row 1184
Sample | length=Medium (~100 words) | id=medium_05 | words=100 | source=Lupac/FakeNewsPhilippines2024_Lupac row 1245
Sample | length=Medium (~100 words) | id=medium_06 | words=100 | source=Lupac/FakeNewsPhilippines2024_Lupac row 1252
Sample | length=Medium (~100 words) | id=medium_07 | words=100 | source=Lupac/FakeNewsPhilippines2024_Lupac row 187
Sample | length=Medium (~100 words) | id=medium_08 | words=100 | source=Lupac/FakeNewsPhilippines2024_Lupac row 2078
Sample | length=Medium (~100 words) | id=medium_09 | words=100 | source=Lupac/FakeNewsPhilippines2024_Lupac row 2320
Sample | length=Long (~200 words) | id=previous_long | words=130 | source=training.rerun_common.BENCHMARK_ARTICLES
Sample | length=Long (~200 words) | id=long_01 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1364
Sample | length=Long (~200 words) | id=long_02 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1795
Sample | length=Long (~200 words) | id=long_03 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 1891
Sample | length=Long (~200 words) | id=long_04 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 2458
Sample | length=Long (~200 words) | id=long_05 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 2484
Sample | length=Long (~200 words) | id=long_06 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 2888
Sample | length=Long (~200 words) | id=long_07 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 2961
Sample | length=Long (~200 words) | id=long_08 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 3154
Sample | length=Long (~200 words) | id=long_09 | words=200 | source=Cruz/FakeNewsFilipino_Cruz2020 row 386

# Render API benchmark
Endpoint: https://fph-ml.onrender.com/check-news
Request JSON format: {"news_body": "<article text>"}
Warmup sample: previous_short (44 words)
Warmup response time: 456.260 ms (excluded)
Warmup response JSON: {'status': True}

Benchmark request | request=01 | length=Short (~50 words) | sample=previous_short | words=44 | round_trip_ms=100.601 | response={'status': True}
Benchmark request | request=02 | length=Short (~50 words) | sample=short_01 | words=50 | round_trip_ms=92.158 | response={'status': True}
Benchmark request | request=03 | length=Short (~50 words) | sample=short_02 | words=50 | round_trip_ms=101.588 | response={'status': True}
Benchmark request | request=04 | length=Short (~50 words) | sample=short_03 | words=50 | round_trip_ms=135.267 | response={'status': True}
Benchmark request | request=05 | length=Short (~50 words) | sample=short_04 | words=50 | round_trip_ms=119.650 | response={'status': True}
Benchmark request | request=06 | length=Short (~50 words) | sample=short_05 | words=50 | round_trip_ms=97.951 | response={'status': True}
Benchmark request | request=07 | length=Short (~50 words) | sample=short_06 | words=50 | round_trip_ms=145.711 | response={'status': True}
Benchmark request | request=08 | length=Short (~50 words) | sample=short_07 | words=50 | round_trip_ms=99.509 | response={'status': True}
Benchmark request | request=09 | length=Short (~50 words) | sample=short_08 | words=50 | round_trip_ms=111.793 | response={'status': True}
Benchmark request | request=10 | length=Short (~50 words) | sample=short_09 | words=50 | round_trip_ms=91.931 | response={'status': True}
Benchmark request | request=11 | length=Medium (~100 words) | sample=previous_medium | words=111 | round_trip_ms=95.718 | response={'status': True}
Benchmark request | request=12 | length=Medium (~100 words) | sample=medium_01 | words=100 | round_trip_ms=199.257 | response={'status': True}
Benchmark request | request=13 | length=Medium (~100 words) | sample=medium_02 | words=100 | round_trip_ms=100.706 | response={'status': False}
Benchmark request | request=14 | length=Medium (~100 words) | sample=medium_03 | words=100 | round_trip_ms=200.032 | response={'status': True}
Benchmark request | request=15 | length=Medium (~100 words) | sample=medium_04 | words=100 | round_trip_ms=100.743 | response={'status': True}
Benchmark request | request=16 | length=Medium (~100 words) | sample=medium_05 | words=100 | round_trip_ms=183.872 | response={'status': True}
Benchmark request | request=17 | length=Medium (~100 words) | sample=medium_06 | words=100 | round_trip_ms=209.822 | response={'status': True}
Benchmark request | request=18 | length=Medium (~100 words) | sample=medium_07 | words=100 | round_trip_ms=106.182 | response={'status': True}
Benchmark request | request=19 | length=Medium (~100 words) | sample=medium_08 | words=100 | round_trip_ms=100.388 | response={'status': False}
Benchmark request | request=20 | length=Medium (~100 words) | sample=medium_09 | words=100 | round_trip_ms=94.858 | response={'status': False}
Benchmark request | request=21 | length=Long (~200 words) | sample=previous_long | words=130 | round_trip_ms=206.647 | response={'status': True}
Benchmark request | request=22 | length=Long (~200 words) | sample=long_01 | words=200 | round_trip_ms=193.064 | response={'status': True}
Benchmark request | request=23 | length=Long (~200 words) | sample=long_02 | words=200 | round_trip_ms=202.374 | response={'status': False}
Benchmark request | request=24 | length=Long (~200 words) | sample=long_03 | words=200 | round_trip_ms=158.536 | response={'status': False}
Benchmark request | request=25 | length=Long (~200 words) | sample=long_04 | words=200 | round_trip_ms=239.922 | response={'status': False}
Benchmark request | request=26 | length=Long (~200 words) | sample=long_05 | words=200 | round_trip_ms=199.007 | response={'status': False}
Benchmark request | request=27 | length=Long (~200 words) | sample=long_06 | words=200 | round_trip_ms=174.319 | response={'status': False}
Benchmark request | request=28 | length=Long (~200 words) | sample=long_07 | words=200 | round_trip_ms=225.188 | response={'status': False}
Benchmark request | request=29 | length=Long (~200 words) | sample=long_08 | words=200 | round_trip_ms=107.374 | response={'status': False}
Benchmark request | request=30 | length=Long (~200 words) | sample=long_09 | words=200 | round_trip_ms=194.554 | response={'status': True}

Table 8 replacement (Render inference benchmark)
Article length       Mean ms  Median ms  SD ms 
-------------------  -------  ---------  ------
Short (~50 words)    109.616  101.095    18.462
Medium (~100 words)  139.158  103.462    51.321
Long (~200 words)    190.099  196.781    37.021
Overall              146.291  127.458    49.916

Measurement note: round-trip HTTP response time because the API response did not include server-side inference time

# Deployment model coefficients
Model path: D:\Creative Corner\Projects\Software\Fake\server\root\models\LogisticRegression.pkl
Pipeline steps: ['features', 'classifier']
FeatureUnion transformers: ['tfidf', 'bow', 'read', 'oov', 'sw', 'trad', 'syll']
Total coefficients: 1451807
Vectorizer coefficients: 1451788
Linguistic coefficients: 19

Table 9 replacement (deployment model linguistic coefficients)
Feature                           Coefficient
--------------------------------  -----------
trad__ave_phrase_count            -0.360418  
syll__cvc_density                 -0.358687  
trad__ave_word_length             -0.320748  
syll__cvcc_density                -0.248347  
syll__consonant_cluster           -0.196940  
syll__cv_density                  -0.153433  
syll__vc_density                  -0.152160  
trad__ave_syllable_count_of_word  -0.136975  
syll__v_density                   -0.136975  
trad__word_count_per_sentence     -0.128265  
syll__vcc_density                 -0.103161  
sw__count_stopwords               -0.072888  
trad__polysyll_count              -0.053378  
trad__word_count                  -0.002800  
syll__ccvcc_density               0.002761   
syll__ccvccc_density              0.020476   
oov__count_oov_words              0.046274   
trad__sentence_count              0.061927   
read__readability_score           0.376049   

Table 10 replacement (deployment model top vectorizer predictors)
Feature      Coefficient
-----------  -----------
bow__upang   -0.955521  
bow__ngunit  -0.904110  
bow__sinabi  -0.802423  
bow__source  2.456921   
bow__below   1.503337   
bow__gma     1.320817
```
