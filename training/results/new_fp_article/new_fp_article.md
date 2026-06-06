# New False Positive Article

This article is freshly composed for deployment-model probing and is intentionally fabricated. It should replace Article 1 only in the deployment-test appendix.

## Selected Filipino Text

MAYNILA - Inihayag umano ng isang source sa Department of Health na pansamantalang ipapatupad ang libreng checkup program sa lahat ng barangay health center simula Lunes. Ayon sa kumalat na abiso, maaaring magtungo ang mga residente kahit walang appointment basta magdala ng valid ID at kopya ng vaccination card. Sinabi rin sa post na prayoridad ang mga senior citizen, buntis, at estudyante na may lagnat o ubo. Nilinaw umano ng source na bahagi ito ng bagong kampanya laban sa mga sakit na nauuso tuwing tag-ulan. Wala pang opisyal na memorandum na inilalabas sa website ng ahensiya, ngunit mabilis nang ibinahagi ang anunsiyo sa ilang community group. Ilang health worker ang nagsabing wala silang natatanggap na kautusan tungkol dito.

## English Translation

MANILA - A source from the Department of Health allegedly announced that a temporary free checkup program would be implemented in all barangay health centers starting Monday. According to the circulating notice, residents could go even without an appointment as long as they brought a valid ID and a copy of their vaccination card. The post also said that senior citizens, pregnant women, and students with fever or cough would be prioritized. The source allegedly clarified that this was part of a new campaign against illnesses common during the rainy season. No official memorandum had been posted on the agency website, but the announcement had already been shared quickly in community groups. Several health workers said they had received no order about it.

## Classification Result

- Candidate: cand_01 (DOH free barangay checkups)
- Word count: 118
- Gold label: Fake (0)
- Predicted label: Real (1)
- Probability Fake: 0.239879
- Probability Real: 0.760121
- Candidate batch false positives: 3 of 25

Existing Article 3 remains a false negative:
- Gold label: Real (1)
- Predicted label: Fake (0)
- Probability Fake: 0.978397
- Probability Real: 0.021603

## Table 13 Replacement - Linguistic Predictors

| Predictor | False Positive (new Article 1) | False Negative (Article 3) |
| --- | --- | --- |
| ave-phrase-count | 1.666667 | 1.6 |
| ave-word-length | 5.213675 | 5.472727 |
| word-count-per-sentence | 19.5 | 22 |
| polysyll-count | 5 | 6 |
| word-count | 117 | 110 |
| sentence-count | 6 | 5 |
| ave-syllable-count-of-word | 2.119658 | 2.118182 |
| cvc-density | 0.948718 | 0.972727 |
| consonant-cluster | 0.666667 | 0.881818 |
| cvcc-density | 0.393162 | 0.490909 |
| vcc-density | 0.538462 | 0.718182 |
| vc-density | 1.606838 | 1.554545 |
| v-density | 2.119658 | 2.118182 |
| cv-density | 1.760684 | 1.763636 |
| ccvcc-density | 0.068376 | 0.136364 |
| ccvccc-density | 0 | 0.009091 |
| readability-score | 13.701 | 13.659 |
| count-oov-words | 0 | 8 |
| count-stopwords | 45 | 47 |

## Table 14 Replacement - Active Vectorizer Predictors

Rows are selected by the largest signed active contribution `coefficient * feature_value` among non-zero TF-IDF/BOW features.

| Article | Direction | Predictor | Coefficient | Value | Impact |
| --- | --- | --- | --- | --- | --- |
| False Positive | Fake | vectorizers--bow--ngunit | -0.90411 | 1 | -0.90411 |
| False Positive | Fake | vectorizers--bow--sinabi | -0.802423 | 1 | -0.802423 |
| False Positive | Real | vectorizers--bow--source | 2.456921 | 2 | 4.913842 |
| False Positive | Real | vectorizers--bow--sa | 0.102224 | 7 | 0.71557 |
| False Negative | Fake | vectorizers--bow--board | -0.389138 | 3 | -1.167415 |
| False Negative | Fake | vectorizers--bow--noong | -0.785753 | 1 | -0.785753 |
| False Negative | Real | vectorizers--bow--sa | 0.102224 | 7 | 0.71557 |
| False Negative | Real | vectorizers--bow--marcial | 0.149055 | 4 | 0.59622 |

## Analytical Note

The selected fabricated article is classified as Real mainly because it uses a cautious journalistic register, has relatively low values on several negative-weight linguistic predictors such as ave-phrase-count (1.666667) and consonant-cluster (0.666667), and activates real-leaning vectorizer terms such as vectorizers--bow--source, vectorizers--bow--sa. Its active fake-leaning terms, including vectorizers--bow--ngunit, vectorizers--bow--sinabi, do not offset the positive evidence enough, leaving the model at a Real probability of 0.760121. This is a useful blind spot for the manuscript because the article is not an obvious keyword-stuffed attack; it reads like a cautious local news brief built around attribution, agency references, and a still-unconfirmed advisory.
