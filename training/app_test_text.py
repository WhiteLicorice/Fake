import root.scripts.SW as SW
import root.scripts.OOV as OOV
import root.scripts.READ as READ
import root.scripts.SYLL as SYLL
import root.scripts.TRAD as TRAD
from root.scripts.BPE import BPETokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

texts = [
"Mahaharap sa kasong administratibo ang isang opisyal ng pulisya matapos magwala sa mismong himpilan, pinasok sa opisina ang kanyang hepe at pinagsasalitaan umano ng masama, Lunes ng gabi, sa Bacoor City, Cavite. Kasong grave misconduct ang kakaharapin ni Chief Insp. Virgilio Rubio, deputy chief ng Bacoor City Police, batay sa reklamo ni Supt. Rommel Estolano. Sa ulat sa tanggapan ni Cavite Police Provincial Office director Senior Supt. Joselito Esquivel, bandang 7:30 ng gabi nang magtungo sa istasyon ng pulisya si Rubio na lasing na lasing, biglang pinaghahagis ang mga upuan at iba pang gamit sa opisina hanggang pumasok sa tanggapan ni Rubio habang may hawak na baril at nagbitiw ng kung anu-anong masasamang salita. Gayunman, naawat ni Senior Insp. Chey Chey Saulog si Rubio at pinalabas sa himpilan ng pulisya",
"Binigyan ng tatlong taong extension para sa kanyang panunungkulan bilang. Commissioner ng Philippine Basketball Association si Willie Marcial. Sa kanilang annual planning session sa Star Hotels sa bansang Italya, gaya ng naging pagkakatalaga sa kanya bilang Commissioner ng liga, naging unanimous ang pagbibigay ng board of governors ng extension sa termino ni Marcial kahapon (Huwebes). May nalalabi pang isang taon sa naunang tatlong taong kontrata na nilagdaan ni Marcial noong 2018 pero binigyan sya ng PBA board ng bagong vote of confidence. Ito'y bunga na rin ng magandang performance nito na nagustuhan ng board.“We're open and very transparent about his performance,” wika ni PBA Chairman Ricky Vargas tungkol kay Marcial."
]

# file1 = open("./results/browserTesting/vocab.txt", "w", encoding="utf-8")
# list_of_vals = [ ]
bow = CountVectorizer()
tfid = TfidfVectorizer(ngram_range=(1,3),tokenizer=BPETokenizer().tokenize)
bow_mat = bow.fit_transform(texts)
tfid_mat = tfid.fit_transform(texts)

# for i in texts:
#     # word_count_per_doc = TRAD.word_count_per_doc(i)
#     # sentence_count_per_doc = TRAD.sentence_count_per_doc(i)
#     # polysyll_count_per_doc = TRAD.polysyll_count_per_doc(i)
#     # ave_word_length = TRAD.ave_word_length(i)
#     # ave_phrase_count_per_doc = TRAD.ave_phrase_count_per_doc(i)
#     # ave_syllable_count_of_word = TRAD.ave_syllable_count_of_word(i)
#     # word_count_per_sentence = TRAD.word_count_per_sentence(i)
#     # consonant_cluster = SYLL.get_consonant_cluster(i)
#     # v_density = SYLL.get_v(i)
#     # cv_density = SYLL.get_cv(i)
#     # vc_density = SYLL.get_vc(i)
#     # cvc_density = SYLL.get_cvc(i)
#     # vcc_density = SYLL.get_vcc(i)
#     # cvcc_density = SYLL.get_cvcc(i)
#     # ccvcc_density = SYLL.get_ccvcc(i)
#     # ccvccc_density = SYLL.get_ccvccc(i)
#     # count_stopwords = SW.count_stopwords(i)
#     # readability_score = READ.compute_readability_score(i)
#     # count_oov_words = OOV.count_oov_words(i)
#     bag_of_words = bow.fit_transform([i])
#     tfidf_mat = tfid.fit_transform([i])


#     file1.write("BOW\n\n")

#     for j,k in enumerate(bow.vocabulary_):
#         file1.write(f"{k}\n")
#     file1.write("\n\nTFIDF\n\n")
#     for j,k in enumerate(tfid.vocabulary_):
#         file1.write(f"{k}\n")

#     bow = CountVectorizer()
#     tfid = TfidfVectorizer(ngram_range=(1,3),tokenizer=BPETokenizer().tokenize)

    # list_of_vals.append([
    #     word_count_per_doc,
    #     sentence_count_per_doc,
    #     polysyll_count_per_doc,
    #     ave_word_length,
    #     ave_phrase_count_per_doc,
    #     ave_syllable_count_of_word,
    #     word_count_per_sentence,
    #     consonant_cluster,
    #     v_density,
    #     cv_density,
    #     vc_density,
    #     cvc_density,
    #     vcc_density,
    #     cvcc_density,
    #     ccvcc_density,
    #     ccvccc_density,
    #     count_stopwords,
    #     readability_score,
    #     count_oov_words,
    # ])

    # list_of_bow.append(bag_of_words)
    # list_of_tfidf.append(tfidf_mat)

# names = [
#     'word_count_per_doc', 'sentence_count_per_doc', 'polysyll_count_per_doc', 'ave_word_length', 'ave_phrase_count_per_doc',
#     'ave_syllable_count_of_word', 'word_count_per_sentence', 'consonant_cluster', 'v_density', 'cv_density', 'vc_density', 'cvc_density', 'vcc_density', 'cvcc_density', 'ccvcc_density', 'ccvccc_density', 'count_stopwords', 'readability_score', 'count_oov_words'
# ]

# file1.close

# print(f"PREDICTORS\t\t\t   FALSE POS  \tFALSE NEGA")
# for i in range(len(names)):
#     print(f"{names[i]:<35s}{list_of_vals[0][i]:<3f}\t{list_of_vals[1][i]:<3f}")


# for i in range(0):
#     print("")
#     print(bow_mat[0])
#     print(bow.vocabulary_)
#     print("")
    # print(list_of_tfidf[i])

# print(bow_mat[0])
# print(bow_mat[1])
# print(bow.vocabulary_)

# print("")
# print(tfid_mat[0])
# print(tfid_mat[1])
# print(tfid.vocabulary_)

# for i in bow.vocabulary_:
#     print(i)