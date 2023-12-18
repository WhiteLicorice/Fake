#   Filipino linguistic feature extractors adapted from: https://github.com/imperialite/filipino-linguistic-extractors
from root.scripts import LM as LM
from root.scripts import SYLL as SYLL
from root.scripts import TRAD as TRAD

from sklearn.base import BaseEstimator, TransformerMixin

# import string

#   Custom transformer for TRAD feature extraction
class TRADExtractor(BaseEstimator, TransformerMixin):
    def word_count_per_doc(self, text):
        return TRAD.word_count_per_doc(text)
    
    def sentence_count_per_doc(self, text):
        return TRAD.sentence_count_per_doc(text)
    
    def polysyll_count_per_doc(self, text):
        return TRAD.polysyll_count_per_doc(text)
    
    def ave_word_length(self, text):
        return TRAD.ave_word_length(text)
    
    def ave_phrase_count_per_doc(self, text):
        return TRAD.ave_phrase_count_per_doc(text)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for doc in X:
            word_count = self.word_count_per_doc(doc)
            sentence_count = self.sentence_count_per_doc(doc)
            polysyll_count = self.polysyll_count_per_doc(doc)
            ave_word_length = self.ave_word_length(doc)
            ave_phrase_count = self.ave_phrase_count_per_doc(doc)
            features.append([word_count, sentence_count, polysyll_count, ave_word_length, ave_phrase_count])
        return features

#   Custom transformer for SYLL feature extraction
class SYLLExtractor(BaseEstimator, TransformerMixin):
    def get_cc_cluster(self, text):
        return SYLL.get_consonant_cluster(text)
    
    def get_v_density(self, text):
        return SYLL.get_v(text)

    def get_cv_density(self, text):
        return SYLL.get_cv(text)

    def get_vc_density(self, text):
        return SYLL.get_vc(text)

    def get_cvc_density(self, text):
        return SYLL.get_cvc(text)

    def get_vcc_density(self, text):
        return SYLL.get_vcc(text)

    def get_cvcc_density(self, text):
        return SYLL.get_cvcc(text)

    def get_ccvcc_density(self, text):
        return SYLL.get_ccvcc(text)

    def get_ccvccc_density(self, text):
        return SYLL.get_ccvccc(text)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for doc in X:
            cc_cluster = self.get_cc_cluster(doc)
            v_density = self.get_v_density(doc)
            cv_density = self.get_cv_density(doc)
            vc_density = self.get_vc_density(doc)
            cvc_density = self.get_cvc_density(doc)
            vcc_density = self.get_vcc_density(doc)
            cvcc_density = self.get_cvcc_density(doc)
            ccvcc_density = self.get_ccvcc_density(doc)
            ccvccc_density = self.get_ccvccc_density(doc)
            features.append([
                cc_cluster, v_density, cv_density, vc_density, cvc_density,
                vcc_density, cvcc_density, ccvcc_density, ccvccc_density
            ])
        return features


# #   Custom transformer for pre-processing text, not used since we do not preprocess the text to preserve mispelled words, spaces, punctuations, etc.
# class TextPreprocessor(BaseEstimator, TransformerMixin):
#     def __init__(self, stop_words_en=None, stop_words_tl=None, stemmer_en=None, stemmer_tl=None):
#         #   Load English and Filipino stopwords adapted from: https://github.com/stopwords-iso/stopwords-iso
#         self.stop_words_en = stop_words_en or set()
#         self.stop_words_tl = stop_words_tl or set()
#         self.stemmer_en = stemmer_en if stemmer_en else PorterStemmer()
#         self.stemmer_tl = stemmer_tl if stemmer_tl else TGLStem
#         #self.tokenizer = tokenizer if tokenizer else Tokenizer()
        
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         processed_text = []
#         for text in X:
#             # Convert to lowercase
#             text = text.lower()

#             # Remove punctuation
#             text = text.translate(str.maketrans('', '', string.punctuation))

#             # Tokenize
#             #words = self.tokenizer.encode(text)

#             # Remove stop words
#             words = [word for word in words if word not in self.stop_words_en and word not in self.stop_words_tl]

#             # Apply stemming
#             words = [self.stemmer_en.stem(word) for word in words]
#             words = [self.stemmer_tl.stemmer(word) for word in words]

#             # Rejoin the words into a single string
#             processed_text.append(' '.join(words))

#         return processed_text