#   Filipino linguistic feature extractors (TRAD, SYLL, LM, LEX, MORPH) adapted from: https://github.com/imperialite/filipino-linguistic-extractors
import root.scripts.LM as LM
import root.scripts.SYLL as SYLL
import root.scripts.TRAD as TRAD
import root.scripts.LEX as LEX
import root.scripts.MORPH as MORPH

import root.scripts.OOV as OOV
import root.scripts.SW as SW

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
    
class LEXExtractor(BaseEstimator, TransformerMixin):
    def get_ttr(self, text):
        return LEX.type_token_ratio(text)
    
    def get_root_ttr(self, text):
        return LEX.root_type_token_ratio(text)
    
    def get_corr_ttr(self, text):
        return LEX.corr_type_token_ratio(text)
    
    def get_log_ttr(self, text):
        return LEX.log_type_token_ratio(text)
    
    def get_noun_tr(self, text):
        return LEX.noun_token_ratio(text)
    
    def get_verb_tr(self, text):
        return LEX.verb_token_ratio(text)
    
    def get_lexical_density(self, text):
        return LEX.lexical_density(text)
    
    def get_foreign_word_count(self, text):
        return LEX.foreign_word_counter(text)
    
    def get_compound_word_ratio(self, text):
        return LEX.compound_word_ratio(text)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for doc in X:
            ttr = self.get_ttr(doc)
            root_ttr = self.get_root_ttr(doc)
            corr_ttr = self.get_corr_ttr(doc)
            log_ttr = self.get_log_ttr(doc)
            noun_tr = self.get_noun_tr(doc)
            verb_tr = self.get_verb_tr(doc)
            lexical_density = self.get_lexical_density(doc)
            foreign_wc = self.get_foreign_word_count(doc)
            compund_wr = self.get_compound_word_ratio(doc)

            features.append([
                ttr, root_ttr, corr_ttr, log_ttr, noun_tr, verb_tr,
                lexical_density, foreign_wc, compund_wr
            ])
        return features
    
class MORPHEXtractor(BaseEstimator, TransformerMixin):
    def get_derivational_morph(self, text):
        return MORPH.get_derivational_morph(text)
    
    def actor_focus_ratio(self, text):
        return MORPH.actor_focus_ratio(text)
    
    def object_focus_ratio(self, text):
        return MORPH.object_focus_ratio(text)
    
    def benefactive_focus_ratio(self, text):
        return MORPH.benefactive_focus_ratio(text)
    
    def locative_focus_ratio(self, text):
        return MORPH.locative_focus_ratio(text)

    def instrumental_focus_ratio(self, text):
        return MORPH.instrumental_focus_ratio(text)

    def referential_focus_ratio(self, text):
        return MORPH.referential_focus_ratio(text)

    def infinitive_verb_ratio(self, text):
        return MORPH.infinitive_verb_ratio(text)

    def participle_verb_ratio(self, text):
        return MORPH.participle_verb_ratio(text)

    def perfective_verb_ratio(self, text):
        return MORPH.perfective_verb_ratio(text)

    def imperfective_verb_ratio(self, text):
        return MORPH.imperfective_verb_ratio(text)

    def contemplative_verb_ratio(self, text):
        return MORPH.contemplative_verb_ratio(text)

    def recent_past_verb_ratio(self, text):
        return MORPH.recent_past_verb_ratio(text)

    def aux_verb_ratio(self, text):
        return MORPH.aux_verb_ratio(text)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for doc in X:
            derivational_morph = self.get_derivational_morph(doc)
            actor_focus_ratio = self.actor_focus_ratio(doc)
            object_focus_ratio = self.object_focus_ratio(doc)
            benefactive_focus_ratio = self.benefactive_focus_ratio(doc)
            locative_focus_ratio = self.locative_focus_ratio(doc)
            instrumental_focus_ratio = self.instrumental_focus_ratio(doc)
            referential_focus_ratio = self.referential_focus_ratio(doc)
            infinitive_verb_ratio = self.infinitive_verb_ratio(doc)
            participle_verb_ratio = self.participle_verb_ratio(doc)
            perfective_verb_ratio = self.perfective_verb_ratio(doc)
            imperfective_verb_ratio = self.imperfective_verb_ratio(doc)
            contemplative_verb_ratio = self.contemplative_verb_ratio(doc)
            recent_past_verb_ratio = self.recent_past_verb_ratio(doc)
            aux_verb_ratio = self.aux_verb_ratio(doc)

            features.append([
                actor_focus_ratio, object_focus_ratio, benefactive_focus_ratio, locative_focus_ratio,
                instrumental_focus_ratio, referential_focus_ratio, infinitive_verb_ratio,
                participle_verb_ratio, perfective_verb_ratio, imperfective_verb_ratio,
                contemplative_verb_ratio, recent_past_verb_ratio, aux_verb_ratio
            ])
            features = derivational_morph + features

        return features


#   Custom transformer for OOV count feature extraction
class OOVExtractor(BaseEstimator, TransformerMixin):
    def get_oov_count(self, text):
        return OOV.count_oov_words(text)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for doc in X:
            oov_count = self.get_oov_count(doc)
            features.append([
                oov_count,
            ])
        return features

#   Custom transformer for StopWords count feature extraction
class StopWordsExtractor(BaseEstimator, TransformerMixin):
    def get_stopwords_count(self, text):
        return SW.count_stopwords(text)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for doc in X:
            stopwords_count = self.get_stopwords_count(doc)
            features.append([
                stopwords_count,
            ])
        return features


