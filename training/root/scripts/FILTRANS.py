#   Filipino linguistic feature extractors (TRAD, SYLL, LM, LEX, MORPH) adapted from: https://github.com/imperialite/filipino-linguistic-extractors
import root.scripts.LM as LM
import root.scripts.SYLL as SYLL
import root.scripts.TRAD as TRAD
import root.scripts.LEX as LEX
import root.scripts.MORPH as MORPH

import root.scripts.OOV as OOV
import root.scripts.SW as SW
import root.scripts.READ as READ

from sklearn.base import BaseEstimator, TransformerMixin

#   Custom transformer for TRAD feature extraction
class TRADExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, from_csv = False):
        self.from_csv = from_csv
        
    def get_word_count_per_doc(self, text):
        return TRAD.word_count_per_doc(text)
    
    def get_sentence_count_per_doc(self, text):
        return TRAD.sentence_count_per_doc(text)
    
    def get_polysyll_count_per_doc(self, text):
        return TRAD.polysyll_count_per_doc(text)
    
    def get_ave_word_length(self, text):
        return TRAD.ave_word_length(text)
    
    def get_ave_phrase_count_per_doc(self, text):
        return TRAD.ave_phrase_count_per_doc(text)

    def get_ave_syllable_count_of_word(self, text):
        return TRAD.ave_syllable_count_of_word(text)
    
    def get_word_count_per_sentence(self, text):
        return TRAD.word_count_per_sentence(text)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        if(self.from_csv):
            for doc in X.itertuples():
                word_count = doc.word_count
                sentence_count = doc.sentence_count
                polysyll_count = doc.polysyll_count
                ave_word_length = doc.ave_word_length
                ave_phrase_count = doc.ave_phrase_count
                ave_syllable_count_of_word = doc.ave_syllable_count_of_word
                word_count_per_sentence = doc.word_count_per_sentence
                features.append([
                    word_count, sentence_count, polysyll_count,
                    ave_word_length, ave_phrase_count, ave_syllable_count_of_word,
                    word_count_per_sentence
                ])
            return features
            
        for doc in X:
            word_count = self.get_word_count_per_doc(doc)
            sentence_count = self.get_sentence_count_per_doc(doc)
            polysyll_count = self.get_polysyll_count_per_doc(doc)
            ave_word_length = self.get_ave_word_length(doc)
            ave_phrase_count = self.get_ave_phrase_count_per_doc(doc)
            ave_syllable_count_of_word = self.get_ave_syllable_count_of_word(doc)
            word_count_per_sentence = self.get_word_count_per_sentence(doc)
            features.append([
                word_count, sentence_count, polysyll_count,
                ave_word_length, ave_phrase_count, ave_syllable_count_of_word,
                word_count_per_sentence
            ])
        return features
    def get_feature_names_out(self, _):
        return ['word_count','sentence_count','polysyll_count','ave_word_length','ave_phrase_count','ave_syllable_count_of_word','word_count_per_sentence']

#   Custom transformer for SYLL feature extraction
class SYLLExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, from_csv = False):
        self.from_csv = from_csv

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
        if(self.from_csv):
            for doc in X.itertuples():
                cc_cluster = doc.consonant_cluster
                v_density = doc.v_density
                cv_density = doc.cv_density
                vc_density = doc.vc_density
                cvc_density = doc.cvc_density
                vcc_density = doc.vcc_density
                cvcc_density = doc.cvcc_density
                ccvcc_density = doc.ccvcc_density
                ccvccc_density = doc.ccvccc_density
                features.append([
                    cc_cluster, v_density, cv_density, vc_density, cvc_density,
                    vcc_density, cvcc_density, ccvcc_density, ccvccc_density
                ])
            return features
        
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
    
    def get_feature_names_out(self, _):
        return ['consonant_cluster','v_density','cv_density','vc_density','cvc_density','vcc_density','cvcc_density','ccvcc_density','ccvccc_density',]
    
class LEXExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, from_csv = False):
        self.from_csv = from_csv
        
    def get_type_token_ratios(self, text):
        return LEX.get_type_token_ratios(text)
    
    def get_token_ratios(self, text):
        return LEX.get_token_ratios(text)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        if(self.from_csv):
            for doc in X.itertuples():
                (ttr, root_ttr, corr_ttr, log_ttr) = (doc.ttr, doc.root_ttr, doc.corr_ttr, doc.log_ttr)
                (noun_tr, verb_tr, lexical_density, foreign_tr, compound_tr) = (doc.noun_tr, doc.verb_tr, doc.lexical_density, doc.foreign_tr, doc.compound_tr)

                features.append([
                    ttr, root_ttr, corr_ttr, log_ttr, noun_tr, verb_tr,
                    lexical_density, foreign_tr, compound_tr
                ])
            return features
        
        for doc in X:
            (ttr, root_ttr, corr_ttr, log_ttr) = self.get_type_token_ratios(doc)
            (noun_tr, verb_tr, lexical_density, foreign_tr, compound_tr) = self.get_token_ratios(doc)

            features.append([
                ttr, root_ttr, corr_ttr, log_ttr, noun_tr, verb_tr,
                lexical_density, foreign_tr, compound_tr
            ])
        return features
    def get_feature_names_out(self, _):
        return ['ttr','root_ttr','corr_ttr','log_ttr','noun_tr','verb_tr','lexical_density','foreign_tr','compound_tr']
    
class MORPHExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, from_csv = False):
        self.from_csv = from_csv
        
    def get_derivational_morph(self, text):
        return MORPH.get_derivational_morph(text)
    
    def get_inflectional_morph(self, text):
        return MORPH.get_inflectional_morph(text)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        if(self.from_csv):
            for doc in X.itertuples():
                (
                    prefix_token_ratio, prefix_derived_ratio, suffix_token_ratio,
                    suffix_derived_ratio, total_affix_token_ratio, total_affix_derived_ratio
                ) = (
                    doc.prefix_token_ratio, doc.prefix_derived_ratio, doc.suffix_token_ratio,
                    doc.suffix_derived_ratio, doc.total_affix_token_ratio, doc.total_affix_derived_ratio
                )

                (
                    actor_focus_ratio, object_focus_ratio, benefactive_focus_ratio,
                    locative_focus_ratio, instrumental_focus_ratio, referential_focus_ratio,
                    infinitive_verb_ratio, participle_verb_ratio, perfective_verb_ratio,
                    imperfective_verb_ratio, contemplative_verb_ratio, recent_past_verb_ratio,
                    aux_verb_ratio
                ) = (
                    doc.actor_focus_ratio, doc.object_focus_ratio, doc.benefactive_focus_ratio,
                    doc.locative_focus_ratio, doc.instrumental_focus_ratio, doc.referential_focus_ratio,
                    doc.infinitive_verb_ratio, doc.participle_verb_ratio, doc.perfective_verb_ratio,
                    doc.imperfective_verb_ratio, doc.contemplative_verb_ratio, doc.recent_past_verb_ratio,
                    doc.aux_verb_ratio
                )

                features.append([
                    prefix_token_ratio, prefix_derived_ratio, suffix_token_ratio,
                    suffix_derived_ratio, total_affix_token_ratio, total_affix_derived_ratio,
                    actor_focus_ratio, object_focus_ratio, benefactive_focus_ratio, locative_focus_ratio,
                    instrumental_focus_ratio, referential_focus_ratio, infinitive_verb_ratio,
                    participle_verb_ratio, perfective_verb_ratio, imperfective_verb_ratio,
                    contemplative_verb_ratio, recent_past_verb_ratio, aux_verb_ratio
                ])

            return features
    
        for doc in X:
            (
                prefix_token_ratio, prefix_derived_ratio, suffix_token_ratio,
                suffix_derived_ratio, total_affix_token_ratio, total_affix_derived_ratio
            ) = self.get_derivational_morph(doc)

            (
                actor_focus_ratio, object_focus_ratio, benefactive_focus_ratio,
                locative_focus_ratio, instrumental_focus_ratio, referential_focus_ratio,
                infinitive_verb_ratio, participle_verb_ratio, perfective_verb_ratio,
                imperfective_verb_ratio, contemplative_verb_ratio, recent_past_verb_ratio,
                aux_verb_ratio
            ) = self.get_inflectional_morph(doc)

            features.append([
                prefix_token_ratio, prefix_derived_ratio, suffix_token_ratio,
                suffix_derived_ratio, total_affix_token_ratio, total_affix_derived_ratio,
                actor_focus_ratio, object_focus_ratio, benefactive_focus_ratio, locative_focus_ratio,
                instrumental_focus_ratio, referential_focus_ratio, infinitive_verb_ratio,
                participle_verb_ratio, perfective_verb_ratio, imperfective_verb_ratio,
                contemplative_verb_ratio, recent_past_verb_ratio, aux_verb_ratio
            ])

        return features
    def get_feature_names_out(self, _):
        return ['prefix_token_ratio','prefix_derived_ratio','suffix_token_ratio','suffix_derived_ratio','total_affix_token_ratio','total_affix_derived_ratio','actor_focus_ratio','object_focus_ratio','benefactive_focus_ratio','locative_focus_ratio','instrumental_focus_ratio','referential_focus_ratio','infinitive_verb_ratio','participle_verb_ratio','perfective_verb_ratio','imperfective_verb_ratio','contemplative_verb_ratio','recent_past_verb_ratio','aux_verb_ratio']


#   Custom transformer for OOV count feature extraction
class OOVExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, from_csv = False):
        self.from_csv = from_csv
        
    def get_oov_count(self, text):
        return OOV.count_oov_words(text)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        if(self.from_csv):
            for doc in X.itertuples():
                oov_count = doc.count_oov_words
                features.append([
                    oov_count,
                ])
            return features

        for doc in X:
            oov_count = self.get_oov_count(doc)
            features.append([
                oov_count,
            ])
        return features
    
    def get_feature_names_out(self, _):
        return["count_oov_words"]

#   Custom transformer for StopWords count feature extraction
class StopWordsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, from_csv = False):
        self.from_csv = from_csv
        
    def get_stopwords_count(self, text):
        return SW.count_stopwords(text)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        if(self.from_csv):
            for doc in X.itertuples():
                stopwords_count = doc.count_oov_words
                features.append([
                    stopwords_count,
                ])
            return features
        
        for doc in X:
            stopwords_count = self.get_stopwords_count(doc)
            features.append([
                stopwords_count,
            ])
        return features

    def get_feature_names_out(self, _):
        return ["count_stopwords"]

#   Custom transformer for StopWords count feature extraction
class READExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, from_csv = False):
        self.from_csv = from_csv
        
    def get_readability_score(self, text):
        return READ.compute_readability_score(text)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        if(self.from_csv):
            for doc in X.itertuples():
                readability_score = doc.readability_score
                features.append([
                    readability_score,
                ])
            return features

        for doc in X:
            readability_score = self.get_readability_score(doc)
            features.append([
                readability_score,
            ])
        return features
    def get_feature_names_out(self, _):
        return ["readability_score"]
