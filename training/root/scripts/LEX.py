import nltk
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize
from root.scripts.TRAD import word_count_per_doc, sentence_count_per_doc, cleaner
import os, math, re

from nltk import data as nltk_data
nltk_data.path.append("./root/runtime_env/nltk_data")       # Tell compiler too look for nltk_data at /runtime_env/nltk_data

unique_tokens = []

# MAIN FUNCTIONS
# Type Token Ratio (TTR)
def type_token_ratio(text):
    global unique_tokens
    unique_tokens = unique_tokentype_identifier(text)   # T - unique word types
    total_tokens = 0                                    # N - total tokens in a text

    total_tokens = word_count_per_doc(text)
    if total_tokens == 0:
        return 0

    return len(unique_tokens) / total_tokens            #return T/N

# Root TTR
def root_type_token_ratio(text):
    global unique_tokens
    #unique_tokens = unique_tokentype_identifier(text)   
    total_tokens = 0                                    

    total_tokens = word_count_per_doc(text)
    if total_tokens == 0:
        return 0

    return len(unique_tokens) / (math.sqrt(total_tokens))   #return T/√N

# Corrected TTR
def corr_type_token_ratio(text):
    global unique_tokens
    #unique_tokens = unique_tokentype_identifier(text)   
    total_tokens = 0                                    

    total_tokens = word_count_per_doc(text)
    if total_tokens == 0:
        return 0

    return len(unique_tokens) / (math.sqrt(2*total_tokens)) #return T/√2N


# Bilogarithmic TTR
def log_type_token_ratio(text):
    global unique_tokens
    #unique_tokens = unique_tokentype_identifier(text)   
    total_tokens = 0                                    

    total_tokens = word_count_per_doc(text)
    if total_tokens == 0:
        return 0

    return math.log10(len(unique_tokens)) / (math.log10(total_tokens))  #return log T/ log N


# Noun-Token Ratio
def noun_token_ratio(text):
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list

    noun_counter = 0
    for i in splitted:
        i = i.strip()
        tagged_text = pos_tagger.tag(word_tokenize(i))
        for x in tagged_text:
            if '|' not in x[0]:
                pos = x[1].split('|')[1]
                if pos == 'NNC' or pos == 'NNP' or pos == 'NNPA':
                    noun_counter += 1

    word_count = word_count_per_doc(text)
    
    if word_count == 0:
        return 0
    return (noun_counter/word_count)


# Verb-Token Ratio
def verb_token_ratio(text):
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list

    verb_counter = 0
    for i in splitted:
        i = i.strip()
        tagged_text = pos_tagger.tag(word_tokenize(i))
        for x in tagged_text:
            if '|' not in x[0]:
                pos = x[1].split('|')[1]
                if pos[:2] == 'VB':
                    verb_counter += 1

    word_count = word_count_per_doc(text)
    
    if word_count == 0:
        return 0
    return (verb_counter/word_count_per_doc(text))


# Lexical Density
def lexical_density(text):
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list

    lexical_item_counter = 0
    for i in splitted:
        i = i.strip()
        tagged_text = pos_tagger.tag(word_tokenize(i))
        for x in tagged_text:
            if '|' not in x[0]:
                pos = x[1].split('|')[1]
                if pos[:2] == 'VB' or pos[:2] == 'NN' or pos[:2] == 'JJ' or pos[:2] == 'RB':
                    lexical_item_counter += 1

    word_count = word_count_per_doc(text)
    
    if word_count == 0:
        return 0
    return (lexical_item_counter/word_count_per_doc(text))


# Foreign Word Counter
def foreign_word_counter(text):
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list

    foreign_word_counter = 0
    for i in splitted:
        i = i.strip()
        tagged_text = pos_tagger.tag(word_tokenize(i))
        for x in tagged_text:
            if '|' not in x[0]:
                pos = x[1].split('|')[1]
                if pos == 'FW':
                    foreign_word_counter += 1

    word_count = word_count_per_doc(text)
    
    if word_count == 0:
        return 0
    return (foreign_word_counter/word_count_per_doc(text))

# Compound Word Ratio
def compound_word_ratio(text):
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list

    compound_counter = 0
    for i in splitted:
        i = i.strip()
        splitted_sents = i.split()
        for item in splitted_sents:
            tagged_text = pos_tagger.tag(word_tokenize(item))
            #print(tagged_text)

            tag = tagged_text[0]
            if tag[0] != '':
                compound_counter += 1

    word_count = word_count_per_doc(text)
    
    if word_count == 0:
        return 0
    return (compound_counter/word_count_per_doc(text))


#UTILITY FUNCTIONS
#returns the T
def unique_tokentype_identifier(text):
    tagged_text = pos_tagger.tag([text.lower()])
    global unique_tokens
    unique_tokens = []

    for i in tagged_text:
        if '|' not in i[0]:
            pos = i[1].split('|')[1]
            if pos not in unique_tokens:
                unique_tokens.append(pos)

    return unique_tokens


# DIRECTORIES FROM train.py when calling 'train.py'
java_path = ".root/runtime_env/custom_java/bin"

os.environ['JAVAHOME'] = java_path

stanford_dir = "./root/runtime_env/stanford-postagger-full-2020-11-17"

modelfile = stanford_dir + "/models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger"
jarfile = stanford_dir + "/stanford-postagger.jar"

pos_tagger=StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx4G")   # Change -Xmx4G to -XmxYG as needed where Y is the heap size in Gigabytes

# text = "Ito ang halimaw ng mga kulay.  Ngayong araw  gumising siyang kakaiba ang pakiramdam  nalilito  tuliro… Hindi niya alam kung ano ang mali sa kaniya.  Nalilito ka na naman? Hindi ka na natuto.  Anong gulo ang ginawa mo sa iyong mga damdamin!"


# print('type_token_ratio:', type_token_ratio(text))
# print('root_type_token_ratio:',root_type_token_ratio(text))
# print('Sentence Count:',sentence_count_per_doc(text))
# print('corr_type_token_ratio:', corr_type_token_ratio(text))
# print('log_type_token_ratio:',log_type_token_ratio(text))
# print('noun_token_ratio:',noun_token_ratio(text))
# print('verb_token_ratio:',verb_token_ratio(text))
# print('lexical_density:',lexical_density(text))
# print('foreign_word_counter:',foreign_word_counter(text))
# print('compound_word_ratio:',compound_word_ratio(text))