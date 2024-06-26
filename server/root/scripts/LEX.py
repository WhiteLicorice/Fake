import nltk
from nltk.tag import StanfordPOSTagger
from nltk import wordpunct_tokenize
import os, math, re
from nltk import data as nltk_data
import gc

# DIRECTORIES FROM train.py when calling 'train.py'
if __name__ == "__main__":
    from TRAD import word_count_per_doc, sentence_count_per_doc, cleaner
    nltk_data.path.append("../runtime_env/nltk_data")       # Tell compiler too look for nltk_data at /runtime_env/nltk_data
    java_path = "../runtime_env/custom_java/bin"
    stanford_dir = "../runtime_env/stanford-postagger-full-2020-11-17"

else:
    from root.scripts.TRAD import word_count_per_doc, sentence_count_per_doc, cleaner
    nltk_data.path.append("./root/runtime_env/nltk_data")       # Tell compiler too look for nltk_data at /runtime_env/nltk_data
    java_path = "./root/runtime_env/custom_java/bin"
    stanford_dir = "./root/runtime_env/stanford-postagger-full-2020-11-17"

unique_tokens = []
total_tokens = 0

# MAIN FUNCTIONS
# Type Token Ratio (TTR)
def get_type_token_ratios(text):
    global unique_tokens
    global total_tokens

    unique_tokens = unique_tokentype_identifier(text)   # T - unique word types
    total_tokens = word_count_per_doc(text)

    if total_tokens == 0:
        return (0, 0, 0, 0)
    
    no_of_unique_tokens = len(unique_tokens)

    type_token_ratio = no_of_unique_tokens / total_tokens                                   # T/N
    root_type_token_ratio = no_of_unique_tokens / (math.sqrt(total_tokens))                 # T/√N
    corr_type_token_ratio = no_of_unique_tokens / (math.sqrt(2*total_tokens))               # T/√2N
    log_type_token_ratio = math.log10(no_of_unique_tokens) / (math.log10(total_tokens))     # log T/ log N

    return (
        type_token_ratio, root_type_token_ratio, corr_type_token_ratio, log_type_token_ratio
    )

def get_token_ratios(text):
    # global pos_tagger
    # pos_tagger=StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx4G")
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list
    tokenized_split = [wordpunct_tokenize(i.strip()) for i in splitted]
    tagged_text = pos_tagger.tag_sents(tokenized_split)

    noun_counter = 0
    verb_counter = 0
    lexical_item_counter = 0
    foreign_word_counter = 0
    compound_word_counter = 0

    for sentence in tagged_text:
        for word in sentence:
            if '|' not in word[0]:
                pos = word[1].split('|')[1]
                if pos == 'NNC' or pos == 'NNP' or pos == 'NNPA':
                    noun_counter += 1   # noun tokens
                if pos[:2] == 'VB':
                    verb_counter += 1   # verb tokens
                if pos[:2] == 'VB' or pos[:2] == 'NN' or pos[:2] == 'JJ' or pos[:2] == 'RB':
                    lexical_item_counter += 1   # lexical tokens
                if pos == 'FW':
                    foreign_word_counter += 1   # foreign word tokens

            if word[0] != '':
                compound_word_counter += 1      # compound word tokens
    
    if total_tokens == 0:
        return (0, 0, 0, 0, 0)
    
    noun_token_ratio = noun_counter/total_tokens
    verb_token_ratio = verb_counter/total_tokens
    lexical_density = lexical_item_counter/total_tokens
    foreign_word_ratio = foreign_word_counter/total_tokens
    compound_word_ratio = compound_word_counter/total_tokens
    # del pos_tagger
    # print(gc.collect())
    return (
        noun_token_ratio, verb_token_ratio, lexical_density, foreign_word_ratio, compound_word_ratio
    )

#UTILITY FUNCTIONS
#returns the T
def unique_tokentype_identifier(text):
    # pos_tagger=StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx4G")   # Change -Xmx4G to -XmxYG as needed where Y is the heap size in Gigabytes
    # global pos_tagger
    # tagged_text = pos_tagger.tag([text.lower()])

    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list
    tokenized_split = [wordpunct_tokenize(i.strip()) for i in splitted]
    tagged_text = pos_tagger.tag_sents(tokenized_split)

    global unique_tokens
    unique_tokens = []

    for i in tagged_text:
        for j in i:
            if '|' not in j[0]:
                pos = j[1].split('|')[1]
                if pos not in unique_tokens:
                    unique_tokens.append(pos)

    # del pos_tagger
    # print(gc.collect())
    return unique_tokens

os.environ['JAVAHOME'] = java_path
modelfile = stanford_dir + "/models/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger"
jarfile = stanford_dir + "/stanford-postagger.jar"

                                                                                                        # Add path para sa local machine kng mag analyze e.g -XX:HeapDumpPath=C:\\Acadsht
pos_tagger=StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx60G")   # Change -Xmx4G to -XmxYG as needed where Y is the heap size in Gigabytes
if __name__ == "__main__":
    import time
    import tracemalloc
    text = "Magpapakalat ang  Philippine National Police (PNP) ng 7,000 sa kanilang tourist policemen para bantayan ang mga aktibidad na may kaugnay sa paggunita ng Semana Santa sa susunod na linggo. Ibinahagi ito ni PNP spokesperson, Col. Jean Fajardo at aniya ang mga naturang tourist policemen ay magbabantay sa seguridad ng mga banyaga at lokal na turista na dadagsa sa ibat-ibang lugar. Bukod dito, 427 police service dogs ang ikakalat din sa mga transport terminals. Ayon kay Fajardo natukoy na nila ang mga kriikal na lugar na higit nilang babantayan gaya ng tourist spots, transport terminals, airports, seaports, simbahan, at maging sa mga tinatawag na religious destinations. Sa ngayon aniya ay wala pa silang natatanggap na anumang “credible threats” sa paggunita ng Semana Santa."

    start = time.time()
    tracemalloc.start()
    for i in get_type_token_ratios(text):
        print(i)
    for i in get_token_ratios(text):
        print(i)
 
    print(f'TIME: {time.time()-start}. SPACE: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()