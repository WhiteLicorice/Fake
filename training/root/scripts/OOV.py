import string
from nltk.tokenize import wordpunct_tokenize

#   Note from Ren: Slightly increases LR accuracy by around 0.003 relative to Base LR with TRAD + SYLL.

"""PREAMBLE"""
#   Change filepaths if oov.py is being loaded as a module or as a standalone script
if __name__ == "__main__":
    #   Paths relative to oov.py
    ENGLISH_DICTIONARY_PATH = "./dictionaries/en.wl"
    FILIPINO_DICTIONARY_PATH = "./dictionaries/fil.wl"
    DICTIONARY_PATHS = [ENGLISH_DICTIONARY_PATH, FILIPINO_DICTIONARY_PATH]  #   Aggregate dictionary paths for possible future-proofing
else:
    #   Paths relative to train.py
    ENGLISH_DICTIONARY_PATH = "root/scripts/dictionaries/en.wl"
    FILIPINO_DICTIONARY_PATH = "root/scripts/dictionaries/fil.wl"
    DICTIONARY_PATHS = [ENGLISH_DICTIONARY_PATH, FILIPINO_DICTIONARY_PATH]  #   Aggregate dictionary paths for possible future-proofing
    
"""UTILITY FUNCTIONS"""
#   Helper method for loading dictionaries
def load_dictionaries(DICTIONARY_PATHS):
        vocabulary = set()
        for dictionary_path in DICTIONARY_PATHS:
            with open(dictionary_path, 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    vocabulary.add(word.lower())
        return vocabulary

"""   CORE FUNCTIONS   """
#   Load the vocabulary
vocabulary = load_dictionaries(DICTIONARY_PATHS)

#   Count the number of words in text not in the vocabulary
def count_oov_words(text):
    # Tokenize the text using NLTK's word_tokenize and remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in wordpunct_tokenize(text)]
    
    # Count the number of out-of-vocabulary words
    oov_count = sum(1 for word in words if word and word not in vocabulary) or 0
    return oov_count

#print(count_oov_words("Si 69 Maria and her barkada nag-decide mag-Bacolod this weekend para mag-try sang authentic na Chicken Inasal sa Manokan Country."))