import string
import os
from nltk.tokenize import wordpunct_tokenize

#   Note from Ren: Slightly increases LR accuracy by around 0.003 relative to Base LR with TRAD + SYLL.

"""PREAMBLE"""
# Base directory for dictionaries
DICTIONARY_BASE_DIR = "dictionaries"

# Get the directory of the current script
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Asbolute paths for dictionaries
ENGLISH_DICTIONARY_PATH = os.path.join(SCRIPT_DIRECTORY, DICTIONARY_BASE_DIR, "en.wl")
FILIPINO_DICTIONARY_PATH = os.path.join(SCRIPT_DIRECTORY, DICTIONARY_BASE_DIR, "fil.wl")
DICTIONARY_PATHS = [ENGLISH_DICTIONARY_PATH, FILIPINO_DICTIONARY_PATH]                      #   Aggregate dictionaries into a list for future-proofing

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