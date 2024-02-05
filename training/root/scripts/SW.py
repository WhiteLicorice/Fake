import string
import os
from nltk.tokenize import wordpunct_tokenize

#   Note from Ren: Slightly reduces LR accuracy by around 0.005 relative to Base LR with TRAD + SYLL.

"""PREAMBLE"""
# Base directory for stopwords
STOPWORDS_BASE_DIR = "stopwords"

# Get the directory of the current script
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Asbolute paths for stopwords
ENGLISH_STOPWORDS_PATH = os.path.join(SCRIPT_DIRECTORY, STOPWORDS_BASE_DIR, "en.sw")
FILIPINO_STOPWORDS_PATH = os.path.join(SCRIPT_DIRECTORY, STOPWORDS_BASE_DIR, "fil.sw")
STOPWORDS_PATHS = [ENGLISH_STOPWORDS_PATH, FILIPINO_STOPWORDS_PATH]                         #   Aggregate stopwords into a list for future-proofing

"""UTILITY FUNCTIONS"""
#   Helper method for loading stopwords
def load_stopwords(STOPWORDS_PATHS):
        stopwords = set()
        for stopword_path in STOPWORDS_PATHS:
            with open(stopword_path, 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    stopwords.add(word.lower())
        return stopwords
    
"""   CORE FUNCTIONS   """
#   Load the vocabulary
stopwords = load_stopwords(STOPWORDS_PATHS)

def count_stopwords(text):
    # Tokenize the text using NLTK's word_tokenize and remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in wordpunct_tokenize(text)]
    
    # Count the number of stopwords
    stopword_count = sum(1 for word in words if word and word in stopwords) or 0
    return stopword_count

#print(count_stopwords("The weather today ay napakainit, but we can still mag-enjoy ng kahit konting lamig sa ilalim ng malaking puno."))