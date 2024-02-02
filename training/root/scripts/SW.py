import string
from nltk.tokenize import wordpunct_tokenize

#   Note from Ren: Slightly reduces LR accuracy by around 0.005 relative to Base LR with TRAD + SYLL.

"""PREAMBLE"""
#   Change filepaths if oov.py is being loaded as a module or as a standalone script
if __name__ == "__main__":
    #   Paths relative to oov.py
    ENGLISH_STOPWORDS_PATH = "./stopwords/en.sw"
    FILIPINO_STOPWORDS_PATH = "./stopwords/fil.sw"
    STOPWORDS_PATHS = [ENGLISH_STOPWORDS_PATH, FILIPINO_STOPWORDS_PATH]  #   Aggregate STOPWORDS paths for possible future-proofing
else:
    #   Paths relative to train.py
    ENGLISH_STOPWORDS_PATH = "root/scripts/stopwords/en.sw"
    FILIPINO_STOPWORDS_PATH = "root/scripts/stopwords/fil.sw"
    STOPWORDS_PATHS = [ENGLISH_STOPWORDS_PATH, FILIPINO_STOPWORDS_PATH]  #   Aggregate STOPWORDS paths for possible future-proofing

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