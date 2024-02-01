import string
from nltk.tokenize import wordpunct_tokenize

#   Paths for dictionaries to be loaded
ENGLISH_DICTIONARY_PATH = r"root/scripts/dictionaries/en.wl"
FILIPINO_DICTIONARY_PATH = r"root/scripts/dictionaries/fil.wl"
DICTIONARY_PATHS = [ENGLISH_DICTIONARY_PATH, FILIPINO_DICTIONARY_PATH]

#   Helper method for loading dictionaries
def load_dictionaries(DICTIONARY_PATHS):
        vocabulary = set()
        for dictionary_path in DICTIONARY_PATHS:
            with open(dictionary_path, 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    vocabulary.add(word.lower())
        return vocabulary
    
#   Load the vocabulary
vocabulary = load_dictionaries(DICTIONARY_PATHS)

#   Count the number of words in text not in the vocabulary
def count_oov_words(text):
    # Tokenize the text using NLTK's word_tokenize and remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in wordpunct_tokenize(text)]

    # Count the number of out-of-vocabulary words
    oov_count = sum(1 for word in words if word and word not in vocabulary) or 0
    return oov_count

def main():
    print(count_oov_words("Si Maria and her barkada nag-decide mag-Bacolod this weekend para mag-try sang authentic na Chicken Inasal sa Manokan Country."))
    
if __name__ == "__main__":
    main()