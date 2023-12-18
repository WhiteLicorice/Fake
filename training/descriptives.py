import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from json import load as js_load

#   Load English and Filipino stopwords adapted from: https://github.com/stopwords-iso/stopwords-iso
global stop_words_tl
with open("root/stopwords/stopwords-tl.json", "r") as tl:
	stop_words_tl = set(js_load(tl))
global stop_words_en
with open("root/stopwords/stopwords-en.txt", 'r') as en:
	stop_words_en = set(en.read().splitlines())

#   Read dataset
data = pd.read_csv("root/datasets/PHNews.csv")

#   Split the data into features (X) and labels (y)
X = data['article']
y = data['label']  # Labels are 0 -> Fake or 1 -> Real

#   Create CountVectorizer with stop words for both English and Filipino
vectorizer = CountVectorizer(stop_words=list(stop_words_tl.union(stop_words_en)))

#   Transform the entire dataset
dtm = vectorizer.fit_transform(X)

#   Separate fake and real articles
fake_dtm = dtm[y == 0]
real_dtm = dtm[y == 1]

#   Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

#   Create DataFrames to store word frequencies
fake_word_freq_df = pd.DataFrame(fake_dtm.toarray(), columns=feature_names)
real_word_freq_df = pd.DataFrame(real_dtm.toarray(), columns=feature_names)

#   Sort and get top ten words for fake articles
top_fake_words = fake_word_freq_df.sum(axis=0).sort_values(ascending=False).head(15)

#   Sort and get top ten words for real articles
top_real_words = real_word_freq_df.sum(axis=0).sort_values(ascending=False).head(15)

#   Display top words
print("Top words in fake articles:\n", top_fake_words)
print("Top words in real articles:\n", top_real_words)
