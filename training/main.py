import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from tokenizers import Tokenizer
from filipino_transformers import TRADExtractor, SYLLExtractor

from json import load as js_load

#   Suppress specific warning about tokenize_pattern from sklearn.feature_extraction.text
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

#   Preamble, load external files
global tokenizer
tokenizer = Tokenizer.from_file("root/tokenizers/tokenizer.json")

global stop_words_tl
with open("root/stopwords/stopwords-tl.json", "r") as tl:
	stop_words_tl = set(js_load(tl))
global stop_words_en
with open("root/stopwords/stopwords-en.txt", 'r') as en:
	stop_words_en = set(en.read().splitlines())

#   Read PHNews dataset
data = pd.read_csv("root/datasets/PHNews.csv")

# print(full_news.columns)
# print(full_news)

# # 1 = Real; All 1s starts from row 1604 and further
# real_news = full_news[1604:]
# # 0 = Fake; All 0s starts from row 1 to 1603
# fake_news = full_news[1:1604]

# real_news.head()
# fake_news.head()

#   Split the data into features (X) and labels (y)
X = data['article']
y = data['label']  # Labels are 0 -> Fake or 1 -> Real

# print(X)
# print(y)

#   Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#   Wrapper around pretrained BPE Tokenizer from Cruz et al.
def custom_tokenizer(doc):
    return tokenizer.encode(doc).tokens

#   Classifiers to test
classifiers = [
    {
        'name': 'Multinomial Naive Bayes',
        'model': MultinomialNB(),
        'params': {
            'classifier__alpha': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'Logistic Regression',
        'model': LogisticRegression(max_iter=10000, n_jobs=-1),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2']
        }
    },
    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(n_jobs=-1),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    {
        'name': 'SVC',
        'model': SVC(),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        },
        'n_jobs': -1 
    },
    {
        'name': 'Voting Classifier',
        'model': VotingClassifier(estimators=[
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=10000, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_jobs=-1)),
            ('svc', SVC())
        ], voting='hard'),
        'params': {
            'voting': ['hard', 'soft']
        }
    }
]


#   Test classifiers with pipeline
for clf_info in classifiers:
    print(f"Training {clf_info['name']}")
    
    # Create the pipeline with TruncatedSVD and the specified classifier
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), tokenizer=custom_tokenizer)),
            ('bow', CountVectorizer()),
            ('trad', TRADExtractor()),
            ('syll', SYLLExtractor())
        ])),
        ('classifier', clf_info['model'])
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(pipeline, clf_info['params'], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    #   Make predictions
    y_pred = grid_search.predict(X_test)

    #   Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    #   Classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)