import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from filipino_transformers import TRADExtractor, SYLLExtractor
from bpe_tokenizer import BPETokenizer

import matplotlib.pyplot as plt
import seaborn as sns

#   Suppress specific warning about tokenize_pattern from sklearn.feature_extraction.text
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

#   Read PHNews dataset
data = pd.read_csv("root/datasets/PHNews.csv")

#   Split the data into features (X) and labels (y)
X = data['article']
y = data['label']  # Labels are 0 -> Fake or 1 -> Real

#   Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# #   Classifiers to test
# classifiers = [
#     {
#         'name': 'Multinomial Naive Bayes',
#         'model': MultinomialNB(),
#         'params': {
#             'classifier__alpha': [0.1, 1.0, 10.0]
#         }
#     },
#     {
#         'name': 'Logistic Regression',
#         'model': LogisticRegression(max_iter=2000, n_jobs=-1),
#         'params': {
#             'classifier__C': [0.1, 1.0, 10.0]
#         }
#     },
#     {
#         'name': 'Random Forest',
#         'model': RandomForestClassifier(n_jobs=-1),
#         'params': {
#             'classifier__n_estimators': [50, 100],
#             'classifier__max_depth': [10, 20],
#             'classifier__min_samples_split': [2, 5, 10]
#         }
#     },
#     {
#         'name': 'SVC',
#         'model': SVC(),
#         'params': {
#             'classifier__C': [0.1, 1.0, 10.0],
#             'classifier__kernel': ['linear', 'rbf']
#         },
#         'n_jobs': -1 
#     },
#     {
#         'name': 'Voting Classifier',
#         'model': VotingClassifier(estimators=[
#             ('lr', LogisticRegression(max_iter=2000, solver='liblinear')),
#             ('rf', RandomForestClassifier(n_jobs=-1)),
#             ('svc', SVC(probability=True))
#         ], voting = 'hard'),
#         'params': {
#             'classifier__voting': ['hard', 'soft']
#         }
#     }
# ]

#   Classifiers to test
classifiers = [
    {
        'name': 'Logistic Regression',
        'model': LogisticRegression(max_iter=2000, n_jobs=-1),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
]

print("CLASSIFIERS WITHOUT GRIDSEARCH")
#   Test classifiers with no gridsearch
for clf_info in classifiers:
    print(f"\nTraining Model: {clf_info['name']}")
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), tokenizer=BPETokenizer().tokenize)),        #   Get unigrams, bigrams, and trigrams
            ('bow', CountVectorizer()),                                                     #   Get bag of words
            ('trad', TRADExtractor()),                                                      #   Extract TRAD features
            ('syll', SYLLExtractor())                                                       #   Extract SYLL features
        ])),
        ('classifier', clf_info['model'])
    ])

    #   Fit the entire pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    #   Dump trained model
    #trained_model = pipeline.named_steps['classifier']
    with open(f"{clf_info['name']}.pkl", 'wb') as file:
        pickle.dump(pipeline, file)

    #   Make predictions
    y_pred = pipeline.predict(X_test)

    #   Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    #   Classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)
    
    #   Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {clf_info["name"]}')
    plt.show()

# print("CLASSIFIERS WITH GRIDSEARCH")
# #   Test classifiers with gridsearch
# for clf_info in classifiers:
#     print(f"\nTraining {clf_info['name']}")
#     pipeline = Pipeline([
#         ('features', FeatureUnion([
#             ('tfidf', TfidfVectorizer(ngram_range=(1, 3), tokenizer=bpe_tokenizer)),        #   Get unigrams, bigrams, and trigrams
#             ('bow', CountVectorizer()),                                                     #   Get bag of words
#             ('trad', TRADExtractor()),                                                      #   Extract TRAD features
#             ('syll', SYLLExtractor())                                                       #   Extract SYLL features
#         ])),
#         ('classifier', clf_info['model'])
#     ])
    
#     #   Perform grid search
#     grid_search = GridSearchCV(pipeline, clf_info['params'], cv=5, scoring='accuracy')
#     grid_search.fit(X_train, y_train)
    
#     # Print the best parameters
#     print(f"Best Estimator for {clf_info['name']}:\n{grid_search.best_estimator_}")

#     #   Make predictions
#     y_pred = grid_search.predict(X_test)

#     #   Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")

#     #   Classification report
#     class_report = classification_report(y_test, y_pred)
#     print("Classification Report:\n", class_report)
    
#     #   Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(f'Confusion Matrix - {clf_info["name"]}')
#     plt.show()
