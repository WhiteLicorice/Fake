import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer

from root.scripts.FILTRANS import TRADExtractor, SYLLExtractor, OOVExtractor, StopWordsExtractor, LEXExtractor, MORPHExtractor, READExtractor
from root.scripts.BPE import BPETokenizer

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

#   Suppress specific warning about tokenize_pattern from sklearn.feature_extraction.text
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

LOAD_FROM_CSV = True

session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # Grab current time to use as timestamp on files

# Load Fake News Filipino by Cruz et al. dataset adapted from: https://github.com/jcblaisecruz02/Tagalog-fake-news
data = pd.read_csv("root/datasets/FakeNewsFilipino.csv")
trad_features = pd.read_csv("root/datasets/TradFeatures.csv")
syll_features = pd.read_csv("root/datasets/SyllFeatures.csv")
oov_features = pd.read_csv("root/datasets/OovFeatures.csv")
sw_features = pd.read_csv("root/datasets/SwFeatures.csv")
read_features = pd.read_csv("root/datasets/ReadFeatures.csv")
lex_features = pd.read_csv("root/datasets/LexFeatures.csv")
morph_features = pd.read_csv("root/datasets/MorphFeatures.csv")

data = pd.concat([data, trad_features, syll_features, oov_features, sw_features, read_features, lex_features, morph_features], axis=1)
# f = open(f"results/CrossVal/test_results_{session_timestamp}.txt", 'w')
print(f"Cross-Validation Testing Results ({session_timestamp})...")

# Split the data into features (X) and labels (y)
y = data['label']  # Labels are 0 -> Fake or 1 -> Real
X = data.drop('label', axis=1)

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define function for training with repeated cross-validation
def train_with_repeated_cv(classifier, classifier_name, params, repetitions, n_folds, X_train, y_train):
    pipelines = []
    feature_space = [
        ('vectorizers', ColumnTransformer(transformers=[
                ('bow',CountVectorizer(),'article'),
                ('tfidf',TfidfVectorizer(ngram_range=(1, 3), tokenizer=BPETokenizer().tokenize),'article')
            ])),                                                                                                            #   Get bag of words
        ('read', READExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract READ features
        ('oov', OOVExtractor(from_csv=LOAD_FROM_CSV)),                                                                  #   Extract OOV features
        ('sw', StopWordsExtractor(from_csv=LOAD_FROM_CSV)),
        ('trad', TRADExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract TRAD features
        ('syll', SYLLExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract SYLL features
        ('lex', LEXExtractor(from_csv=LOAD_FROM_CSV)),
        ('morph', MORPHExtractor(from_csv=LOAD_FROM_CSV))
    ]
    features_used = []
    # Define feature extraction pipeline
    # pipeline = Pipeline(steps=[
    #     ('features', FeatureUnion([
    #         ('vectorizers', ColumnTransformer(transformers=[
    #             ('bow',CountVectorizer(),'article'),
    #             ('tfidf',TfidfVectorizer(ngram_range=(1, 3), tokenizer=BPETokenizer().tokenize),'article')
    #         ])),                                                                                                            #   Get bag of words
    #         ('read', READExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract READ features
    #         ('oov', OOVExtractor(from_csv=LOAD_FROM_CSV)),                                                                  #   Extract OOV features
    #         ('sw', StopWordsExtractor(from_csv=LOAD_FROM_CSV)),
    #         ('trad', TRADExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract TRAD features
    #         ('syll', SYLLExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract SYLL features
    #         ('lex', LEXExtractor(from_csv=LOAD_FROM_CSV)),
    #         ('morph', MORPHExtractor(from_csv=LOAD_FROM_CSV))
    #     ])),
    #     ('classifier', classifier)
    # ])
    
    # Define cross-validation strategy
    cv = RepeatedKFold(n_splits=n_folds, n_repeats=repetitions, random_state=42)
    
    # Perform cross-validation
    for features_i in feature_space:
        features_used.append(features_i)
        accuracies = []
        pipeline = Pipeline(steps=[
            ('features', FeatureUnion(features_used)),
            ('classifier', classifier)
        ])

        for train_index, test_index in cv.split(X_train):
            # Split the data into training and validation sets
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            
            # Train the model on the training set
            pipeline.fit(X_train_fold, y_train_fold)
            
            # Make predictions on the validation set
            y_val_pred = pipeline.predict(X_val_fold)
            
            # Evaluate the model
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            accuracies.append(accuracy)
            
        # Compute and print the average accuracy
        avg_accuracy = sum(accuracies) / len(accuracies)
        # print(f"Features: {[i[0] for i in features_used]}", file=f)
        # print(f"Accuracies: {accuracies}", file=f)
        # print(f"Average accuracy for {repetitions} repetitions of {n_folds}-fold cross-validation with {classifier_name}: {avg_accuracy:.4f}", file=f)
        # print("", file=f)

        print(f"Features: {[i[0] for i in features_used]}")
        print(f"Accuracies: {accuracies}")
        print(f"Average accuracy for {repetitions} repetitions of {n_folds}-fold cross-validation with {classifier_name}: {avg_accuracy:.4f}")
        print("")

# Define classifiers to test
classifiers = [
    {
        'name': 'Multinomial Naive Bayes',
        'model_id': 'MultinomialNB',
        'model': MultinomialNB(),
        'params': {
            'classifier__alpha': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'Logistic Regression',
        'model_id': 'LogisticRegression',
        'model': LogisticRegression(max_iter=2000, n_jobs=-1),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'Random Forest',
        'model_id': 'RandomForest',
        'model': RandomForestClassifier(n_jobs=-1),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    {
        'name': 'SVC',
        'model_id': 'SVC',
        'model': SVC(),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        },
        'n_jobs': -1 
    },
]

# Train each classifier with repeated cross-validation
for clf_info in classifiers:
    train_with_repeated_cv(clf_info['model'], clf_info['name'], clf_info['params'], repetitions=6, n_folds=5, X_train=X_train, y_train=y_train)