import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from root.scripts.FILTRANS import TRADExtractor, SYLLExtractor, OOVExtractor, StopWordsExtractor, LEXExtractor, MORPHExtractor, READExtractor
from root.scripts.BPE import BPETokenizer

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

LOAD_FROM_CSV = True
FEATURE_COUNT = 47-28


session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # Grab current time to use as timestamp on files


# Load Fake News Filipino by Cruz et al. dataset adapted from: https://github.com/jcblaisecruz02/Tagalog-fake-news
data_cruz = pd.read_csv("root/datasets/Cruz/FakeNewsFilipino_Cruz2020.csv")
trad_features_cruz = pd.read_csv("root/datasets/Cruz/TradFeatures.csv")
syll_features_cruz = pd.read_csv("root/datasets/Cruz/SyllFeatures.csv")
oov_features_cruz = pd.read_csv("root/datasets/Cruz/OovFeatures.csv")
sw_features_cruz = pd.read_csv("root/datasets/Cruz/SwFeatures.csv")
read_features_cruz = pd.read_csv("root/datasets/Cruz/ReadFeatures.csv")
lex_features_cruz = pd.read_csv("root/datasets/Cruz/LexFeatures.csv")
morph_features_cruz = pd.read_csv("root/datasets/Cruz/MorphFeatures.csv")

# Load Fake News Filipino by Cruz et al. dataset adapted from: https://github.com/jcblaisecruz02/Tagalog-fake-news
data_lupac = pd.read_csv("root/datasets/Lupac/FakeNewsPhilippines2024_Lupac.csv")
trad_features_lupac = pd.read_csv("root/datasets/Lupac/TradFeatures.csv")
syll_features_lupac = pd.read_csv("root/datasets/Lupac/SyllFeatures.csv")
oov_features_lupac = pd.read_csv("root/datasets/Lupac/OovFeatures.csv")
sw_features_lupac = pd.read_csv("root/datasets/Lupac/SwFeatures.csv")
read_features_lupac = pd.read_csv("root/datasets/Lupac/ReadFeatures.csv")
lex_features_lupac = pd.read_csv("root/datasets/Lupac/LexFeatures.csv")
morph_features_lupac = pd.read_csv("root/datasets/Lupac/MorphFeatures.csv")

data = pd.concat([data_cruz, data_lupac], ignore_index=True)
trad_features = pd.concat([trad_features_cruz, trad_features_lupac], ignore_index=True)
syll_features = pd.concat([syll_features_cruz, syll_features_lupac], ignore_index=True)
oov_features = pd.concat([oov_features_cruz, oov_features_lupac], ignore_index=True)
sw_features = pd.concat([sw_features_cruz, sw_features_lupac], ignore_index=True)
read_features = pd.concat([read_features_cruz, read_features_lupac], ignore_index=True)
lex_features = pd.concat([lex_features_cruz, lex_features_lupac], ignore_index=True)
morph_features = pd.concat([morph_features_cruz, morph_features_lupac], ignore_index=True)

data = pd.concat([data,trad_features, syll_features, oov_features, sw_features, read_features], axis=1)
# data.to_csv("root/datasets/All_doc.csv", index=False)
# exit(0)
print(f"LR Coefficients ({session_timestamp})...")

# Split the data into features (X) and labels (y)
y = data['label']  # Labels are 0 -> Fake or 1 -> Real
X = data.drop('label', axis=1)

#   Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#   Classifiers to test
classifiers = [
    # {
    #     'name': 'Multinomial Naive Bayes',
    #     'model_id': 'MultinomialNB',
    #     'model': MultinomialNB(),
    #     'params': {
    #         'classifier__alpha': [0.1, 1.0, 10.0]
    #     }
    # },
    {
        'name': 'Logistic Regression',
        'model_id': 'LogisticRegression',
        'model': LogisticRegression(max_iter=3000, n_jobs=-1),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
    # {
    #     'name': 'Random Forest',
    #     'model_id': 'RandomForest',
    #     'model': RandomForestClassifier(n_jobs=-1),
    #     'params': {
    #         'classifier__n_estimators': [50, 100],
    #         'classifier__max_depth': [10, 20],
    #         'classifier__min_samples_split': [2, 5, 10]
    #     }
    # },
    # {
    #     'name': 'SVC',
    #     'model_id': 'SVC',
    #     'model': SVC(),
    #     'params': {
    #         'classifier__C': [0.1, 1.0, 10.0],
    #         'classifier__kernel': ['linear', 'rbf']
    #     },
    #     'n_jobs': -1 
    # },
]

# #   Classifiers to test
# classifiers = [
#     {
#         'name': 'Logistic Regression',
#         'model_id': 'LogisticRegression',
#         'model': LogisticRegression(max_iter=2000, n_jobs=-1),
#         'params': {
#             'classifier__C': [0.1, 1.0, 10.0]
#         }
#     },
# ]

# print("CLASSIFIERS WITHOUT GRIDSEARCH")
#   Test classifiers with no gridsearch
for clf_info in classifiers:

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
        # ('lex', LEXExtractor(from_csv=LOAD_FROM_CSV)),
        # ('morph', MORPHExtractor(from_csv=LOAD_FROM_CSV))
    ]
    classifier = clf_info['model']
    pipeline = Pipeline(steps=[
            ('features', FeatureUnion(feature_space)),
            ('classifier', classifier)
        ])

    print(f"\nTraining Model: {clf_info['name']}")

    # #   Fit the entire pipeline on the training data
    pipeline.fit(X_train, y_train)


    # GET FEATURE IMPORTANCE (COEFFICIENTS FOR LR)
    # ----------------------------------------------------------------------------------------

    feature_names = pipeline['features'].get_feature_names_out()
    feature_names_linguistic = feature_names[-FEATURE_COUNT:]
    feature_names_vectorizers = feature_names[:-FEATURE_COUNT]
    assert len(feature_names_linguistic) + len(feature_names_vectorizers) == len(feature_names), "vec and linguistic should be equal to total"

    feature_values = pipeline['classifier'].coef_[0]
    feature_values_linguistic = feature_values[-FEATURE_COUNT:]
    feature_values_vectorizers = feature_names[:-FEATURE_COUNT]

    coef_full = dict()
    coef_linguistic = dict()
    coef_vect = dict()
    for i in range(len(feature_names)):
        coef_full[feature_names[i]] = feature_values[i]
        if(i > (len(feature_names)) - (FEATURE_COUNT + 1)):
            coef_linguistic[feature_names[i]] = feature_values[i]
        else:
            coef_vect[feature_names[i]] = feature_values[i]

    coef_full = {k: v for k,v in sorted(coef_full.items(), key=lambda item: item[1])}
    coef_linguistic = {k: v for k,v in sorted(coef_linguistic.items(), key=lambda item: item[1])}
    coef_vect = {k: v for k,v in sorted(coef_vect.items(), key=lambda item: item[1])}

    print(len(coef_linguistic))
    print("")
    

    print("TOP 10 ON ALL FEATURES")
    print("------------------------------------")
    for i, k in enumerate(coef_full):
        if((i < 9) or (i > len(coef_full)-10)):
            print(f"{k}: {coef_full[k]}")
    
    print("\nTOP 10 ON VECTORIZERS")
    print("------------------------------------")
    for i, k in enumerate(coef_vect):
        if((i < 9) or (i > len(coef_vect)-10)):
            print(f"{k}: {coef_vect[k]}")

    print("\nALL LINGUISTIC FEATURES")
    print("------------------------------------")
    for i, k in enumerate(coef_linguistic):
        print(f"{k}: {coef_linguistic[k]}")
    
    # -------------------------------------------------------------------------------------------
    
    # Dump trained model
    # trained_model = pipeline.named_steps['classifier']
    # with open(f"{clf_info['model_id']}_{session_timestamp}.pkl", 'wb') as file:
    #     pickle.dump(pipeline, file)

    # #   Make predictions
    # y_pred = pipeline.predict(X_test)

    # #   Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # #   Classification report
    # class_report = classification_report(y_test, y_pred)
    # print("Classification Report:\n", class_report)
    
    # #   Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title(f'Confusion Matrix - {clf_info["name"]}')
    # #plt.show()
    # plt.savefig(f".\\results\\hyperparam\\{clf_info['name']}_none_{session_timestamp}.png", bbox_inches = 'tight')   #   Silently save confusion matrices for overnight training
    # plt.close()
    
# print("CLASSIFIERS WITH GRIDSEARCH")
# #   Test classifiers with gridsearch
# for clf_info in classifiers:
#     print(f"\nTraining {clf_info['name']}")
#     feature_space = [
#         ('vectorizers', ColumnTransformer(transformers=[
#                 ('bow',CountVectorizer(),'article'),
#                 ('tfidf',TfidfVectorizer(ngram_range=(1, 3), tokenizer=BPETokenizer().tokenize),'article')
#             ])),                                                                                                            #   Get bag of words
#         ('read', READExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract READ features
#         ('oov', OOVExtractor(from_csv=LOAD_FROM_CSV)),                                                                  #   Extract OOV features
#         ('sw', StopWordsExtractor(from_csv=LOAD_FROM_CSV)),
#         ('trad', TRADExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract TRAD features
#         ('syll', SYLLExtractor(from_csv=LOAD_FROM_CSV)),                                                                #   Extract SYLL features
#         ('lex', LEXExtractor(from_csv=LOAD_FROM_CSV)),
#         ('morph', MORPHExtractor(from_csv=LOAD_FROM_CSV))
#     ]
#     classifier = clf_info['model']
#     pipeline = Pipeline(steps=[
#             ('features', FeatureUnion(feature_space)),
#             ('classifier', classifier)
#         ])
    
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
#     plt.savefig(f".\\results\\hyperparam\\{clf_info['name']}_tuning_{session_timestamp}.png", bbox_inches = 'tight')   #   Silently save confusion matrices for overnight training
#     plt.close()