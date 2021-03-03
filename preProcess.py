# Import all of the necessary packages
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import nltk
import pathlib
import os
import shutil
#from pycontractions import Contractions
from helper import *
import contractions


"""
For each .txt file in MIREX-like_mood/dataset/Lyrics/, creates a corresponding
processed .txt file in /processed/
"""
def preProcessLyrics():

    # nltk.download('punkt')
    # nltk.download('stopwords')

    #cont = Contractions(api_key="glove-twitter-100")

    dir = os.path.join(os.getcwd(), 'processed')
    lyricsPath = os.path.join(os.getcwd(), 'MIREX-like_mood', 'dataset', 'Lyrics')

    if os.path.exists(dir):
        shutil.rmtree(dir)

    # Create /processed/
    os.makedirs(dir)

    count = 0

    for file in os.listdir(lyricsPath):
        filePath = os.path.join(lyricsPath, file)
        output = ''

        with open(filePath,encoding='utf8') as f:
            lyrics = f.read()

            # Make lowercase, lemmatize, expand contractions, remove stop words
            
            # adds periods @ new line 
            lyrics  = re.sub(r"(?<![[^\w\s']])\n(?!\n)",". ",lyrics)
            lyrics = lyrics.lower()

            stop_words = set(stopwords.words('english'))

            lyrics = [l for l in lyrics.split() if l not in stop_words]
            #print(lyrics)

            lyrics = lemmatize(lyrics)
            #print(lyrics)

            #lyrics = cont.expand_texts([lyrics])
            lyrics = [contractions.fix(l) for l in lyrics]
            #print(lyrics)
            
             # handle negation
            lyrics = handle_negation(" ".join(lyrics))
            #print(lyrics)

            # remove punctuation, while keeping negation terms. remove multiple white spaces  do we want to keep "[chorus]"  tags?
            output = re.sub(r'\s{1,}'," ", re.sub(r'[^\w\s_]','',lyrics))
            #print(output)

        with open(os.path.join(dir, file[:-4] + "p" + '.txt'), 'w', encoding='utf8') as f:
            f.write(output)

        count += 1

def tfidf_features(random=None):
    path = 'processed/'
    lyrics = []
    labels = []
    all_labels = []

    # Read all labels
    with open('MIREX-like_mood/dataset/clusters.txt', 'r') as f:
        for line in f:
            all_labels.append(int(line[8:9]))

    # Read lyrics and labels for the songs with lyrics
    for fname in os.listdir(path):
        with open(path + fname, 'r') as f:
            lyrics.append(f.readlines()[0])
            num = int(fname[:3]) - 1 # Subtract 1 to get 0 indexed
            labels.append(all_labels[num])

    lyrics = np.array(lyrics).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)
    df = pd.DataFrame(np.hstack((lyrics, labels)), columns=['lyrics', 'labels'])

    # 80-20 Train-test split
    if random is None:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df.labels)
    else:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=random, stratify=df.labels)

    tfidf = TfidfVectorizer()
    tfidf.fit(train_df.lyrics)
    X_train = tfidf.transform(train_df.lyrics).toarray()
    X_test = tfidf.transform(test_df.lyrics).toarray()

    print(df.labels.value_counts())

    return X_train, train_df.labels, X_test, test_df.labels

def svm_train(gamma_vals, k=5):
    X_train, y_train, X_test, y_test = tfidf_features(random=250)
    best_score, best_gamma = -1, -1

    for gamma in gamma_vals:
        clf = SVC(gamma=gamma)
        score = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
        print('gamma: ' + str(gamma) + '\tscore: ' + str(score))
        if score > best_score:
            best_score = score
            best_gamma = gamma

    print('\nBest gamma: ' + str(best_gamma) + '\n')

    clf = SVC(gamma=best_gamma)
    clf.fit(X_train, y_train)
    return clf, X_train, y_train, X_test, y_test

def get_metrics(y_pred, y):
    print('Accuracy:', str(metrics.accuracy_score(y, y_pred)))
    print('Confusion Matrix', str(metrics.confusion_matrix(y, y_pred)))

def svm_main(search='random', search_vals=25, k=5):
    if search == 'random':
        # Log Uniform
        exp = np.random.uniform(-3, 3, search_vals)
        gamma_vals = 10 ** exp
    elif search == 'grid':
        gamma_vals = np.logspace(-3, 3, search_vals)
    else:
        assert False, 'Search must be random or grid'

    clf, X_train, y_train, X_test, y_test = svm_train(gamma_vals, k)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    print('Performance on training dataset:')
    get_metrics(y_train_pred, y_train)
    print('\n\nPerformance on test data:')
    get_metrics(y_pred, y_test)


        


def preProcessAudio():
    pass

def main():
    # preProcessLyrics()
    # preProcessAudio()
    svm_main(search_vals=10)

if __name__ == "__main__":
    main()