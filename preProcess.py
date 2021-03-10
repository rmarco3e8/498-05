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
from nrclex import NRCLex


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


def file_to_df():
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

    return df


def tfidf_features(random=None):
    df = file_to_df()

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


def to_nrclex(data):
    out_data = np.zeros((len(data), 10))
    for idx in range(len(data)):
        lyrics = data.iloc[idx]
        d = NRCLex(lyrics).raw_emotion_scores
        tokens = len(nltk.word_tokenize(lyrics))

        emo = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'postive', 'negative']
        for emo_idx in range(len(emo)):
            if emo[emo_idx] not in d.keys():
                out_data[idx, emo_idx] = 0
            else:
                out_data[idx, emo_idx] = (d[emo[emo_idx]] / tokens)

    return out_data


def nrclex_features(random=None):
    df = file_to_df()

    # 80-20 Train-test split
    if random is None:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df.labels)
    else:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=random, stratify=df.labels)

    X_train = to_nrclex(train_df.lyrics)
    X_test = to_nrclex(test_df.lyrics)

    return X_train, train_df.labels, X_test, test_df.labels


def preProcessAudio():
    pass

def main():
    preProcessLyrics()
    # preProcessAudio()

if __name__ == "__main__":
    main()