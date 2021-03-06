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

def preProcessAudio():
    pass

def main():
    preProcessLyrics()
    preProcessAudio()

if __name__ == "__main__":
    main()