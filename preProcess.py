# Import all of the necessary packages
#%load_ext autoreload
#%autoreload 2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from convokit import Corpus, download
from sklearn.model_selection import train_test_split
from numpy import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.stats import norm
import nltk
from statsmodels.graphics.gofplots import qqplot
import pathlib
import os
import shutil
from pycontractions import Contractions


"""
For each .txt file in MIREX-like_mood/dataset/Lyrics/, creates a corresponding
processed .txt file in /processed/
"""
def preProcessLyrics():

    nltk.download('punkt')
    nltk.download('stopwords')

    cont = Contractions(api_key="glove-twitter-100")

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

        # Make lowercase, expand contractions
        with open(filePath) as f:
            lyrics = f.read()
            lyrics = lyrics.lower()
            lyrics = cont.expand_texts([lyrics])

            for word in lyrics:
                output += word + ' '

        with open(os.path.join(dir, str(count) + '.txt'), 'w') as f:
            f.write(output)

        count += 1

def preProcessAudio():
    pass

def main():
    preProcessLyrics()
    preProcessAudio()

if __name__ == "__main__":
    main()