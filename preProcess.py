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
from midi2audio import FluidSynth
import csv
from pydub import AudioSegment


"""
For each .txt file in MIREX-like_mood/dataset/Lyrics/, creates a corresponding
processed .txt file in /processed/
"""
def preProcessLyrics():

    # nltk.download('punkt')
    # nltk.download('stopwords')

    #cont = Contractions(api_key="glove-twitter-100")

    dir = os.path.join(os.getcwd(), 'processed')
    audio_dir = os.path.join(os.getcwd(), 'audio_processed')
    lyricsPath = os.path.join(os.getcwd(), 'MIREX-like_mood', 'dataset', 'Lyrics')
    audioPath = os.path.join(os.getcwd(), 'MIREX-like_mood', 'dataset', 'Audio')

    if os.path.exists(dir):
        shutil.rmtree(dir)

    # Create /processed/
    os.makedirs(dir)

    count = 0
    
    lyrics_count = 0
    audio_count = 0

    for file in os.listdir(lyricsPath):
        filePath = os.path.join(lyricsPath, file)
        audioFilePath = os.path.join(audioPath, file[:-4] + ".mp3")
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
            lyrics_count += 1

        #print(audioFilePath)
        if os.path.exists(audioFilePath):
            #print("HERE")
            with open(os.path.join(audio_dir, file[:-4] + "p" + '.txt'), 'w', encoding='utf8') as f:
                f.write(output)
                audio_count += 1

        count += 1

    print(lyrics_count)
    print(len(os.listdir(lyricsPath)))
    print(audio_count)
    print(len(os.listdir(audioPath)))

def file_to_df(audio=False):
    path = 'processed/'
    if audio:
        path = 'audio_processed/'
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

def marsyas_df_zerocrossings():
    #path = 'MIDIsCSVzero/'
    path = 'AudioCSVzero/'
    lyrics_path = 'processed/'
    #processed_path = 'audio_processed/'
    marsyas_path = 'marsyas-0.5.0/bin/'

    zeroCrossings = []
    labels = []
    all_labels = []

    # Read all labels
    with open('MIREX-like_mood/dataset/clusters.txt', 'r') as f:
        for line in f:
            all_labels.append(int(line[8:9]))

    # Read lyrics and labels for the songs with lyrics
    for fname in os.listdir(path):

        lyrics_filename = os.path.join(lyrics_path, fname[:-4] + '.txt')
        if os.path.exists(lyrics_filename):

            with open(path + fname, newline='') as f:

                reader = csv.reader(f, delimiter=' ')
                #ncol = len(next(reader))
                #f.seek(0)

                total = 0
                #totals = np.zeros(ncol)
                #total = ""
                count_samples = 0

                # Sum zero crossings for the first audio channel
                for row in reader:
                    #total += str(row[0]) + " "
                    count_samples += 1
                    total += float(row[0])
                    #for col in range(ncol):
                        #totals[col] += row[col]

                #print(total)
                #print(count_samples)
                zeroCrossings.append(int(total))
                #zeroCrossings.append(total)
                num = int(fname[:3]) - 1 # Subtract 1 to get 0 indexed
                labels.append(all_labels[num])

    zeroCrossings = np.array(zeroCrossings).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)
    df = pd.DataFrame(np.hstack((zeroCrossings, labels)), columns=['zeroCrossings', 'labels'])

    return df

def marsyas_df_yinpitch():
    #path = 'MIDIsCSVyin/'
    path = 'AudioCSVyin/'
    processed_path = 'audio_processed/'
    marsyas_path = 'marsyas-0.5.0/bin/'

    yinPitch = []
    labels = []
    all_labels = []

    # Read all labels
    with open('MIREX-like_mood/dataset/clusters.txt', 'r') as f:
        for line in f:
            all_labels.append(int(line[8:9]))

    # Read lyrics and labels for the songs with lyrics
    for fname in os.listdir(path):

        with open(path + fname, newline='') as f:

            reader = csv.reader(f, delimiter=' ')
            #ncol = len(next(reader))
            #f.seek(0)

            #totals = np.zeros(ncol)
            total = 0

            # Sum zero crossings for the first audio channel
            for row in reader:
                total += float(row[0])
                #for col in range(ncol):
                    #totals[col] += row[col]

            yinPitch.append(int(total))
            num = int(fname[:3]) - 1 # Subtract 1 to get 0 indexed
            labels.append(all_labels[num])

    yinPitch = np.array(yinPitch).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)
    df = pd.DataFrame(np.hstack((yinPitch, labels)), columns=['yinPitch', 'labels'])

    return df

def marsyas_df_energy():
    #path = 'MIDIsCSVzero/'
    path = 'AudioCSVenergy/'
    #processed_path = 'audio_processed/'
    marsyas_path = 'marsyas-0.5.0/bin/'

    energy = []
    labels = []
    all_labels = []

    # Read all labels
    with open('MIREX-like_mood/dataset/clusters.txt', 'r') as f:
        for line in f:
            all_labels.append(int(line[8:9]))

    # Read lyrics and labels for the songs with lyrics
    for fname in os.listdir(path):

        with open(path + fname, newline='') as f:

            reader = csv.reader(f, delimiter=' ')
            #ncol = len(next(reader))
            #f.seek(0)

            total = 0
            #totals = np.zeros(ncol)
            #total = ""
            count_samples = 0

            # Sum zero crossings for the first audio channel
            for row in reader:
                #total += str(row[0]) + " "
                count_samples += 1
                total += float(row[0])
                #for col in range(ncol):
                    #totals[col] += row[col]

            #print(total)
            #print(count_samples)
            energy.append(int(total))
            #zeroCrossings.append(total)
            num = int(fname[:3]) - 1 # Subtract 1 to get 0 indexed
            labels.append(all_labels[num])

    energy = np.array(energy).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)
    df = pd.DataFrame(np.hstack((energy, labels)), columns=['energy', 'labels'])

    return df

def marsyas_df_power():
    #path = 'MIDIsCSVzero/'
    path = 'AudioCSVpower/'
    #processed_path = 'audio_processed/'
    marsyas_path = 'marsyas-0.5.0/bin/'

    power = []
    labels = []
    all_labels = []

    # Read all labels
    with open('MIREX-like_mood/dataset/clusters.txt', 'r') as f:
        for line in f:
            all_labels.append(int(line[8:9]))

    # Read lyrics and labels for the songs with lyrics
    for fname in os.listdir(path):

        with open(path + fname, newline='') as f:

            reader = csv.reader(f, delimiter=' ')
            #ncol = len(next(reader))
            #f.seek(0)

            total = 0
            #totals = np.zeros(ncol)
            #total = ""
            count_samples = 0

            # Sum zero crossings for the first audio channel
            for row in reader:
                #total += str(row[0]) + " "
                count_samples += 1
                total += float(row[0])
                #for col in range(ncol):
                    #totals[col] += row[col]

            #print(total)
            #print(count_samples)
            power.append(int(total))
            #zeroCrossings.append(total)
            num = int(fname[:3]) - 1 # Subtract 1 to get 0 indexed
            labels.append(all_labels[num])

    power = np.array(power).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)
    df = pd.DataFrame(np.hstack((power, labels)), columns=['power', 'labels'])

    return df

def tfidf_features(random=None, audio=False):
    df = file_to_df(audio)

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

def nrclex_features(random=None, audio=False):
    df = file_to_df(audio)

    # 80-20 Train-test split
    if random is None:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df.labels)
    else:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=random, stratify=df.labels)

    X_train = to_nrclex(train_df.lyrics)
    X_test = to_nrclex(test_df.lyrics)

    return X_train, train_df.labels, X_test, test_df.labels

def to_zerocrossing(data):
    #arr = data.split(str=" ")
    #print(arr)
    test = data.iloc[0].split()
    out_data = np.zeros((len(data), len(test)))
    for idx in range(len(data)):
        zeroCrossings = data.iloc[idx]
        arr = zeroCrossings.split()
        print(len(arr))
        for i in range(len(arr)):
            out_data[idx][i] = float(arr[i])

    print(out_data)

    return out_data

def marsyas_features_zero(random=None):
    df = marsyas_df_zerocrossings()

    if random is None:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df.labels)
    else:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=random, stratify=df.labels)

    X_train = np.array(train_df.zeroCrossings).reshape(-1, 1)
    X_test = np.array(test_df.zeroCrossings).reshape(-1,1)

    #X_train = to_zerocrossing(train_df.zeroCrossings)
    #X_test = to_zerocrossing(test_df.zeroCrossings)

    return X_train, train_df.labels, X_test, test_df.labels

def marsyas_features_yin(random=None):
    df = marsyas_df_yinpitch()

    if random is None:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df.labels)
    else:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=random, stratify=df.labels)

    X_train = np.array(train_df.yinPitch).reshape(-1, 1)
    X_test = np.array(test_df.yinPitch).reshape(-1,1)

    return X_train, train_df.labels, X_test, test_df.labels

def marsyas_features_energy(random=None):
    df = marsyas_df_energy()

    if random is None:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df.labels)
    else:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=random, stratify=df.labels)

    X_train = np.array(train_df.energy).reshape(-1, 1)
    X_test = np.array(test_df.energy).reshape(-1,1)

    #X_train = to_zerocrossing(train_df.zeroCrossings)
    #X_test = to_zerocrossing(test_df.zeroCrossings)

    return X_train, train_df.labels, X_test, test_df.labels

def marsyas_features_power(random=None):
    df = marsyas_df_power()

    if random is None:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, stratify=df.labels)
    else:
        train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=random, stratify=df.labels)

    X_train = np.array(train_df.power).reshape(-1, 1)
    X_test = np.array(test_df.power).reshape(-1,1)

    #X_train = to_zerocrossing(train_df.zeroCrossings)
    #X_test = to_zerocrossing(test_df.zeroCrossings)

    return X_train, train_df.labels, X_test, test_df.labels

def preProcessAudio():

    dir = os.path.join(os.getcwd(), 'audio_processed')

    """
    
    MIDIsPath = os.path.join(os.getcwd(), 'MIREX-like_mood', 'dataset', 'MIDIs')
    MIDIsWavPath = os.path.join(os.getcwd(), 'MIDIsWav')
    MIDIsCSVPathZero = os.path.join(os.getcwd(), 'MIDIsCSVzero')
    MIDIsCSVPathYin = os.path.join(os.getcwd(), 'MIDIsCSVyin')

    """
    AudioPath = os.path.join(os.getcwd(), 'Mirex-like_mood', 'dataset', 'Audio')
    AudioWavPath = os.path.join(os.getcwd(), 'AudioWav')
    AudioCSVPathZero = os.path.join(os.getcwd(), 'AudioCSVzero')
    AudioCSVPathYin = os.path.join(os.getcwd(), 'AudioCSVyin')
    AudioCSVPathSkewness = os.path.join(os.getcwd(), 'AudioCSVskewness')
    AudioCSVPathSFM = os.path.join(os.getcwd(), 'AudioCSVsfm')
    AudioCSVPathEnergy = os.path.join(os.getcwd(), 'AudioCSVenergy')
    AudioCSVPathPower = os.path.join(os.getcwd(), 'AudioCSVpower')

    marsyas_path = 'marsyas-0.5.0/bin/'

    if os.path.exists(dir):
        shutil.rmtree(dir)

    # Create /audio_processed/
    os.makedirs(dir)

    ##################################
    """

    if os.path.exists(MIDIsWavPath):
        shutil.rmtree(MIDIsWavPath)

    # Create /MIDIsWav/
    os.makedirs(MIDIsWavPath)

    if os.path.exists(MIDIsCSVPathZero):
        shutil.rmtree(MIDIsCSVPathZero)

    # Create /MIDIsCSVzero/
    os.makedirs(MIDIsCSVPathZero)

    if os.path.exists(MIDIsCSVPathYin):
        shutil.rmtree(MIDIsCSVPathYin)

    # Create /MIDIsCSVyin/
    os.makedirs(MIDIsCSVPathYin)

    ##################################
    """
    """

    if os.path.exists(AudioWavPath):
        shutil.rmtree(AudioWavPath)

    # Create /AudioWav/
    os.makedirs(AudioWavPath)

    if os.path.exists(AudioCSVPathZero):
        shutil.rmtree(AudioCSVPathZero)

    # Create /AudioCSVzero/
    os.makedirs(AudioCSVPathZero)

    if os.path.exists(AudioCSVPathYin):
        shutil.rmtree(AudioCSVPathYin)

    # Create /AudioCSVyin/
    os.makedirs(AudioCSVPathYin)

    if os.path.exists(AudioCSVPathSkewness):
        shutil.rmtree(AudioCSVPathSkewness)

    # Create /AudioCSVskewness/
    os.makedirs(AudioCSVPathSkewness)

    if os.path.exists(AudioCSVPathSFM):
        shutil.rmtree(AudioCSVPathSFM)

    # Create /AudioCSVsfm/
    os.makedirs(AudioCSVPathSFM)



    if os.path.exists(AudioCSVPathEnergy):
        shutil.rmtree(AudioCSVPathEnergy)

    # Create /AudioCSVenergy/
    os.makedirs(AudioCSVPathEnergy)

    """

    if os.path.exists(AudioCSVPathPower):
        shutil.rmtree(AudioCSVPathPower)

    # Create /AudioCSVpower/
    os.makedirs(AudioCSVPathPower)

    """
    ##################################################


    fs = FluidSynth(sound_font="GeneralUser_GS_1.471/GeneralUser_GS_v1.471.sf2")

    for file in os.listdir(MIDIsPath):
        fs.midi_to_audio(os.path.join(MIDIsPath, file), os.path.join(MIDIsWavPath, file[:-4] + "p" + '.wav'))

    for file in os.listdir(MIDIsWavPath):
        os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'zero_crossings.mrs -c input/filename="MIDIsWav/' + file + '"')
        os.rename('result_zerocrossings.csv', 'MIDIsCSVzero/' + file[:-4] + '.csv')

        os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'yin_pitch.mrs -c input/filename="MIDIsWav/' + file + '"')
        os.rename('result_yinpitch.csv', 'MIDIsCSVyin/' + file[:-4] + '.csv')


    ##################################################
    """

    #for file in os.listdir(AudioPath):
        #sound = AudioSegment.from_mp3(os.path.join(AudioPath, file))
        #sound.export(os.path.join(AudioWavPath, file[:-4] + "p" + '.wav'), format="wav")

    for file in os.listdir(AudioWavPath):
        #os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'zero_crossings.mrs -c input/filename="AudioWav/' + file + '"')
        #os.rename('result_zerocrossings.csv', 'AudioCSVzero/' + file[:-4] + '.csv')

        #os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'yin_pitch.mrs -c input/filename="AudioWav/' + file + '"')
        #os.rename('result_yinpitch.csv', 'AudioCSVyin/' + file[:-4] + '.csv')

        #os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'skewness.mrs -c input/filename="AudioWav/' + file + '"')
        #os.rename('result_skewness.csv', 'AudioCSVskewness/' + file[:-4] + '.csv')

        #os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'sfm.mrs -c input/filename="AudioWav/' + file + '"')
        #os.rename('result_sfm.csv', 'AudioCSVsfm/' + file[:-4] + '.csv')

        #os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'energy.mrs -c input/filename="AudioWav/' + file + '"')
        #os.rename('result_energy.csv', 'AudioCSVenergy/' + file[:-4] + '.csv')

        os.system('./' + marsyas_path + 'marsyas-run.exe ' + marsyas_path + 'power.mrs -c input/filename="AudioWav/' + file + '"')
        os.rename('result_power.csv', 'AudioCSVpower/' + file[:-4] + '.csv')


def main():
    preProcessLyrics()
    #preProcessAudio()

if __name__ == "__main__":
    main()