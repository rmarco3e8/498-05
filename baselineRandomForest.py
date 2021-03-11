import os
import re
import numpy as np 
from preProcess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.stats import uniform, truncnorm, randint
from nrclex import NRCLex

DATA_DIR = "processed"

def get_features(song):
	"""use NRCLex to get emotional scores @ the song level 
	"""
	features = {'anger': 0, 'anticipation': 1, 'disgust': 2, 'fear': 3, 'joy': 4, 'sadness': 5,'surprise': 6, 'trust': 7, 'negative': 8, 'positive': 9}

	emotion_d = NRCLex(song).raw_emotion_scores
	vec = np.zeros(10)

	for k,v in emotion_d.items():
		vec[features[k]] = (v/len(song.split()))

	return vec 

def load_dataset(data = DATA_DIR, both = False, nrc_lex = False, idf = None):
	"""featurizes preprocessed data and loads into train/test split
	"""
	corpus = []
	y = []
	# load  all labels 
	labels = open(os.path.join('MIREX-like_mood',"dataset","clusters.txt"),"r").read().splitlines()

	for f in os.listdir(data):
		text = open(os.path.join(data,f),"r").read()
		# get index number of file to link back to clusters
		index = int(re.match(r'\d+', f).group(0))
		label = labels[index-1]
		corpus.append(text)
		y.append(label)

	X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20, random_state=42)
	vectorizer = TfidfVectorizer()	
	# fit the vectorizer on only the training set to avoid data leakage
	vectorizer.fit(X_train)

	if both == True:
		# get emotional features for each observation following split & before fit transform 
		X_train_lex = np.vstack([get_features(c) for c in X_train ])
		X_test_lex = np.vstack([get_features(c) for c in X_test ])

		X_train = vectorizer.transform(X_train).toarray()
		X_test = vectorizer.transform(X_test).toarray()

		# following fit transform. append lexical vectors to each observation	
		X_train = np.hstack((X_train,X_train_lex))
		X_test = np.hstack((X_test,X_test_lex))

	elif nrc_lex == True:
		X_train = np.vstack([get_features(c) for c in X_train ])
		X_test = np.vstack([get_features(c) for c in X_test ])

	elif idf == True:
		X_train = vectorizer.transform(X_train)
		X_test = vectorizer.transform(X_test)

	return X_train, X_test, y_train, y_test

def train_model(X, y, random_search = False):
	model_params = {
    # randomly sample numbers from 4 to 204 estimators
    'n_estimators': randint(4,200),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': uniform(0.01, 0.199)
	}
	clf = RandomForestClassifier(random_state = 42)
	if random_search == True:
		clf = RandomizedSearchCV(clf, model_params, n_iter =50, cv = 5, random_state = 42 )
		clf.fit(X,y)
		return clf
	else:		
		clf.fit(X,y)
		return clf

# class_weight='balanced' 
def main():
	X_train, X_test, y_train, y_test = load_dataset(nrc_lex = True)
	model = train_model(X_train,y_train, random_search= True)
	y_pred = model.predict(X_test)
	print(classification_report(y_test,y_pred))

if __name__ == '__main__':
	main()
