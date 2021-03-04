import os
import re
from preProcess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

DATA_DIR = "processed"

def load_dataset(data = DATA_DIR):
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

	X_train = vectorizer.transform(X_train)
	X_test = vectorizer.transform(X_test)

	return X_train, X_test, y_train, y_test

def train_model(X, y):
	clf = RandomForestClassifier(random_state = 42)
	clf.fit(X,y)
	return clf


def main():
	X_train, X_test, y_train, y_test = load_dataset()
	model = train_model(X_train,y_train)
	y_pred = model.predict(X_test)
	print(classification_report(y_test,y_pred))

if __name__ == '__main__':
	main()
