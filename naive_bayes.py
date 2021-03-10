import os
import re
from preProcess import *
from sklearn.naive_bayes import GaussianNB
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
	clf = GaussianNB()
	clf.fit(X,y)
	return clf


def main():
	X_train, X_test, y_train, y_test = load_dataset()
	model = train_model(X_train.toarray(),y_train)
	y_pred = model.predict(X_test.toarray())
	print(classification_report(y_test,y_pred))

if __name__ == '__main__':
	main()

"""from preProcess import preProcessLyrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd

#for testing purposes
from sklearn.datasets import fetch_20newsgroups

def naiveBayes():

    clustersPath = os.path.join(os.getcwd(), 'MIREX-like_mood', 'dataset', 'clusters.txt')
    processedPath = os.path.join(os.getcwd(), 'processed')

    target = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5']

    #format of df: index, song, cluster, line, tf-idf, classification

    #placehonder
    #twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    #print(twenty_train)

    # Build clusters column
    df_clusters = pd.read_csv(clustersPath, delimiter='\n')

    processed = []
    
    # Build processed songs column
    for file in os.listdir(processedPath):
        lines = pd.read_csv(os.path.join(processedPath, file), delimiter='\n')
        index = int(os.path.realpath(__file__)[0:3])

        for line in lines:
            processed.append([index, line, df_clusters[index]])

    # Make dataframe from list
    df_processed = pd.DataFrame(processed, columns=['index', 'line', 'cluster'])

    #df = pd.DataFrame(np.random.randn(100, 2))
    msk = np.random.rand(len(df_processed)) < 0.8
    train = df_processed[msk]
    test = df_processed[~msk]

    train_lines = train.loc['line']

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_lines)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    clf = MultinomialNB().fit(X_train_tfidf, target)
    lines_new = ['I am so sad so very sad', 'The sun is shining and I am vibing']
    X_new_counts = count_vect.transform(lines_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    print(predicted)

    #Testing

    test_lines = test.loc['line']
    predicted = clf.predict(test_lines)
    np.mean(predicted == target)

def main():
    naiveBayes()

if __name__ == "__main__":
    main()
    """