from preProcess import preProcessLyrics
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