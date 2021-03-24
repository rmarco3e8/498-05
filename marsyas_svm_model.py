# Import all of the necessary packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from preProcess import *
import pickle

def svm_train(gamma_vals, k=5, feature_type='tfidf'):
    if feature_type == 'tfidf':
        X_train, y_train, X_test, y_test = tfidf_features(random=250)
    elif feature_type == 'nrclex':
        X_train, y_train, X_test, y_test = nrclex_features(random=250)
    elif feature_type == 'both':
        X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf = tfidf_features(random=250)
        X_train_nrc, y_train_nrc, X_test_nrc, y_test_nrc = nrclex_features(random=250)
        assert np.all(y_train_tfidf == y_train_nrc), 'Mismatch in y_train'
        assert np.all(y_test_tfidf == y_test_nrc), 'Mismatch in y_test'
        X_train = np.hstack((X_train_tfidf, X_train_nrc))
        X_test = np.hstack((X_test_tfidf, X_test_nrc))
        y_train = y_train_tfidf
        y_test = y_test_tfidf
    elif feature_type == 'marsyas':
        X_train, y_train, X_test, y_test = marsyas_features(random=250)
    else:
        assert False, 'Invalid Feature Type in svm_train()'

    best_score, best_gamma = -1, -1
    results_d = {}

    for gamma in gamma_vals:
        clf = SVC(gamma=gamma, class_weight='balanced')
        score = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
        print('gamma: ' + str(gamma) + '\tscore: ' + str(score))
        results_d[gamma] = score
        if score > best_score:
            best_score = score
            best_gamma = gamma

    print('\nBest gamma: ' + str(best_gamma) + '\n')

    clf = SVC(gamma=best_gamma, class_weight='balanced')
    clf.fit(X_train, y_train)
    with open('models_marsyas/trained_svc.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('models_marsyas/validation_results.txt', 'w') as f:
        for key in results_d.keys():
            f.write('gamma: ' + str(key) + '\tscore: ' + str(results_d[key]) + '\n')
    return clf, X_train, y_train, X_test, y_test

def get_metrics(y_pred, y):
    print('Accuracy:', str(metrics.accuracy_score(y, y_pred)))
    print('Confusion Matrix', str(metrics.confusion_matrix(y, y_pred)))
    print('Classification Report', str(metrics.classification_report(y, y_pred)))

def svm_main(search='random', search_vals=25, k=5, feature_type='tfidf'):
    if search == 'random':
        # Log Uniform
        exp = np.random.uniform(-3, 3, search_vals)
        gamma_vals = 10 ** exp
    elif search == 'grid':
        gamma_vals = np.logspace(-3, 3, search_vals)
    else:
        assert False, 'Search must be random or grid'

    clf, X_train, y_train, X_test, y_test = svm_train(gamma_vals, k, feature_type)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    print('Performance on training dataset:')
    get_metrics(y_train_pred, y_train)
    print('\n\nPerformance on test data:')
    get_metrics(y_pred, y_test)


if __name__ == "__main__":
    #svm_main(search_vals=20, feature_type='tfidf')
    #svm_main(search_vals=100, feature_type='nrclex')
    #svm_main(search_vals=20, feature_type='both')
    svm_main(search_vals=20, feature_type='marsyas')