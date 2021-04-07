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
        """
        print("X_train:")
        print(X_train)
        print("y_train:")
        print(y_train)
        print("X_test:")
        print(X_test)
        print("y_test:")
        print(y_test)
        """
    elif feature_type == 'both':
        X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf = tfidf_features(random=250)
        X_train_nrc, y_train_nrc, X_test_nrc, y_test_nrc = nrclex_features(random=250)
        """
        print("y_train_tfidf:")
        print(y_train_tfidf)
        print("y_train_nrc")
        print(y_train_nrc)
        """
        assert np.all(y_train_tfidf == y_train_nrc), 'Mismatch in y_train'
        assert np.all(y_test_tfidf == y_test_nrc), 'Mismatch in y_test'
        X_train = np.hstack((X_train_tfidf, X_train_nrc))
        X_test = np.hstack((X_test_tfidf, X_test_nrc))
        y_train = y_train_tfidf
        y_test = y_test_tfidf
    elif feature_type == 'marsyas_zero':
        X_train, y_train, X_test, y_test = marsyas_features_zero(random=250)
        """
        print("X_train:")
        print(X_train)
        print("y_train:")
        print(y_train)
        print("X_test:")
        print(X_test)
        print("y_test:")
        print(y_test)
        """
    elif feature_type == 'marsyas_yin':
        X_train, y_train, X_test, y_test = marsyas_features_yin(random=250)
        print(y_test)
        """
        print("X_train:")
        print(X_train)
        print("y_train:")
        print(y_train)
        print("X_test:")
        print(X_test)
        print("y_test:")
        print(y_test)
        """
    elif feature_type == 'marsyas_energy':
        X_train, y_train, X_test, y_test = marsyas_features_energy(random=250)
    elif feature_type == 'marsyas_power':
        X_train, y_train, X_test, y_test = marsyas_features_power(random=250)
    elif feature_type == 'zero_and_energy':
        X_train_zero, y_train_zero, X_test_zero, y_test_zero = marsyas_features_zero(random=250)
        X_train_energy, y_train_energy, X_test_energy, y_test_energy = marsyas_features_energy(random=250)
        assert np.all(y_train_zero == y_train_energy), 'Mismatch in y_train'
        assert np.all(y_test_zero == y_test_energy), 'Mismatch in y_test'
        X_train = np.hstack((X_train_zero, X_train_energy))
        X_test = np.hstack((X_test_zero, X_test_energy))
        y_train = y_train_zero
        y_test = y_test_zero
    elif feature_type == 'marsyas_both':
        X_train_zero, y_train_zero, X_test_zero, y_test_zero = marsyas_features_zero(random=250)
        X_train_yin, y_train_yin, X_test_yin, y_test_yin = marsyas_features_yin(random=250)
        #print("y_train_zero:")
        #print(y_train_zero)
        #print("y_train_yin")
        #print(y_train_yin)
        assert np.all(y_train_zero == y_train_yin), 'Mismatch in y_train'
        assert np.all(y_test_zero == y_test_yin), 'Mismatch in y_test'
        X_train = np.hstack((X_train_zero, X_train_yin))
        X_test = np.hstack((X_test_zero, X_test_yin))
        y_train = y_train_zero
        y_test = y_test_zero
    elif feature_type == 'lyrics_and_zero':
        X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf = tfidf_features(random=250, audio=True)
        X_train_nrc, y_train_nrc, X_test_nrc, y_test_nrc = nrclex_features(random=250, audio=True)
        X_train_zero, y_train_zero, X_test_zero, y_test_zero = marsyas_features_zero(random=250)
        #rows, cols = np.shape(X_train_zero)
        #print("X_train_zero")
        #print(X_train_zero)
        #print("X_train_tfidf")
        #print(X_train_tfidf)
        #print("y_train_zero")
        #print(y_train_zero)
        #print("y_train_tfidf")
        #print(y_train_tfidf)

        X_train = np.hstack((X_train_tfidf, X_train_nrc, X_train_zero))
        X_test = np.hstack((X_test_tfidf, X_test_nrc, X_test_zero))
        y_train = y_train_tfidf
        y_test = y_test_tfidf



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
    with open('models_marsyas_yin/trained_svc.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('models_marsyas_yin/validation_results.txt', 'w') as f:
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
    #svm_main(search_vals=20, feature_type='marsyas_zero')
    #svm_main(search_vals=20, feature_type='marsyas_yin')
    #svm_main(search_vals=20, feature_type='marsyas_energy')
    #svm_main(search_vals=20, feature_type='marsyas_power')
    #svm_main(search_vals=20, feature_type='zero_and_energy')
    #svm_main(search_vals=20, feature_type = 'marsyas_both')
    svm_main(search_vals=20, feature_type='lyrics_and_zero')