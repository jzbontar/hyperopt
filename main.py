#! /usr/bin/env python2

import sys
import pickle

import numpy as np

from sklearn.metrics import roc_auc_score

from hyperopt import hyperopt

def ss(X, Y, model, n_iter=10, score=roc_auc_score):
    scores = []

    rng = np.random.RandomState(42)
    for i in range(n_iter):
        inds = rng.permutation(X.shape[0])
        n_tr = int(0.6 * X.shape[0])
        n_te = int(0.2 * X.shape[0])
        tr = inds[:n_tr]
        te = inds[n_tr:n_tr + n_te]
        va = inds[n_tr + n_te:]

        model.fit(X[tr], Y[tr])
        score_te = score(Y[te], model.predict_proba(X[te])[:,1])
        score_va = score(Y[va], model.predict_proba(X[va])[:,1])
        scores.append((score_te, score_va))
    return np.mean(scores, axis=0)

np.random.seed(42)
#from sklearn.datasets import make_classification
#X, y = make_classification(n_samples=500, n_features=100, n_informative=10)

#from sklearn.datasets import load_iris
#data = load_iris()
#X, y = data['data'], data['target']
#y = (y == 2).astype(int)

#from sklearn.datasets import fetch_mldata
#data = fetch_mldata('german.numer_scale')
#X, y = data['data'], data['target']

X, Y, y = pickle.load(open('../astrazeneca/pkl/ccris.pkl'))

if sys.argv[1] == 'rf_sk':
    if sys.argv[2] == 'hyperopt':
        params = [
            {'name': 'n_estimators', 'type': 'i', 'trans': lambda i: 2**(i + 7)}, # n_estimators
            {'type': 'c', 'categories': ('gini', 'entropy')}, # criterion
            {'type': 'f', 'trans': lambda i: i * 0.02 + 0.1}, # max_features
        ]
        hyperopt('rf_sk', params)

    else:
        from sklearn.ensemble import RandomForestClassifier

        n_estimators = int(sys.argv[2])
        criterion = sys.argv[3]
        max_features = float(sys.argv[4])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
        )

if sys.argv[1] == 'svm_sk':
    if sys.argv[2] == 'hyperopt':
        params = [
            {'type': 'f', 'trans': lambda i: 2**(2 * i - 5)},  # g
            {'type': 'f', 'trans': lambda i: 2**(2 * i + 5)},  # c
        ]
        hyperopt('svm_sk', params)
    else:
        from sklearn.svm import SVC

        g = float(sys.argv[2])
        c = float(sys.argv[3])

        X = (X - X.mean(axis=0)) / X.std(axis=0)
        model = SVC(probability=True, C=c, gamma=g)


if sys.argv[1] == 'ada_sk':
    if sys.argv[2] == 'hyperopt':
        params = [
            {'type': 'i', 'trans': lambda i: 2**(i + 8), 'max': 2**12}, # n_estimators
            {'type': 'f', 'trans': lambda i: 2**(i - 1)}, # learning_rate
            {'type': 'i', 'trans': lambda i: i + 3},  # max_depth
            {'type': 'c', 'categories': ('SAMME.R', 'SAMME')},  # algorithm
        ]
        hyperopt('ada_sk', params)
    else:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        n_estimators = int(sys.argv[2])
        learning_rate = float(sys.argv[3])
        max_depth = int(sys.argv[4])
        algorithm = sys.argv[5]

        dt_stump = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=1)
        model = AdaBoostClassifier(
            base_estimator=dt_stump,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm)

te, va = ss(X, y, model)
print te, va
