import subprocess
import sys
import shelve
import signal
import time

import numpy as np
import matplotlib.pyplot as plt

class NN:
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_te):
        y_te = []
        for x in X_te:
            d = np.sum((self.X - x)**2, axis=1)
            y_te.append(np.mean(self.y[d <= np.min(d)]))
        return np.array(y_te)

def hyperopt(args, params):
    score_model = NN()
    def handler(signum, frame):
        print '\nBEST:', 
        eval(args, x2params(x_best))
        sys.exit()
    signal.signal(signal.SIGINT, handler)

    def x2params(xs):
        ys = []
        for x, p in zip(xs, params):
            if p['type'] == 'c':
                ys.append(p['categories'][int(x)])
            elif p['type'] == 'i':
                ys.append(int(p['trans'](x)))
            elif p['type'] == 'f':
                ys.append(round(p['trans'](x), 12))
            else:
                assert False
        return ys

    def X2dataset(X):
        X = np.array(X)
        cols = []
        for j, p in enumerate(params):
            if p['type'] == 'c' and len(p['categories']) > 2:
                cols.append(np.eye(len(p['categories']))[X[:,j]])
            elif p['type'] in {'i', 'f'} or \
              (p['type'] == 'c' and len(p['categories']) == 2):
                cols.append(X[:,j])
            else:
                assert False
        return np.column_stack(cols)
        
    # init params
    X = [[0] * len(params)]
    for i, param in enumerate(params):
        if param['type'] == 'c':
            for j in range(1, len(param['categories'])):
                x = list(X[0])
                x[i] = j
                X.append(x)
        elif param['type'] in {'i', 'f'}:
            x = list(X[0])
            x[i] = 1
            X.append(x)
        else:
            assert False
    y_score = np.array([eval(args, x2params(x))['va'] for x in X])

    visited = set(map(tuple, X))
    X_stack = []
    y_stack = []
    while True:
        # candidates
        candidates = set()
        for x in X:
            for j, p in enumerate(params):
                if p['type'] == 'c':
                    for i in range(len(p['categories'])):
                        n = list(x)
                        n[j] = i
                        candidates.add(tuple(n))
                elif p['type'] in {'f', 'i'}:
                    n1 = list(x)
                    n2 = list(x)
                    n1[j] = n1[j] + 1
                    n2[j] = n2[j] - 1
                    candidates.add(tuple(n1))
                    candidates.add(tuple(n2))
                else:
                    assert False

        candidates = np.array(list(candidates - visited))
        np.random.shuffle(candidates)
        
        X_te = X2dataset(candidates)
        X_tr = X2dataset(X)
        score_model.fit(X_tr, y_score)
        score = score_model.predict(X_te)
        x_best = X[np.argmax(y_score)]

        ind = np.argsort(-score)
        for i in ind:
            visited.add(tuple(candidates[i]))
            ps = x2params(candidates[i])

            valid = True
            for p, param in zip(ps, params):
                if ('max' in param and p > param['max']):
                    valid = False
            if not valid:
                continue
                    
            obj = eval(args, ps)
            if obj is not None:
                break

        # append new result
        X.append(candidates[i])
        y_score = np.append(y_score, obj['va'])

        if 1:
            f1 = np.linspace(np.min(X_tr[:,0]) - 2, np.max(X_tr[:,0]) + 2, 40)
            f2 = np.linspace(np.min(X_tr[:,1]) - 2, np.max(X_tr[:,1]) + 2, 40)
            xx, yy = np.meshgrid(f1, f2)
            X_te = np.column_stack((xx.ravel(), yy.ravel()))
            score = score_model.predict(X_te)
            x_best = X2dataset([x_best])[0]
            plt.plot(X_tr[:,0], X_tr[:,1], 'bo')
            plt.plot(x_best[0], x_best[1], 'go')
            plt.plot(X_tr[-1,0], X_tr[-1,1], 'yo')
            plt.imshow(score.reshape((f1.size, f2.size)), extent=(f1.min(), f1.max(), f2.min(), f2.max()), origin='lower', interpolation='None')
            plt.show()
            plt.close()


eval_cache = shelve.open('pkl/eval_{}.pkl'.format('_'.join(sys.argv[1:])))
def eval(method, params):
    params_key = repr(params)
    if params_key not in eval_cache:
        try:
            t = time.time()
            o = subprocess.check_output(['./main.py', method] + map(str, params), stderr=subprocess.PIPE)
            te, va = map(float, o.replace('[', '').replace(']', '').split())
            eval_cache[params_key] = {'te': te, 'va': va, 'time': time.time() - t, 'params': params}
        except subprocess.CalledProcessError:
            eval_cache[params_key] = None
    obj = eval_cache[params_key]
    if obj:
        print obj
    return obj

"""
if sys.argv[1] == 'lr_sk':
    params = [
        {'type': 'c', 'categories': ('l2', 'l1')},  # penalty
        {'type': 'c', 'categories': (0, 1)},  # dual
        {'type': 'f', 'trans': lambda i: 2**i},  # C
        {'type': 'c', 'categories': (1, 0)},  # fit_intercept
        {'type': 'f', 'trans': lambda i: 3**i},  # intercept_scaling
        {'type': 'f', 'trans': lambda i: 2**(i - 10)},  # tol
    ]
    hyperopt('lr_sk', params)

if sys.argv[1] == 'rf_sk':
    params = [
        {'name': 'n_estimators', 'type': 'i', 'trans': lambda i: 2**(i + 7)}, # n_estimators
        {'type': 'c', 'categories': ('gini', 'entropy')}, # criterion
        {'type': 'f', 'trans': lambda i: i * 0.02 + 0.1}, # max_features
    ]
    hyperopt('sk linear_model.LogisticRegression', params)

if sys.argv[1] == 'svm_sk':
    params = [
        {'type': 'f', 'trans': lambda i: 2**(3 * i - 5)},  # g
        {'type': 'f', 'trans': lambda i: 2**(3 * i + 5)},  # c
    ]
    hyperopt('svm_sk', params)

if sys.argv[1] == 'mlp_bfgs':
    params = [
        {'type': 'i', 'trans': lambda i: 2**(0.5 * i + 4)}, # num_hidden
        {'type': 'f', 'trans': lambda i: 2**i}, # lambda_
        {'type': 'i', 'trans': lambda i: 2**(i + 8)}, # maxfun
    ]
    hyperopt('mlp_bfgs', params)

if sys.argv[1] == 'gbm_sk':
    params = [
        {'type': 'f', 'trans': lambda i: 2**(i - 3)},  # learning_rate
        {'type': 'i', 'trans': lambda i: 2**(i + 8)},  # n_estimators
        {'type': 'f', 'trans': lambda i: 0.1 * i + 0.5},  # subsample
        {'type': 'i', 'trans': lambda i: i + 3},  # max_depth
        {'type': 'f', 'trans': lambda i: i * 0.02 + 0.1}, # max_features
    ]
    hyperopt('gbm_sk', params)


if sys.argv[1] == 'mlp_sgd':
    params = [
        {'type': 'i', 'trans': lambda i: 2**(0.5 * i + 7)}, # num_hidden
        {'type': 'f', 'trans': lambda i: 0.2 * i + 0.5}, # dropout
        {'type': 'f', 'trans': lambda i: 2**(i - 10)}, # lambda_
        {'type': 'f', 'trans': lambda i: 2**(i - 3)}, # learning_rate
        {'type': 'i', 'trans': lambda i: 2**(i + 9)}, # iterations
        {'type': 'f', 'trans': lambda i: 10**(i - 2)}, # scale
        {'type': 'i', 'trans': lambda i: 2**(i + 3)}, # batch_size
    ]
    hyperopt('mlp_sgd', params)

if sys.argv[1] == 'ada_sk':
    params = [
        {'type': 'i', 'trans': lambda i: 2**(i + 8), 'max': 2**12}, # n_estimators
        {'type': 'f', 'trans': lambda i: 2**(i - 1)}, # learning_rate
        {'type': 'i', 'trans': lambda i: i + 3},  # max_depth
        {'type': 'c', 'categories': ('SAMME.R', 'SAMME')},  # algorithm
    ]
    hyperopt('ada_sk', params)

if sys.argv[1] == 'ert_sk':
    params = [
        {'type': 'i', 'trans': lambda i: 2**(i + 7)}, # n_estimators
        {'type': 'c', 'categories': ('gini', 'entropy')}, # criterion
        {'type': 'f', 'trans': lambda i: i * 0.02 + 0.1}, # max_features
    ]
    hyperopt('ert_sk', params)
"""
