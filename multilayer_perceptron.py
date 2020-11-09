import pandas as pd
import numpy  as np
import statistics as st
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def my_little_poney(train, validation):
    # Targets
    target_train       = train['Class']
    target_validation  = validation['Class']

    # Features e deleta a coluna Target das Features
    features_train = train.drop(['Class'], axis=1)
    features_validation = validation.drop(['Class'], axis=1)
    

    find_best_poney(features_train, target_train, features_validation, target_validation)
    

def find_best_poney(features_train, target_train, features_validation, target_validation):
    i, j, k, y = 1, 1, 1, 0
    mean = [1.0, 1.0, 1.0]
    classifier_constant, classifier_inv, classifier_adapt = [], [], []
    stop = [False, False, False]

    while True:
        if not(stop[0]):
            clf_constant = MLPClassifier(hidden_layer_sizes=(i, j), learning_rate='constant', max_iter=k)
            classifier_constant.append((clf_constant.fit(features_train, target_train), clf_constant.fit(features_train, target_train).score(features_validation, target_validation)))
            mean[0] = st.mean(classifier_constant[1])
        if((mean[0] > classifier_constant[1][-1] and max(classifier_constant[1]) > 0.75)):
            stop[0] = True
        
        if not(stop[1]):
            clf_invscaling = MLPClassifier(hidden_layer_sizes=(i, j), learning_rate='invscaling', max_iter=k)
            classifier_inv.append((clf_constant.fit(features_train, target_train), clf_constant.fit(features_train, target_train).score(features_validation, target_validation)))
            mean[1] = st.mean(classifier_inv[1])
        if((mean[1] > classifier_inv[1][-1] and max(classifier_inv[1]) > 0.75)):
            stop[1] = True
        
        if not(stop[2]):
            clf_adaptive = MLPClassifier(hidden_layer_sizes=(i, j), learning_rate='adaptive', max_iter=k)
            classifier_adapt.append((clf_constant.fit(features_train, target_train), clf_constant.fit(features_train, target_train).score(features_validation, target_validation)))
            mean[2] = st.mean(classifier_adapt[1])
        if((mean[2] > classifier_adapt[1][-1] and max(classifier_adapt[1]) > 0.75)):
            stop[2] = True

        if (i >= 100):
            i = 1
        else:
            i += 10
        if (j >= 100):
            j = 1
        else:
            j += 10
        if (k >= 300):
            k = 1
        else:
            k += 10
        if (y == 1000):
            break
        y += 1

    print(max(classifier_constant[1]))
    print(max(classifier_inv[1]))
    print(max(classifier_adapt[1]))