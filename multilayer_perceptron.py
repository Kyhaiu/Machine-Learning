import pandas as pd
import numpy  as np
import statistics as st
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def my_little_poney(train, validation):
    # Targets
    target_train       = train['Class']
    target_validation  = validation['Class']

    # Features e deleta a coluna Target das Features
    features_train = train.drop(['Class'], axis=1)
    features_validation = validation.drop(['Class'], axis=1)
    
    #retorna o mlp com taxa constant, invscalling, adaptative
    return [find_best_poney(features_train, target_train, features_validation, target_validation)]
    

def find_best_poney(features_train, target_train, features_validation, target_validation):
    i, j, k = 1, 1, 1
    mean = [0.0, 0.0, 0.0]
    classifier_constant, classifier_inv, classifier_adapt = [], [], []
    acc_const, acc_inv, acc_adapt = [], [], []
    stop = [False, False, False]

    while True:
        
        if not(stop[0]):
            clf_constant = MLPClassifier(hidden_layer_sizes=(i, j), learning_rate='constant', max_iter=k)

            classifier_constant.append(clf_constant.fit(features_train, target_train))
            
            acc_const.append(classifier_constant[-1].score(features_validation, target_validation))
            
            mean[0] = st.mean(acc_const)
        if((mean[0] > acc_const[-1] and max(acc_const) > 0.65)):
            stop[0] = True
        
        if not(stop[1]):
            clf_invscaling = MLPClassifier(hidden_layer_sizes=(i, j), learning_rate='invscaling', max_iter=k)
            
            classifier_inv.append(clf_invscaling.fit(features_train, target_train))
            
            acc_inv.append(classifier_inv[-1].score(features_validation, target_validation))
            
            mean[1] = st.mean(acc_inv)
        
        if((mean[1] > acc_inv[-1] and max(acc_inv) > 0.65)):
            stop[1] = True
        
        if not(stop[2]):
            clf_adaptive = MLPClassifier(hidden_layer_sizes=(i, j), learning_rate='adaptive', max_iter=k)
        
            classifier_adapt.append(clf_adaptive.fit(features_train, target_train))
        
            acc_adapt.append(classifier_adapt[-1].score(features_validation, target_validation))
        
            mean[2] = st.mean(acc_adapt)
        
        if((mean[2] > acc_adapt[-1] and max(acc_adapt) > 0.65)):
            stop[2] = True


        if ((i >= 100 and j >= 100 and k >= 300) or (stop[0] and stop[1] and stop[2])):
            break
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
        
    return [classifier_constant[acc_const.index(max(acc_const), 0, -1)], classifier_inv[acc_inv.index(max(acc_inv), 0, -1)], classifier_adapt[acc_adapt.index(max(acc_adapt), 0, -1)]]