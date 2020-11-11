import pandas                  as pd
import numpy                   as np
import statistics              as st
import sklearn.model_selection as skms

from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.testing   import ignore_warnings
from sklearn.exceptions      import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def my_little_poney(train, validation):
    target_tr = train['Class']	
    target_v  = validation['Class']	

    # Features	
    features_tr = train	
    features_v  = validation	

    # Deleta a coluna Target das Features	
    ft_train = features_tr.drop(['Class'], axis=1)	
    ft_validation = features_v.drop(['Class'], axis=1)
    
    #retorna o mlp com taxa constant, invscalling, adaptative
    best_poney_constant, best_poney_inv, best_poney_adapt = find_best_poney(ft_train, target_tr, ft_validation, target_v)

    if best_poney_constant.score(ft_validation, target_v) >= best_poney_inv.score(ft_validation, target_v) and best_poney_constant.score(ft_validation, target_v) >= best_poney_adapt.score(ft_validation, target_v):
        return best_poney_constant
    elif best_poney_inv.score(ft_validation, target_v) >= best_poney_constant.score(ft_validation, target_v) and best_poney_inv.score(ft_validation, target_v) >= best_poney_adapt.score(ft_validation, target_v):
        return best_poney_inv
    elif best_poney_adapt.score(ft_validation, target_v) >= best_poney_constant.score(ft_validation, target_v) and best_poney_adapt.score(ft_validation, target_v) >= best_poney_inv.score(ft_validation, target_v):
        return best_poney_adapt
    

def find_best_poney(features_train, target_train, features_validation, target_validation):
    i, j, k, y = 1, 1, 1, 0
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


        if ((i >= 100 and j >= 100 and k >= 300) or (stop[0] and stop[1] and stop[2]) or (y == 50)):
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
        y += 1
        
    return (classifier_constant[acc_const.index(max(acc_const), 0, -1)], classifier_inv[acc_inv.index(max(acc_inv), 0, -1)], classifier_adapt[acc_adapt.index(max(acc_adapt), 0, -1)])