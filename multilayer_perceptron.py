import pandas as pd
import numpy  as np
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
    
    parameters = {
                    'learning_rate':('constant', 'invscaling', 'adaptive'),
                    'hidden_layer_sizes':(list(range(100, 350, 1)), (list(range(100, 350, 1)))),
                    'max_iter':list(range(100, 350, 1))
                }
                """ANTES DE RODAR RESUZIR O RANGE PRA NÃO ESTOURAR A MEMORIA DO PC"""
    clf_constant =  MLPClassifier()
    clf = GridSearchCV(clf_constant, parameters, n_jobs=1) #cuidado com os ranges e com esse numero de jobs
    clf.fit(features_train, target_train)
    print(clf.score(features_validation, target_validation))
    #dar uma olhada no randonGridSearch tbm, ver se tem como definir um limiar pra ele parar essa busca