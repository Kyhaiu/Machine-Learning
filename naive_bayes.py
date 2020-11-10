import pandas as pd
import numpy  as np

import sklearn.model_selection as skms
import sklearn.naive_bayes     as nb

def naive_bayes(train):
    classes = train['Class']
    database = skms.train_test_split(train, test_size = 0.25, train_size = 0.75, shuffle = True, stratify=classes)

    del classes

    target_tr = database[0]['Class']
    target_v  = database[1]['Class']

    # Features
    features_tr = database[0]
    features_v  = database[1]

    # Deleta a coluna Target das Features
    features_tr = features_tr.drop(['Class'], axis=1)
    features_v  = features_v.drop(['Class'], axis=1)
        
    nb_gaus =  nb.GaussianNB().fit(features_tr, target_tr)

    nb_bernoulliNB = nb.BernoulliNB().fit(features_tr, target_tr)

    # Como o conjunto de caracteristicas possuem valores negativos não é possivel utilizar o MultinominalNB
    # Uma solução seria utilizar o GaussianoNB, ou normalizar as caracteristicas entre 0 e 1

    #print('Score of Naive-Bayes Gaussian     : \t' + str(nb_gaus.score(features_v, target_v)))
    #print('Score of Naive-Bayes Bernoulli    : \t' + str(nb_bernoulliNB.score(features_v, target_v)))

    if nb_bernoulliNB.score(features_v, target_v) > nb_gaus.score(features_v, target_v):
        return nb_bernoulliNB
    else:
        return nb_gaus