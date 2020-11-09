import Screen as sc
import pandas as pd
import numpy  as np
import sklearn.model_selection as skms
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# PARAMS {
#   Valor do erro (C)
#   Tipo de Kernel (Polinomial ou Radial)
# }

def svm(cnj1, cnj2):
    # Target Class
    target_cnj1 = cnj1['Class']
    target_cnj2 = cnj2['Class']

    # Features
    features_cnj1 = cnj1
    features_cnj2 = cnj2

    # Deleta a coluna Target, ou seja, separa ela das Features
    features_cnj1 = features_cnj1.drop(['Class'], axis=1)
    features_cnj1 = features_cnj2.drop(['Class'], axis=1)

    clf_poly_C, clf_rbf_C = findBestC(features_cnj1, target_cnj1, features_cnj2, target_cnj2)
    
    return [clf_poly_C, clf_rbf_C] #retorna os svm com os kernels poly e rbf
    
# Função pra encontrar a melhor reta de divisão
def findBestC(train_features, train_target, validation_features, validation_target):
    i = 0.1
    best_poly = [None,0] # Classifier, Score
    best_rbf  = [None,0]
    best      = [0,0]
    while i <= 20:
        clf_poly = make_pipeline(StandardScaler(), SVC(gamma='scale', kernel='poly', C=i))
        clf_poly.fit(train_features, train_target)
        
        clf_rbf  = make_pipeline(StandardScaler(), SVC(gamma='scale', kernel='rbf', C=i))
        clf_rbf.fit(train_features, train_target)

        best[0] = clf_poly.score(validation_features, validation_target)
        best[1] = clf_rbf.score(validation_features, validation_target)

        if(best_poly[1] < best[0]):
            best_poly = [clf_poly, best[0]]
        if(best_rbf[1] < best[1]):
            best_rbf = [clf_rbf, best[1]]

        i+=0.1
    
    return (best_poly, best_rbf)