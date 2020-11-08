import Screen as sc
import pandas as pd
import numpy  as np
import sklearn.model_selection as skms
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import plot_confusion_matrix

def svm(cnj1, cnj2):
    # Target Class
    target_cnj1 = cnj1['Class']
    target_cnj2 = cnj2['Class']

    # Features
    features_cnj1 = cnj1
    features_cnj2 = cnj2

    # Deleta a coluna Target, ou seja, separa ela das Features
    features_cnj1.drop(['Class'], axis=1)
    features_cnj2.drop(['Class'], axis=1)
    
# Função pra encontrar a melhor reta de divisão
def findBestStraight():
    pass