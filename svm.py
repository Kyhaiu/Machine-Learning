import Screen as sc
import pandas as pd
import numpy  as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# PARAMS {
#   Valor do erro (C)
#   Tipo de Kernel (Polinomial ou Radial)
# }

def svm(k_validation):
    # Target Class
    target = k_validation['Class']

    # Features
    features = k_validation

    # Deleta a coluna Target, ou seja, separa ela das Features
    features = features.drop(['Class'], axis=1)

    parameters = {'kernel':('linear', 'rbf'), 'C':list(range(1, 21, 1))}

    clf = GridSearchCV(SVC(), parameters, cv=3)
    clf = clf.fit(features, target)

    return clf.best_estimator_