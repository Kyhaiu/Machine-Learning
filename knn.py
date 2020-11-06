import Screen as sc
import pandas as pd
import numpy  as np
import sklearn.model_selection as skms
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

# Problema Multiclasse (9) = {
#   A1. RI: refractive index
#   A2. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
#   A3. Mg: Magnesium
#   A4. Al: Aluminum
#   A5. Si: Silicon
#   A6.  K: Potassium
#   A7. Ca: Calcium
#   A8. Ba: Barium
#   A9. Fe: Iron
# } => ESTES SERÃO OS INPUTS

# OUTPUT => {
# Type of glass: (class attribute)
#   1 building_windows_float_processed
#   2 building_windows_non_float_processed
#   3 vehicle_windows_float_processed
#   4 vehicle_windows_non_float_processed (none in this database)
#   5 containers
#   6 tableware
#   7 headlamps
# }

def knn(train, validation, test, k):
    # Target Class
    target_tr = train['Class']
    target_v  = validation['Class']

    # Features
    features_tr = train
    features_v  = validation

    # Deleta a coluna Target, ou seja, separa ela das Features
    features_tr.drop(['Class'], axis=1)
    features_v.drop(['Class'], axis=1)

    # Valor de K                                        => n_neighbors
    # Métrica de Distância {
    #   não ponderado                                   => weights = 'uniform'
    #   ponderado pelo inverso da distância euclidiana  => weights = 'distance'
    #   ponderado por 1-distância normalizada           => ???
    #}
    # The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski')
    neigh_fit = neigh.fit(features_tr, target_tr)

    neigh_pred = neigh.predict(features_v)
    acc = neigh.score(features_v, target_v)

    #plot(neigh_fit, features_v, target_v)
    return acc


def findBestK(train, validation, test):
    i, j, best, med = 0, 1, 0, 0
    while(j < 107):
        while(i < 20):
            med += knn(train, validation, test, j)
            i += 1
        med /= 20
        if(med > best):
            best = med
            print(best, j)
        j += 1
        i, med = 0, 0


def plot(neigh_fit, features_v, target_v):
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(neigh_fit, features_v, target_v, display_labels=['1', '2', '3', '4', '5', '6'], cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()