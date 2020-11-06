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

def knn(cnj1, cnj2, k, _type):
    # Target Class
    target_cnj1 = cnj1['Class']
    target_cnj2 = cnj2['Class']

    # Features
    features_cnj1 = cnj1
    features_cnj2 = cnj2

    # Deleta a coluna Target, ou seja, separa ela das Features
    features_cnj1.drop(['Class'], axis=1)
    features_cnj2.drop(['Class'], axis=1)

    # Valor de K                                        => n_neighbors
    # Métrica de Distância {
    #   não ponderado                                   => weights = 'uniform'
    #   ponderado pelo inverso da distância euclidiana  => weights = 'distance'
    #   ponderado por 1-distância normalizada           => ???
    #}
    # The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    
    if(_type == 0):
        neigh = KNeighborsClassifier(n_neighbors=k)
    else:
        neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=2)

    neigh_fit = neigh.fit(features_cnj1, target_cnj1)

    neigh_pred = neigh.predict(features_cnj2)
    acc = neigh.score(features_cnj2, target_cnj2)

    #plot(neigh_fit, features_cnj2, target_cnj2)
    return acc


def findBestK(train, validation, _type):
    i, best, tmp = 1, (0, 0), 0
    while(i < len(train)):
        if(_type == 0):
            tmp = knn(train, validation, i, 0)
        else:
            tmp = knn(train, validation, i, 1)

        if(tmp > best[0]):
            best = (tmp, i)
        i+=1

    return best


def plot(neigh_fit, features_v, target_v):
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(neigh_fit, features_v, target_v, display_labels=['1', '2', '3', '4', '5', '6'], cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()