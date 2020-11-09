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

def knn(cnj1, cnj2, k):
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

    neigh_np = KNeighborsClassifier(n_neighbors=k)
    neigh_ecl_inv = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=2)
    #neigh_pond = ???

    neigh_np = neigh_np.fit(features_cnj1, target_cnj1)
    neigh_ecl_inv = neigh_ecl_inv.fit(features_cnj1, target_cnj1)

    acc_np = neigh_np.score(features_cnj2, target_cnj2)
    acc_ecl_inv = neigh_ecl_inv.score(features_cnj2, target_cnj2)
    
    #plot(neigh_fit, features_cnj2, target_cnj2)
    return [[acc_np, acc_ecl_inv], [neigh_np, neigh_ecl_inv]]


def findBestKNN(train, validation):
    i, best = 1, [[0, 0], [None, None], [None, None]]
    while(i < len(train)):
        tmp = knn(train, validation, i)

        if tmp[0][0] > best[0][0]:
            best[0][0] = tmp[0][0] #acuracia do neigh_np
            best[1][0] = tmp[1][0] #classificador do neigh_np
            best[2][0] = i         #valor de k usado
        
        if tmp[0][1] > best[0][1]:
            best[0][1] = tmp[0][1] #acuracia do neigh_ecl_inv
            best[1][1] = tmp[1][1] #classificador do neigh_ecl_inv
            best[2][1] = i         #valor de k usado
        
        i+=1
    #print('KNN_NP      = Acc : ' + str(best[0][0]) + '\t, K : ' + str(best[2][0]))
    #print('KNN_ECL_INV = Acc : ' + str(best[0][1]) + '\t, K : ' + str(best[2][1]))

    return best

"""
def plot(neigh_fit, features_v, target_v):
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(neigh_fit, features_v, target_v, display_labels=['1', '2', '3', '4', '5', '6'], cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()
"""