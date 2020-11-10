import Screen as sc
import pandas as pd
import numpy  as np
import sklearn.model_selection as skms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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

def knn(k_validation):
    # Target Class
    target = k_validation['Class']

    # Features
    features = k_validation

    # Deleta a coluna Target, ou seja, separa ela das Features
    features = features.drop(['Class'], axis=1)

    # Valor de K                                        => n_neighbors
    # Métrica de Distância {
    #   não ponderado                                   => weights = 'uniform'
    #   ponderado pelo inverso da distância euclidiana  => weights = 'distance'
    #   ponderado por 1-distância normalizada           => ???
    #}
    # The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    parameters = {'n_neighbors':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21], 'weights':('distance', 'uniform')}

    clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=3)
    clf = clf.fit(features, target)

    return clf.best_estimator_

"""
def plot(neigh_fit, features_v, target_v):
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None)]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(neigh_fit, features_v, target_v, display_labels=['1', '2', '3', '4', '5', '6'], cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()
"""