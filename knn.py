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

def knn(train, validation, k):
    # Target Class
    classes = train['Class']
   
   database = skms.train_test_split(train, test_size = 0.25, train_size = 0.75, shuffle = True, stratify=classes)

   del classes

   target_tr = train['Class']
   target_v  = validation['Class']

   # Features
   features_tr = train
   features_v  = validation

   # Deleta a coluna Target das Features
   ft_train = features_tr.drop(['Class'], axis=1)
   ft_validation = features_v.drop(['Class'], axis=1)

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

def findBestKNN(train, validation):
    i, best = 1, [[0, 0], [None, None], [None, None]]
    while(i < len(train)/2):
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

    return best
