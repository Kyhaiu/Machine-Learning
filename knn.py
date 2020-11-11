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

    neigh_np = KNeighborsClassifier(n_neighbors=k)	
    neigh_ecl_inv = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=2)	
    #neigh_pond = ???	

    neigh_np = neigh_np.fit(ft_train, target_tr)	
    neigh_ecl_inv = neigh_ecl_inv.fit(ft_train, target_tr)	

    acc_np = neigh_np.score(ft_validation, target_v)	
    acc_ecl_inv = neigh_ecl_inv.score(ft_validation, target_v)	
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
    if(best[0][0] > best[0][1]):
        return best[1][0]
    else:
        return best[1][1]