import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

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

def decision_tree(train, validation, test):
   # Target Class
   target_tr = train['Class']
   target_v  = validation['Class']

   # Features
   features_tr = train
   features_v  = validation

   # Deleta a coluna Target das Features
   ft_train = features_tr.drop(['Class'], axis=1)
   ft_validation = features_v.drop(['Class'], axis=1)
   
   # o classificador encontra padrões nos dados de treinamento
   clf = tree.DecisionTreeClassifier() # instância do classificador
   clf = clf.fit(ft_train, target_tr) # fit encontra padrões nos dados
   plot(clf)


def plot(clf):
   fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
   fn = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
   tree.plot_tree(clf,
                  feature_names = fn,
                  class_names = ['1', '2', '3', '4', '5', '6'],
                  filled = True,
                  );
   fig.savefig('Result.png')

"""
   Metodo de gerar a arvore de decisão utilizando o ganho de informação
      Ganho de Informação =  Entropia_pai - Sum(peso_filho * entropio_filho)

      
      Entropia = - (Sum(pi * log2(pi))), onde pi = probalidade de ocorrencia
      
      peso_filho = nº de amostra do filho/nº de amostrar do filho
      
      *Entropia = Entropia_var_Decisão(é testado para todas as features e pega a que tem maior valor e repete esse processo até conseguir um ganho de informação = 0 em todas as features)
""" 