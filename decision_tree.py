import pandas                  as pd
import numpy                   as np
import matplotlib.pyplot       as plt
import sklearn.model_selection as skms

from sklearn import tree


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

"""
   Metodo de gerar a arvore de decisão utilizando o ganho de informação
      Ganho de Informação =  Entropia_pai - Sum(peso_filho * entropio_filho)

      
      Entropia = - (Sum(pi * log2(pi))), onde pi = probalidade de ocorrencia
      
      peso_filho = nº de amostra do filho/nº de amostrar do filho
      
      *Entropia = Entropia_var_Decisão(é testado para todas as features e pega a que tem maior valor e repete esse processo até conseguir um ganho de informação = 0 em todas as features)
""" 

def decision_tree(train, validation):

   target_tr = train['Class']
   target_v  = validation['Class']

   # Features
   features_tr = train
   features_v  = validation

   # Deleta a coluna Target das Features
   ft_train = features_tr.drop(['Class'], axis=1)
   ft_validation = features_v.drop(['Class'], axis=1)

   best_params = find_best_pruning(ft_train, target_tr, ft_validation, target_v)

   if best_params[0].score(ft_validation, target_v) > best_params[1].score(ft_validation, target_v):
      return best_params[0]#arvore podada
   else:
      return best_params[1]#arvore completa


def find_best_pruning(features_train, targets_train, features_validation, target_validation):
 
   full_decision_tree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
   full_decision_tree = full_decision_tree.fit(features_train, targets_train) 

   i, tmp_acc, best = 2, 0, (0, 0)
   #print('Depth of full tree = ' + str(full_decision_tree.get_depth()))
   #print('Score of full tree = ' + str(full_decision_tree.score(features_validation, target_validation)) + '\n\n')

   #print('Finding best point to pruning (Pre-Pruning):\n')
   while i < full_decision_tree.get_depth():
      
      dtc = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_leaf_nodes=i) # instância do classificador
      dtc = dtc.fit(features_train, targets_train) # fit encontra padrões nos dados
      tmp_acc = dtc.score(features_validation, target_validation)

      if tmp_acc > best[0]:
         best = (tmp_acc, i, dtc)
         #print('Tree ' + ' with pruning in ' + str(i) + ': Score = ' + str(best[0]) + '\t, Max_Depth = ' + str(best[1]))
         
      i += 1

   return (best[2], full_decision_tree) # melhores parametros juntamente com sua arvore gerada e a arvore completa

"""
def plot(dtc):
   fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4, 4), dpi=1000)
   fn = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
   tree.plot_tree(dtc,
                  feature_names = fn,
                  class_names = ['1', '2', '3', '4', '5', '6'],
                  filled = True,
                  );
   fig.savefig('Result.png')
""" 