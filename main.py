#import Screen as sc
import warnings
import pandas                  as pd
import numpy                   as np
import knn                     as knn
import decision_tree           as dt
import naive_bayes             as nb
import svm                     as svm
import multilayer_perceptron   as mlp
import sklearn.model_selection as skms

from scipy.stats           import mannwhitneyu
from sklearn.utils.testing import ignore_warnings

def main():
    warnings.simplefilter("ignore", UserWarning)
    csv = pd.read_csv('Glass.csv', sep=',')
    
    #Coluna responsavel por classicar as classes de dados(parametro usado no STRATIFY{serve para manter a proporção dos elementos na hora de realizar as divisões})
    classes = csv['Class']

    """
        realiza uma divisão sobre o do conjunto de dados
        database[0]' = conjunto de treino => 75%
        database[1]' = conjutno de teste  => 25%
    """
    database = skms.train_test_split(csv, test_size = 0.25, train_size = 0.75, shuffle = True, stratify=classes)
    k_validation = database[0]
    test = database[1]

    # Target
    target_test = test['Class']
    # Features
    features_test = test
    # Deleta a coluna Target, ou seja, separa ela das Features
    features_test = features_test.drop(['Class'], axis=1)

    clfs = [None, None, None, None, None]
    
    #clfs[0] = choose_best_classifier(knn.findBestKNN(train, validation), features_test, target_test)
    #clfs[1] = choose_best_classifier(dt.decision_tree(train, validation), features_test, target_test)
    #clfs[2] = choose_best_classifier(nb.naive_bayes(train, validation), features_test, target_test)
    #clfs[3] = choose_best_classifier(svm.svm(train, validation), features_test, target_test)
    #clfs[4] = choose_best_classifier(mlp.my_little_poney(train, validation), features_test, target_test)

    clfs[0] = testingClassifiers(knn.knn(k_validation), features_test, target_test)                      #KNN Euclidiano
    clfs[1] = testingClassifiers(dt.decision_tree(k_validation), features_test, target_test)             #Decision-Tree completa(sem poda)
    clfs[2] = testingClassifiers(nb.naive_bayes(k_validation), features_test, target_test)      #Naive-Bayes Bernoulli
    clfs[3] = testingClassifiers(svm.svm(k_validation), features_test, target_test)                      #SMV kernel RBF
    clfs[4] = testingClassifiers(mlp.my_little_poney(k_validation), features_test, target_test) #MLP Constant

    del classes, csv, database, test, target_test, features_test
    return (clfs)

def testingClassifiers(clf, features_test, target_test):
    return clf.score(features_test, target_test)

def mann_whitney(mean):
    # compare samples
    i, j = 0, 0
    while i < 5:
        j = i+1
        while j < 5:
            stat, p = mannwhitneyu(mean[i], mean[j])
            # interpretação
            alpha = 0.05
            if p > alpha:
                if (i == 0 and j == 1): print('KNN tem a mesma distibuição que Decision-Tree                              portanto, não rejeita-se H0') 
                if (i == 0 and j == 2): print('KNN tem a mesma distibuição que Support-Vector-Machines                    portanto, não rejeita-se H0')
                if (i == 0 and j == 3): print('KNN tem a mesma distibuição que Naive-Bayes                                portanto, não rejeita-se H0') 
                if (i == 0 and j == 4): print('KNN tem a mesma distibuição que Multi-Layer-Perceptron                     portanto, não rejeita-se H0') 
                if (i == 1 and j == 2): print('Decision-Tree tem a mesma distibuição que Naive-Bayes                      portanto, não rejeita-se H0') 
                if (i == 1 and j == 3): print('Decision-Tree tem a mesma distibuição que Support-Vector-Machines          portanto, não rejeita-se H0') 
                if (i == 1 and j == 4): print('Decision-Tree tem a mesma distibuição que Multi-Layer-Perceptron           portanto, não rejeita-se H0') 
                if (i == 2 and j == 3): print('Naive-Bayes tem a mesma distibuição que Support-Vector-Machines            portanto, não rejeita-se H0') 
                if (i == 2 and j == 4): print('Naive-Bayes tem a mesma distibuição que Multi-Layer-Perceptron             portanto, não rejeita-se H0') 
                if (i == 3 and j == 4): print('Support-Vector-Machines tem a mesma distibuição que Multi-Layer-Perceptron portanto, não rejeita-se H0') 
            else:
                if (i == 0 and j == 1): print('KNN tem a mesma distibuição que Decision-Tree                              portanto, rejeita-se H0') 
                if (i == 0 and j == 2): print('KNN tem a mesma distibuição que Support-Vector-Machines                    portanto, rejeita-se H0')
                if (i == 0 and j == 3): print('KNN tem a mesma distibuição que Naive-Bayes                                portanto, rejeita-se H0') 
                if (i == 0 and j == 4): print('KNN tem a mesma distibuição que Multi-Layer-Perceptron                     portanto, rejeita-se H0') 
                if (i == 1 and j == 2): print('Decision-Tree tem a mesma distibuição que Naive-Bayes                      portanto, rejeita-se H0') 
                if (i == 1 and j == 3): print('Decision-Tree tem a mesma distibuição que Support-Vector-Machines          portanto, rejeita-se H0') 
                if (i == 1 and j == 4): print('Decision-Tree tem a mesma distibuição que Multi-Layer-Perceptron           portanto, rejeita-se H0') 
                if (i == 2 and j == 3): print('Naive-Bayes tem a mesma distibuição que Support-Vector-Machines            portanto, rejeita-se H0') 
                if (i == 2 and j == 4): print('Naive-Bayes tem a mesma distibuição que Multi-Layer-Perceptron             portanto, rejeita-se H0') 
                if (i == 3 and j == 4): print('Support-Vector-Machines tem a mesma distibuição que Multi-Layer-Perceptron portanto, rejeita-se H0') 
            j += 1
        i += 1
    

# Função rodada 5 vezes, e realizado uma analise manual para definir qual é o melho classificador entre os tipos de classificadores retornados
"""
def choose_best_classifier(clfs, features_test, target_test):
    i, best, acc_tmp = 0, [None, 0], 0
    while i < len(clfs):
        acc_tmp = clfs[i].score(features_test, target_test)
        if acc_tmp > best[1]:
            best = [clfs[i], acc_tmp]
        i += 1
    
    return best[0]
"""

i = 1
mean = [[], [], [], [], []]
while i <= 5:
    _knn, _dt, _nb, _svm, _mlp = None, None, None, None, None
    print('Iteration ' + str(i))
    _knn, _dt, _nb, _svm, _mlp = main()
    mean[0].append(_knn)
    mean[1].append(_dt)
    mean[2].append(_nb)
    mean[3].append(_svm)
    mean[4].append(_mlp)
    print("KNN           : " + str(_knn) + "\n" +
          "DT            : " + str(_dt)  + '\n' +
          "Naive-Bayes   : " + str(_nb)  + '\n' +
          "SVM           : " + str(_svm) + '\n' + 
          "MLP           : " + str(_mlp) + '\n')
    i += 1

print('\n')

mann_whitney([mean[0], mean[1], mean[2], mean[3], mean[4]])
mean[0] = np.mean(mean[0])
mean[1] = np.mean(mean[1])
mean[2] = np.mean(mean[2])
mean[3] = np.mean(mean[3])
mean[4] = np.mean(mean[4])
