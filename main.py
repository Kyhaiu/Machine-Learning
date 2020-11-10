#import Screen as sc
import pandas                as pd
import numpy                 as np
import knn                   as knn
import decision_tree         as dt
import naive_bayes           as nb
import svm                   as svm
import multilayer_perceptron as mlp

import sklearn.model_selection as skms

def main():
    csv = pd.read_csv('Glass.csv', sep=',')
    
    #Coluna responsavel por classicar as classes de dados(parametro usado no STRATIFY{serve para manter a proporção dos elementos na hora de realizar as divisões})
    classes = csv['Class']

    """
        realiza uma divisão sobre o do conjunto de dados
        database[0]' = conjunto de treino => 75%
        database[1]' = conjutno de teste  => 25%
    """
    database = skms.train_test_split(csv, test_size = 0.75, train_size = 0.25, shuffle = True, stratify=classes)
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
    #clfs[1] = testingClassifiers(dt.decision_tree(train, validation)[1], features_test, target_test)    #Decision-Tree completa(sem poda)
    #clfs[2] = testingClassifiers(nb.naive_bayes(train, validation)[1], features_test, target_test)      #Naive-Bayes Bernoulli
    #clfs[3] = testingClassifiers(svm.svm(train, validation)[1], features_test, target_test)             #SMV kernel RBF
    #clfs[4] = testingClassifiers(mlp.my_little_poney(train, validation)[0], features_test, target_test) #MLP Constant

    del classes, csv, database, test, target_test, features_test
    return (clfs)

def testingClassifiers(clf, features_test, target_test):
    return clf.best_estimator_.score(features_test, target_test)

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
while i <= 20:
    _knn, _dt, _nb, _svm, _mlp = None, None, None, None, None

    print('Iteration ' + str(i))
    _knn, _dt, _nb, _svm, _mlp = main()
    print("KNN           : " + str(_knn) + "\n" +
          "DT            : " + str(_dt)  + '\n' +
          "Naive-Bayes   : " + str(_nb)  + '\n' +
          "SVM           : " + str(_svm) + '\n' + 
          "MLP           : " + str(_mlp) + '\n')
    i += 1