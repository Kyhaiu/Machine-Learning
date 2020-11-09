#import Screen as sc
import pandas        as pd
import numpy         as np
import knn           as knn
import decision_tree as dt
import naive_bayes   as nb
import svm           as svm

import sklearn.model_selection as skms

def main():
    #Construtor da classe tela(serve só para selecionar o arquivo da base de dados{POR ENQUANTO})
    #screen = sc.Screen()
    #Realiza a importação da base de dados do trabalho
    #csv = pd.read_csv(screen.getFilename(), sep=',')
    csv = pd.read_csv('Glass.csv', sep=',')
    
    #Coluna responsavel por classicar as classes de dados(parametro usado no STRATIFY{serve para manter a proporção dos elementos na hora de realizar as divisões})
    classes = csv['Class']

    """
        realiza o shuffle e divided em 2 conjuntos de tamanho iguais (1/2) => 50% para o conjunto de teste
        database[0] = conjunto de teste
        database[1] = resto do conjunto
    """
    database = skms.train_test_split(csv, test_size = 0.5, train_size = 0.5, shuffle = True, stratify=classes)
    train = database[0]
    classes = database[1]['Class']

    """
        realiza uma segunda divisão sobre o resto do conjunto de dados ((1/2)/2) => 25% para o conjunto de validação e teste
        database[0]' = conjunto de validação
        database[1]' = conjutno de teste
    """
    database = skms.train_test_split(database[1], test_size = 0.5, train_size = 0.5, shuffle = True, stratify=classes)
    validation = database[0]
    test = database[1]
    
    clfs = []

    clfs[0] = knn.findBestKNN(train, validation)
    clfs[1] = dt.decision_tree(train, validation)
    clfs[2] = nb.naive_bayes(train, validation)
    clfs[3] = _svm = svm.svm(train, validation)
    #mlp = mlp.

    testingClassifiers(train, test)

def testingClassifiers(train, test):
    pass

"""
i = 1
while i <= 20:
    _knn, _dt, _nb, _svm = [], (None, None), (None, None), (None, None)

    print('Iteration ' + str(i))
    _knn, _dt, _nb, _svm = main()
    print("KNN Não Ponderado                            : " + str(_knn[0][0]) + "\n" +
          "KNN Ponderado Inverso da Distância Euclidiana: " + str(_knn[0][1]) + '\n' +
          "DT Com Poda                                  : " + str(_dt[0])     + '\n' +
          "DT Sem Poda                                  : " + str(_dt[3])     + '\n' +
          "Naive-Bayes Gaus                             : " + str(_nb[1])     + '\n' +
          "Naive-Bayes Bernoulli                        : " + str(_nb[3])     + '\n' +
          "SVM Polinomial                               : " + str(_svm[0][1]) + '\n' +
          "SVM Radial                                   : " + str(_svm[1][1]) + '\n' + '\n')
    i += 1
"""