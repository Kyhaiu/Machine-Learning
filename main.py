#import Screen as sc
import os
import warnings
import itertools
import pandas                  as pd
import numpy                   as np
import knn                     as knn
import decision_tree           as dt
import naive_bayes             as nb
import svm                     as svm
import multilayer_perceptron   as mlp
import sklearn.model_selection as skms

from scipy.stats           import mannwhitneyu, kruskal
from sklearn.utils.testing import ignore_warnings

def main():
    warnings.simplefilter("ignore", UserWarning)
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

    target_test = test['Class']

    # Features
    features_test = test

    # Deleta a coluna Target, ou seja, separa ela das Features
    features_test = features_test.drop(['Class'], axis=1)
    
    clfs = []

    #classificadores treinados
    clfs = [None, None, None, None, None]
    #scores dos classificadores em cima do conjunto de teste
    clfs_scores = [None, None, None, None, None, None, None, None]

    clfs[0] = knn.findBestKNN(train, validation)  #KNN Euclidiano
    clfs[1] = dt.decision_tree(train, validation) #Decision-Tree completa(sem poda)
    clfs[2] = nb.naive_bayes(train, validation)        #Naive-Bayes Bernoulli
    clfs[3] = svm.svm(train, validation)               #SMV kernel RBF
    clfs[4] = mlp.my_little_poney(train, validation)   #MLP Constant

    clfs_scores[0] = testingClassifiers(clfs[0], features_test, target_test) #KNN 
    clfs_scores[1] = testingClassifiers(clfs[1], features_test, target_test) #Decision-Tree
    clfs_scores[2] = testingClassifiers(clfs[2], features_test, target_test) #Naive-Bayes
    clfs_scores[3] = testingClassifiers(clfs[3], features_test, target_test) #SMV kernel 
    clfs_scores[4] = testingClassifiers(clfs[4], features_test, target_test) #MLP
    #temp_sum = VotingClassifier(estimators=[('knn', clfs[0]), ('dt', clfs[0]), ('nb', clfs[0]), ('svm', clfs[0]), ('mlp', clfs[0])], voting='hard') 
    clfs_scores[5] = score(rule_of_sum(clfs, features_test, target_test), target_test)  #Regra da Soma
    clfs_scores[6] = score(rule_of_prod(clfs, features_test, target_test), target_test) #Regra do Produto
    clfs_scores[7] = score(borda_count(clfs, features_test, target_test), target_test)  #RBorda Count


    del classes, csv, database, test, target_test, features_test
    return clfs_scores

def testingClassifiers(clf, features_test, target_test):
    return clf.score(features_test, target_test)

def rule_of_sum(clfs, features, target):
    #clfs[0] -> KNN
    #clfs[1] -> Decision-Tree
    #clfs[2] -> Naive-Bayes
    #clfs[3] -> Suport-Vector-Machine
    #clfs[4] -> Multi-Layer-Perceptron
    number_of_classes = 6
    result  = []
    tmp_sum = [0, 0, 0, 0, 0, 0]

    for (i, j, k, l, m) in itertools.zip_longest(clfs[0].predict_proba(features), clfs[1].predict_proba(features), clfs[2].predict_proba(features), clfs[3].predict_proba(features), clfs[4].predict_proba(features)):
        count = 0
        while count < number_of_classes:
            tmp_sum[count] = i[count] + j[count] + k[count] + l[count] + m[count]
            count += 1
        result.append(tmp_sum.index(max(tmp_sum))+1)

    return result

def rule_of_prod(clfs, features, target):
    #clfs[0] -> KNN
    #clfs[1] -> Decision-Tree
    #clfs[2] -> Naive-Bayes
    #clfs[3] -> Suport-Vector-Machine
    #clfs[4] -> Multi-Layer-Perceptron
    number_of_classes = 6
    result  = []
    tmp_prod = [0, 0, 0, 0, 0, 0]

    for (i, j, k, l, m) in itertools.zip_longest(clfs[0].predict_proba(features), clfs[1].predict_proba(features), clfs[2].predict_proba(features), clfs[3].predict_proba(features), clfs[4].predict_proba(features)):
        count = 0
        while count < number_of_classes:
            tmp_prod[count] = i[count] * j[count] * k[count] * l[count] * m[count]
            count += 1
        result.append(tmp_prod.index(max(tmp_prod))+1)

    return result

def borda_count(clfs, features, target):
    #clfs[0] -> KNN
    #clfs[1] -> Decision-Tree
    #clfs[2] -> Naive-Bayes
    #clfs[3] -> Suport-Vector-Machine
    #clfs[4] -> Multi-Layer-Perceptron

    number_of_classes = 6
    result  = []
    tmp_knn, tmp_dt, tmp_nb, tmp_svm, tmp_mlp = [], [], [], [], []
    k = 0

    tmp_knn.append([])
    tmp_dt.append([])
    tmp_nb.append([])
    tmp_svm.append([])
    tmp_mlp.append([])
    for (a, b, c, d, e) in itertools.zip_longest(clfs[0].predict_proba(features), clfs[1].predict_proba(features), clfs[2].predict_proba(features), clfs[3].predict_proba(features), clfs[4].predict_proba(features)):
        i = 0
        
        while i < number_of_classes:
            tmp_knn[k].append([a[i], i+1, -1])
            
            tmp_dt[k].append([b[i], i+1, -1])
            
            tmp_nb[k].append([c[i], i+1, -1])

            tmp_svm[k].append([d[i], i+1, -1])

            tmp_mlp[k].append([e[i], i+1, -1])
            i += 1

        tmp_knn.append([])
        tmp_dt.append([])
        tmp_nb.append([])
        tmp_svm.append([])
        tmp_mlp.append([])
                
        tmp_knn[k].sort(key=lambda tup: tup[0], reverse=True)
        tmp_dt[k].sort(key=lambda tup: tup[0], reverse=True)
        tmp_nb[k].sort(key=lambda tup: tup[0], reverse=True)
        tmp_svm[k].sort(key=lambda tup: tup[0], reverse=True)
        tmp_mlp[k].sort(key=lambda tup: tup[0], reverse=True)
        k += 1

        i = 0

    i, j, k  = 0, 0, 6
    while(i < (len(tmp_knn) - 1)):
        j = 0
        while (j < 6):
            if(k == 0):
                k = 6
            tmp_knn[i][j][2] = k
            tmp_dt[i][j][2]  = k
            tmp_nb[i][j][2]  = k
            tmp_svm[i][j][2] = k
            tmp_mlp[i][j][2] = k
            k -= 1
            j += 1
        i += 1

    i, j = 0, 0
    while(i < (len(tmp_knn)-1)):
        tmp_knn[k].sort(key=lambda tup: tup[1], reverse=False)
        tmp_dt[k].sort(key=lambda tup: tup[1], reverse=False)
        tmp_nb[k].sort(key=lambda tup: tup[1], reverse=False)
        tmp_svm[k].sort(key=lambda tup: tup[1], reverse=False)
        tmp_mlp[k].sort(key=lambda tup: tup[1], reverse=False)
        i += 1

    i, j = 0, 0
    sum_ranking = [0, 0, 0, 0, 0, 0]
    while (i < (len(tmp_knn)-1)):
        sum_ranking = [0, 0, 0, 0, 0, 0]
        j = 0 
        while (j < 6):
            sum_ranking[j] = tmp_knn[i][j][2] + tmp_dt[i][j][2] + tmp_nb[i][j][2] + tmp_svm[i][j][2] + tmp_mlp[i][j][2]
            #print(' C1 : ' + str(tmp_knn[i][j][2]) + '; C2 : ' + str(tmp_dt[i][j][2]) + '; C3 : ' + str(tmp_nb[i][j][2])+ '; C4 : ' + str(tmp_svm[i][j][2])+ '; C5 : ' + str(tmp_mlp[i][j][2]))            
            j += 1
        i += 1
        result.append(sum_ranking.index(max(sum_ranking)) + 1)

    return result

def score(prevision_targets, true_targets):
    i = 0
    for i, j in itertools.zip_longest(prevision_targets, true_targets):
        if(i == j):
            i += 1
    return (i/len(prevision_targets))



def kruskal_wallis(mean):
    stat, p = kruskal(mean[0], mean[1], mean[2], mean[3], mean[4])
    alpha = 0.05
    print('Hipoteses: \n\t * H0 : não há diferença significativa dos classificadores. \n\t * H1 : há diferença significativa dos classificadores. \n')
    if p > alpha:
        print('Não existe diferença significativa entre os classificadores       , portanto não rejeita-se H0')
    else:
        print('Existe diferença significativa em pelo meno um dos classificadores, portanto rejeita-se H0')
        return True

def mann_whitney(mean):
    print('Hipoteses: \n\t * H0 : não há diferença significativa dos classificadores. \n\t * H1 : há diferença significativa dos classificadores. \n')
    i, j = 0, 0
    while i < 5:
        j = i+1
        while j < 5:
            stat, p = mannwhitneyu(mean[i], mean[j])
            alpha = 0.05
            # interpretação
            if p > alpha:
                if (i == 0 and j == 1): print('KNN é igual ao Decision-Tree                              portanto, não rejeita-se H0') 
                if (i == 0 and j == 2): print('KNN é igual ao Naive-Bayes                                portanto, não rejeita-se H0')
                if (i == 0 and j == 3): print('KNN é igual ao Support-Vector-Machines                    portanto, não rejeita-se H0') 
                if (i == 0 and j == 4): print('KNN é igual ao Multi-Layer-Perceptron                     portanto, não rejeita-se H0') 
                if (i == 1 and j == 2): print('Decision-Tree é igual ao Naive-Bayes                      portanto, não rejeita-se H0') 
                if (i == 1 and j == 3): print('Decision-Tree é igual ao Support-Vector-Machines          portanto, não rejeita-se H0') 
                if (i == 1 and j == 4): print('Decision-Tree é igual ao Multi-Layer-Perceptron           portanto, não rejeita-se H0') 
                if (i == 2 and j == 3): print('Naive-Bayes é igual ao Support-Vector-Machines            portanto, não rejeita-se H0') 
                if (i == 2 and j == 4): print('Naive-Bayes é igual ao Multi-Layer-Perceptron             portanto, não rejeita-se H0') 
                if (i == 3 and j == 4): print('Support-Vector-Machines é igual ao Multi-Layer-Perceptron portanto, não rejeita-se H0') 
            else:
                if (i == 0 and j == 1): print('KNN não é igual ao Decision-Tree                              portanto, rejeita-se H0')
                if (i == 0 and j == 2): print('KNN não é igual ao Naive-Bayes                                portanto, rejeita-se H0')
                if (i == 0 and j == 3): print('KNN não é igual ao Support-Vector-Machines                    portanto, rejeita-se H0') 
                if (i == 0 and j == 4): print('KNN não é igual ao Multi-Layer-Perceptron                     portanto, rejeita-se H0') 
                if (i == 1 and j == 2): print('Decision-Tree não é igual ao Naive-Bayes                      portanto, rejeita-se H0') 
                if (i == 1 and j == 3): print('Decision-Tree não é igual ao Support-Vector-Machines          portanto, rejeita-se H0') 
                if (i == 1 and j == 4): print('Decision-Tree não é igual ao Multi-Layer-Perceptron           portanto, rejeita-se H0') 
                if (i == 2 and j == 3): print('Naive-Bayes não é igual ao Support-Vector-Machines            portanto, rejeita-se H0') 
                if (i == 2 and j == 4): print('Naive-Bayes não é igual ao Multi-Layer-Perceptron             portanto, rejeita-se H0') 
                if (i == 3 and j == 4): print('Support-Vector-Machines não é igual ao Multi-Layer-Perceptron portanto, rejeita-se H0')

            j += 1
        i += 1

i = 1
mean = [[], [], [], [], []]
_knn, _dt, _nb, _svm, _mlp, _sum, _prod, _borda = None, None, None, None, None, None, None, None
kruskal_return = False
while i <= 20:
    print('Iteration ' + str(i))
    _knn, _dt, _nb, _svm, _mlp, _sum, _prod, _borda = main()
    mean[0].append(_knn)
    mean[1].append(_dt)
    mean[2].append(_nb)
    mean[3].append(_svm)
    mean[4].append(_mlp)
    print('\n')
    print("KNN              : " + str(_knn)   + "\n" +
          "DT               : " + str(_dt)    + '\n' +
          "Naive-Bayes      : " + str(_nb)    + '\n' +
          "SVM              : " + str(_svm)   + '\n' + 
          "MLP              : " + str(_mlp)   + '\n' +
          "Regra da Soma    : " + str(_sum)   + '\n' +
          "Regra do Produto : " + str(_prod)  + '\n' +
          "Borda Count      : " + str(_borda) + '\n')
    i += 1

print('Resultado do teste de Kruskal-Wallis')
kruskal_return = kruskal_wallis([mean[0], mean[1], mean[2], mean[3], mean[4]])
if(kruskal_return):
    print('\n' + 'Resultado do teste Mann-Whitney\n')
    mann_whitney([mean[0], mean[1], mean[2], mean[3], mean[4]])

mean[0] = np.mean(mean[0])
mean[1] = np.mean(mean[1])
mean[2] = np.mean(mean[2])
mean[3] = np.mean(mean[3])
mean[4] = np.mean(mean[4])

print('\n' + 'Resultado das médias')
print('Média KNN                    : ' + str(mean[0]))
print('Média Decision-Tree          : ' + str(mean[1]))
print('Média Naive-Bayes            : ' + str(mean[2]))
print('Média Suport-Vector-Machine  : ' + str(mean[3]))
print('Média Multi-Layer-Perceptron : ' + str(mean[4]))
