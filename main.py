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

    #classificadores treinados
    clfs = [None, None, None, None, None]
    #scores dos classificadores em cima do conjunto de teste
    clfs_scores = [None, None, None, None, None]

    clfs[0] = knn.knn(k_validation)             #KNN Euclidiano
    clfs[1] = dt.decision_tree(k_validation)    #Decision-Tree completa(sem poda)
    clfs[2] = nb.naive_bayes(k_validation)      #Naive-Bayes Bernoulli
    clfs[3] = svm.svm(k_validation)             #SMV kernel RBF
    clfs[4] = mlp.my_little_poney(k_validation) #MLP Constant

    clfs_scores[0] = testingClassifiers(clfs[0], features_test, target_test) #KNN 
    clfs_scores[1] = testingClassifiers(clfs[1], features_test, target_test) #Decision-Tree
    clfs_scores[2] = testingClassifiers(clfs[2], features_test, target_test) #Naive-Bayes
    clfs_scores[3] = testingClassifiers(clfs[3], features_test, target_test) #SMV kernel 
    clfs_scores[4] = testingClassifiers(clfs[4], features_test, target_test) #MLP
    #tmp_sum  = rule_of_sum(clfs, features_test, target_test)                 #Regra da Soma
    #tmp_prod = rule_of_prod(clfs, features_test, target_test)                #Regra do Produto
    tmp_bord = count_board(clfs, features_test, target_test)
    #print(tmp_sum)
    #print(tmp_prod)

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

def count_board(clfs, features, target):
    #clfs[0] -> KNN
    #clfs[1] -> Decision-Tree
    #clfs[2] -> Naive-Bayes
    #clfs[3] -> Suport-Vector-Machine
    #clfs[4] -> Multi-Layer-Perceptron

    number_of_classes = 6
    result  = []
    tmp_knn, tmp_dt, tmp_nb, tmp_svm, tmp_mlp = [], [], [], [], []

    l = 0
    for (a, b, c, d, e) in itertools.zip_longest(clfs[0].predict_proba(features), clfs[1].predict_proba(features), clfs[2].predict_proba(features), clfs[3].predict_proba(features), clfs[4].predict_proba(features)):
        i = 0
        while i < number_of_classes:
            l = np.where(a == max(a))
            tmp_knn.append([a[i], int(l[0])+1, -1])

            l = np.where(b == max(b))
            tmp_dt.append([b[i], int(l[0])+1, -1])
            
            l = np.where(c == max(c))
            tmp_nb.append([c[i], int(l[0])+1, -1])

            l = np.where(d == max(d))
            tmp_svm.append([d[i], int(l[0])+1, -1])

            l = np.where(e == max(e))
            tmp_mlp.append([e[i], int(l[0])+1, -1])
            i += 1
                
        tmp_knn.sort(key=lambda tup: tup[0], reverse=True)
        tmp_dt.sort(key=lambda tup: tup[0], reverse=True)
        tmp_nb.sort(key=lambda tup: tup[0], reverse=True)
        tmp_svm.sort(key=lambda tup: tup[0], reverse=True)
        tmp_mlp.sort(key=lambda tup: tup[0], reverse=True)

    i = 0
    j = 6
    while(i < len(tmp_knn)):
        if(j == 0):
            j = 6
        tmp_knn[i][2] = j
        tmp_dt[i][2] = j
        tmp_nb[i][2] = j
        tmp_svm[i][2] = j
        tmp_mlp[i][2] = j
        i += 1
        j -= 1

    tmp_knn.sort(key=lambda tup: tup[1], reverse=False)
    tmp_dt.sort(key=lambda tup: tup[1], reverse=False)
    tmp_nb.sort(key=lambda tup: tup[1], reverse=False)
    tmp_svm.sort(key=lambda tup: tup[1], reverse=False)
    tmp_mlp.sort(key=lambda tup: tup[1], reverse=False)

    i = 0
    while(i < len(tmp_knn)):
        print(tmp_knn[i])
        i += 1

    return result


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
_knn, _dt, _nb, _svm, _mlp = None, None, None, None, None
kruskal_return = False
while i <= 1:
    print('Iteration ' + str(i))
    _knn, _dt, _nb, _svm, _mlp = main()
    mean[0].append(_knn)
    mean[1].append(_dt)
    mean[2].append(_nb)
    mean[3].append(_svm)
    mean[4].append(_mlp)
    print('\n')
    print("KNN           : " + str(_knn) + "\n" +
          "DT            : " + str(_dt)  + '\n' +
          "Naive-Bayes   : " + str(_nb)  + '\n' +
          "SVM           : " + str(_svm) + '\n' + 
          "MLP           : " + str(_mlp) + '\n')
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
