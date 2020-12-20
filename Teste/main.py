from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as AD
from sklearn import preprocessing
import numpy as np
import pandas as pd
import sklearn as sk


def opt_ad(x_train, y_train, x_valid, y_valid):
    """Obtem o melhor valor de alpha para a poda da AD"""

    ad = AD()

    path = ad.cost_complexity_pruning_path(x_train, y_train)
    alphas = path.ccp_alphas

    res = np.zeros((len(alphas), 1))

    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

    for i, alpha in enumerate(alphas):
        ad = AD(random_state=0, ccp_alpha=alpha)
        ad.fit(x_train, y_train)
        res[i] = sk.metrics.mean_squared_error(ad.predict(x_valid),
                                               y_valid)

    i = res.argmin()

    print(f'SVM = {res[i]}')
    print(f':: alpha = {alphas[i]}')

    return res[i]


def opt_svm(x_train, y_train, x_valid, y_valid):
    """Obtem os melhores valores de C e kernel para o SVM"""

    C = [0.01, 0.1, 1, 10, 100]
    kernel = ['poly', 'rbf']

    res = np.zeros((len(kernel), len(C)))

    for i, k in enumerate(kernel):
        for j, c in enumerate(C):
            svm = SVM(C=c, kernel=k, random_state=0)
            svm.fit(x_train, y_train)
            res[i][j] = sk.metrics.mean_squared_error(svm.predict(x_valid),
                                                      y_valid)

    i, j = np.unravel_index(res.argmin(), res.shape)

    print(f'SVM = {res[i][j]}')
    print(f':: kernel = {kernel[i]}')
    print(f':: C      = {C[j]}')

    return res[i][j]


def opt_mlp(x_train, y_train, x_valid, y_valid):
    """Obtem os melhores parametros para o mlp."""

    hidden_layers = [(100, 100, 100),  # tres camadas com 100 neuronios
                     (100, 100, 100, 100),
                     (100, 100, 100, 100, 100),
                     (100, 100, 100, 100, 100, 100),
                     (100, 100, 100, 100, 100, 100, 100)]

    max_iters = [100, 200, 300, 400]

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

    res = np.zeros((len(hidden_layers),
                    len(max_iters),
                    len(learning_rates)))

    for i, h in enumerate(hidden_layers):
        for j, m in enumerate(max_iters):
            for k, r in enumerate(learning_rates):
                mlp = MLP(max_iter=m,
                          learning_rate_init=r,
                          hidden_layer_sizes=h,
                          random_state=0)
                mlp.fit(x_train, y_train)
                res[i][j][k] = sk.metrics.mean_squared_error(mlp.predict(x_valid),
                                                             y_valid)

    i, j, k = np.unravel_index(res.argmin(), res.shape)

    print(f'MLP = {res[i][j][k]}')
    print(f':: layers = {hidden_layers[i]}')
    print(f':: iter   = {max_iters[j]}')
    print(f':: rate   = {learning_rates[k]}')

    return res[i][j][k]


if __name__ == '__main__':

    data = pd.read_csv('Vehicle/Car_details_v3.csv')
    data.dropna(inplace=True)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(data)

    y = data.pop('selling_price')

    quali_labels = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']
    for ql in quali_labels:
        ohe = preprocessing.LabelEncoder()
        d = data[ql].to_numpy().reshape(-1, 1)
        tql = ohe.fit_transform(d)
        data[ql] = tql
        print(tql)

    print(data)

    x_train, x_test, y_train, y_test = \
        sk.model_selection.train_test_split(data, y,
                                            test_size=0.3,
                                            shuffle=True,
                                            random_state=0)

    reg = LinearRegression().fit(x_train, y_train)
    print(f'RL = {sk.metrics.mean_squared_error(reg.predict(x_test), y_test)}')

    # opt_ad(x_train, y_train, x_test, y_test)
    opt_svm(x_train, y_train, x_test, y_test)
    opt_mlp(x_train, y_train, x_test, y_test)
