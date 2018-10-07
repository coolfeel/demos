# encoding:utf-8

import numpy as np
import scipy
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_data(n):

    np.random.seed(0)

    #样本
    X = 5 * np.random.rand(n, 1)

    #样本对应的值
    y = np.sin(X).ravel()

    noise_num = (int)(n / 5)

    #一行数据，每过5个加数
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))

    #分别对应训练样本集，测试样本及其对应的值
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    return X_train, X_test, y_train, y_test



def test_Decision_Tree(X_train, X_test, y_train, y_test):
    cls = DecisionTreeRegressor()
    cls.fit(X_train, y_train)

    print('train_score:%f' % (cls.score(X_train, y_train)))
    print('test_score:%f' % (cls.score(X_test, y_test)))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.show()
    X = np.arange(0.0, 5.0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    #newaxis增加维数
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = cls.predict(X)

    ax.scatter(X_train, y_train, label = 'train sam', c = 'g')
    ax.scatter(X_test, y_test, label = 'test sam', c = 'r')

    ax.plot(X, Y, label = 'predict_value', linewidth = 2, alpha = 0.5)
    plt.show()



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = create_data(1000)
    test_Decision_Tree(X_train, X_test, y_train, y_test)