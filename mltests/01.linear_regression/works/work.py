# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt



#读取数据集
def load_datasets(filename):
    datasets = pd.read_excel(filename, header = 1).values
    years, incomes, pays = datasets[:, 0], datasets[:, 1], datasets[:, 2]
    return years, incomes, pays



#最小二乘法估计
def least_square(x, y):
    x_size = len(x)
    ones = np.ones((x_size, 1))
    x = x.reshape((x_size, -1))
    X = np.mat(np.c_[ones, x])
    Y = np.mat(y).T
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)).dot(X.T), Y)
    return theta



#得到预测y
def get_results(x, theta):
    ones = np.ones((1, len(x)))
    x = np.array(x, ndmin = 2)
    X = np.r_[ones, x]
    theta = np.array(theta)
    a = X * theta
    results = np.sum(a, axis = 0)
    return results



#梯度下降法
def grad_scent(x, y):
    #学习率
    alpha = 0.01
    #最大迭代次数
    max_cycle = 5
    x_size = len(x)
    ones = np.ones((x_size, 1))
    x = x.reshape((x_size, -1))
    X = np.mat(np.c_[ones, x])
    Y = np.mat(y).T

    # 初始化theta2
    theta2 = np.mat(np.ones((X.shape[1], 1)))


    # 迭代
    for i in range(3):
        Y_pred = X * theta2

        # 误差
        error = Y_pred - Y
        a = X.T * error

        theta2 = theta2 - alpha * a

    return theta2





if __name__ == '__main__':
    year, x, y = load_datasets('assignment.xlsx')
    #最小二乘法
    theta = least_square(x, y)
    results = get_results(x, theta)

    #梯度下降
    theta2 = grad_scent(x, y)
    results2 = get_results(x, theta2)

    np.set_printoptions(suppress = True)
    print(results2)
    print(theta2)


    plt.figure()
    plt.plot(x, results2, c = 'r')
    plt.scatter(x, y)
    plt.show()