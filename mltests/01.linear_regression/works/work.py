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



#一般梯度下降法
def grad_scent(x, y, max_cycle, alpha):
    x_size = len(x)
    ones = np.ones((x_size, 1))
    x = x.reshape((x_size, -1))
    X = np.mat(np.c_[ones, x])
    Y = np.mat(y).T
    #损失值
    loss_list = []
    # 初始化theta2
    theta2 = np.mat(np.ones((X.shape[1], 1)))
    # 迭代
    for i in range(max_cycle):
        Y_pred = X * theta2
        # 误差
        error = Y_pred - Y
        #计算损失值
        loss = np.sum(np.multiply(error, error)) / x_size
        loss_list.append(loss)
        theta2 = theta2 - alpha * X.T * error
    return theta2, loss_list





if __name__ == '__main__':
    year, x, y = load_datasets('assignment.xlsx')
    #最小二乘法
    theta = least_square(x, y)
    results = get_results(x, theta)

    #最大迭代次数
    max_cycle = 100
    #学习率
    alpha = 0.0000001
    loss_lists = []
    #一般梯度下降法,改变alpha观察损失
    for a in np.arange(alpha, alpha + 0.000001, 0.0000001):
        theta2, loss_list = grad_scent(x, y, max_cycle, a)
        loss_lists.append(loss_list[-1])

    #results2 = get_results(x, theta2)
    #np.set_printoptions(suppress = True)

    print(loss_lists)
    plt.figure()
    plt.title('loss and alphas')
    #plt.title('loss and iterations')
    plt.xlabel('alphas')
    #plt.xlabel('iteration_times')
    plt.ylabel('loss')
    plt.plot(np.arange(alpha, alpha + 0.000001, 0.0000001), loss_lists, c = 'r')
    #plt.plot(range(max_cycle), loss_list, c = 'r')
    #plt.scatter(x, y)
    plt.grid()
    plt.show()