# -*- coding:utf-8 -*-

from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#分类
def classify(X_train, y_train):

    #显示数据的颜色
    colors = 'rgb'

    #训练模型
    clf = tree.DecisionTreeClassifier(max_depth = 5, criterion = 'entropy').fit(X_train, y_train)

    #构建坐标矩阵
    axis_x, axis_y = np.meshgrid(np.arange(X[:, 0].min(), X[:, 0].max(), 0.01), np.arange(X[:, 1].min(), X[:, 1].max(), 0.01))

    #预测结果
    pred = clf.predict(np.c_[axis_x.ravel(), axis_y.ravel()])

    #将结果也化为矩阵
    result = pred.reshape(axis_x.shape)

<<<<<<< HEAD
    #画分类线
    plt.contourf(axis_x, axis_y, result, alpha = 0.5)
=======
#训练模型
clf = tree.DecisionTreeClassifier(max_depth = 5, criterion = 'entropy').fit(X, y)
>>>>>>> d3fa37050579e2e394ae2dbcb9f6a702efae17a8

    #用前2个特征的值画散点图，并按照0,1,2在颜色中分配颜色
    for i in range(irisdata.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], c = colors[y[i]], linewidths = np.random.randint(1, 4))


    plt.title('classification')
    plt.xlabel('length')
    plt.ylabel('width')
    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.grid(alpha = 0.4)
    plt.show()



#评估
def evaluate(X_train, y_train):

    # 存放错误率
    error_rate_list = []

    # 获得不同深度的错误率
    for i in range(1, 15):
        clf = tree.DecisionTreeClassifier(max_depth = i, criterion = 'entropy').fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        error_rate = 1 - accuracy_score(y_test, y_pred)
        error_rate_list.append(error_rate)

    plt.title('depth and overfitting')
    plt.xlabel('depth')
    plt.ylabel('error rate')
    plt.scatter(range(1, 15), error_rate_list)
    plt.plot(range(1, 15), error_rate_list)
    plt.grid()
    plt.show()




if __name__ == '__main__':

    irisdata = pd.read_csv('8.iris.data', header = -1)

    #将花的种类对应为0,1,2，用于画分类线
    flower_species = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

    #构建1个列表，存放花的种类的数字化值，将花种类对应为0，1，2
    str_to_num = [flower_species[irisdata.values[i, 4]] for i in range(irisdata.shape[0])]

    #选取前2个特征的值
    X = irisdata.values[:, : 2]

    #取种类对应的值
    y = str_to_num

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    classify(X_train, y_train)
    evaluate(X_train, y_train)