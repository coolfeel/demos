# -*- coding:utf-8 -*-

from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


irisdata = pd.read_csv('8.iris.data', header = -1)

#显示数据的颜色
colors = 'rgb'

#将花的种类对应为0,1,2，用于画分类线
flower_species = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

#构建1个列表，存放花的种类的数字化值，将花种类对应为0，1，2
str_to_num = [flower_species[irisdata.values[i, 4]] for i in range(irisdata.shape[0])]

#选取前2个特征的值
X = irisdata.values[:, : 2]

#取种类对应的值
y = str_to_num

#训练模型
clf = tree.DecisionTreeClassifier(max_depth = 4).fit(X, y)

#构建坐标矩阵
axis_x, axis_y = np.meshgrid(np.arange(X[:, 0].min(), X[:, 0].max(), 0.01), np.arange(X[:, 1].min(), X[:, 1].max(), 0.01))

#预测结果
pred = clf.predict(np.c_[axis_x.ravel(), axis_y.ravel()])

#将结果也化为矩阵
result = pred.reshape(axis_x.shape)

#画分类线
plt.contourf(axis_x, axis_y, result, alpha = 0.5)

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