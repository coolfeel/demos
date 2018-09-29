
# encoding:utf-8

from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

str_to_num = []             #将花种类对应为0,1,2，用来划分类线
colors = 'rgb'              #显示测试集数据的颜色

irisdata = pd.read_csv('8.iris.data')


for i in range(irisdata.shape[0]):              #将花的种类映射为0,1,2数字化
    if irisdata.values[i, 4] == 'Iris-setosa':
        str_to_num.append(0)
    elif irisdata.values[i, 4] == 'Iris-versicolor':
        str_to_num.append(1)
    elif irisdata.values[i, 4] == 'Iris-virginica':
        str_to_num.append(2)



X = irisdata.values[:, : 2]   #只有前2个属性数据值，前2列，
y = str_to_num     #取结果值


clf = tree.DecisionTreeClassifier().fit(X, y)

axis_x, axis_y = np.meshgrid(np.arange(4, 8, 0.005), np.arange(1, 5, 0.005)) #用坐标形成网格状，即矩阵

pdt = clf.predict(np.c_[axis_x.ravel(), axis_y.ravel()])  #根据前2个属性 来预测结果

result = pdt.reshape(axis_x.shape)        #将结果也化为矩阵

bgd = plt.contourf(axis_x, axis_y, result, alpha = 0.5)       #画分类线

plt.xlabel('length')
plt.ylabel('width')



for i in range(irisdata.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], c = colors[y[i]])        #对每个测试集画散点图，并按照0,1,2在颜色中分配颜色

plt.show()


