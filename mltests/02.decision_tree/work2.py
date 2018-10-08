# -*- coding:utf-8 -*-

from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


irisdata = pd.read_csv('8.iris.data', header = -1)

#将花的种类对应为0,1,2，用于画分类线
flower_species = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

#构建1个列表，存放花的种类的数字化值，将花种类对应为0，1，2
str_to_num = [flower_species[irisdata.values[i, 4]] for i in range(irisdata.shape[0])]

#选取前2个列属性的值
X = irisdata.values[:, : 2]

#取结果值
y = str_to_num

#将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#存放错误率
error_rate_list = []

#获得不同深度的错误率
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