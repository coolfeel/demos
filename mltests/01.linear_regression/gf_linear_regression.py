
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


diabetes = datasets.load_diabetes()   #糖病患者数据

diabetes_X = diabetes.data[:, np.newaxis, 2] #取数据集的第3列

diabetes_X_train = diabetes_X[: -20]
diabetes_X_test = diabetes_X[-20 :]

diabetes_y_train = diabetes.target[: -20] #target为结果
diabetes_y_test = diabetes.target[-20 :]


regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabetes_X_test)  #用测试集做预测

print('Coefficients: \n',regr.coef_)

print('mean_squared_error: %.2f' % (mean_squared_error(diabetes_y_test, diabetes_y_pred)))  #真实值与预测值的均方差

print('variance score: %.2f' % (r2_score(diabetes_y_test, diabetes_y_pred)))   #用R方评估 1是最好

plt.scatter(diabetes_X_test, diabetes_y_test, color = 'black')  #测试集即真实值

plt.plot(diabetes_X_test, diabetes_y_pred, color = 'blue', linewidth = 3)


plt.show()








