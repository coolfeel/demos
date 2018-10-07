
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


#训练集X,y
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis = 0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

#训练模型
regr_1 = DecisionTreeRegressor(max_depth = 2)
regr_2 = DecisionTreeRegressor(max_depth = 5)

regr_1.fit(X, y)
regr_2.fit(X, y)


#测试集X
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)


plt.figure()
plt.scatter(X, y, s = 20, edgecolors = 'black', c = 'darkorange', label = 'data')
plt.plot(X_test, y_1, color = 'cornflowerblue', label = 'maxlength = 2', linewidth = 2)
plt.plot(X_test, y_2, color = 'yellowgreen', label = 'maxlength = 5', linewidth = 3)
plt.xlabel('data')
plt.ylabel('target')
plt.title('decision tree regression')
plt.legend()

plt.show()


