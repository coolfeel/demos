
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

colors = 'byg'

iris = datasets.load_iris()


X = iris.data[:, [0, 1]]
y = iris.target

clf = tree.DecisionTreeClassifier().fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  #花萼长
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  #花萼宽

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
print(xx.shape)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z.shape)
Z = Z.reshape(xx.shape)
print(Z)

cs = plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.5)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])



for i, color in zip(range(3), colors):
    idx = np.where(y == i)                     #返回y的值在target中的index
    plt.scatter(X[idx, 0], X[idx, 1], c = color, label = iris.target_names[i], cmap = plt.cm.Paired)

plt.show()
