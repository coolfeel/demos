
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

h = 0.02
camp_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
camp_bold = ListedColormap(['r', 'g', 'b'])

for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors = 15, weights = weights)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = camp_light)
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = camp_bold, edgecolors = 'k', s = 20)



plt.show()