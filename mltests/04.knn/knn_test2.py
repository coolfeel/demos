
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



digits = datasets.load_digits()

X = digits.data
y = digits.target

# some_digit = X[1]
#
# some_digit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary)
# plt.show()
#
# print(y[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)
knn_clf = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

print(accuracy_score(y_pred, y_test))

