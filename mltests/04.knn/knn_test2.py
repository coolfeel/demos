
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV



digits = datasets.load_digits()

X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)

# some_digit = X[1]
#
# some_digit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary)
# plt.show()
#
# print(y[1])


# knn_clf = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
# knn_clf.fit(X_train, y_train)
# y_pred = knn_clf.predict(X_test)
#
# print(accuracy_score(y_pred, y_test))


# best_score = 0.0
# best_k = -1
# for method in ['uniform', 'distance']:
#     for k in range(1, 11):
#         knn_clf = KNeighborsClassifier(n_neighbors = k, weights = method)
#         knn_clf.fit(X_train, y_train)
#         score = knn_clf.score(X_test, y_test)
#         if score > best_score:
#             best_k = k
#             best_score = score
#             best_method = method
#
# print(best_k, best_score, best_method)

#网格搜索
param_grid = [
    {
        'weights' : ['uniform'],
        'n_neighbors' : [i for i in range(1, 11)]
    },
    {
        'weights' : ['distance'],
        'n_neighbors' : [i for i in range(1, 11)],
        'p' : [i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs = -1, verbose = 4)


grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)















