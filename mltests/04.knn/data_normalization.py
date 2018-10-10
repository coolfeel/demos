import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# x = np.random.randint(0, 100, size = 100)
# print((x - np.min(x)) / (np.max(x) - np.min(x)))

# x2 = np.random.randint(0, 100, (50, 2))
#
# x2 = np.array(x2, dtype = float)
# print(x2)
# x2[:, 0] = (x2[:, 0] - np.mean(x2[:, 0])) / np.std(x2[:, 0])
# x2[:, 1] = (x2[:, 1] - np.mean(x2[:, 1])) / np.std(x2[:, 1])
#
# plt.scatter(x2[:, 0], x2[:, 1])
# plt.show()


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)

std_scaler = StandardScaler().fit(X_train)

print(std_scaler.transform(X_train))



















