
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


plt.figure(figsize = (12, 12))

n_samples = 100
random_state = 1

X, y = make_blobs(n_samples = n_samples, center_box = (20, 20), cluster_std = 1, random_state = random_state)

np.set_printoptions(threshold = np.inf, suppress = True)
print(X)
print(y)

y_pred = KMeans(n_clusters = 3, init = 'k-means++', n_init = 10).fit_predict(X)
np.set_printoptions(threshold = np.inf)
print(y_pred)
acc = accuracy_score(y, y_pred)
print(acc)


plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c = y_pred)
plt.title('true numbers of blobs')
plt.show()