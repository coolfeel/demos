
import numpy as np
from sklearn.cluster import KMeans
import random
import math
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score



def cal_center(datasets, k):
    print(datasets.shape[0])
    k = random.sample([i for i in range(datasets.shape[0])], k)
    cluster_center = []
    for i in range(len(k)):
        cluster_center.append(datasets[k[i]])
    return cluster_center


def update_center(datasets, k, cen, n):
    sample_label = []
    print(cen)
    for j in range(n):
        print(datasets[j])
        min_cen_dis = np.inf
        for i in range(k):
            print('中心: ', cen[i])
            distance = math.sqrt(np.sum((datasets[j] - cen[i]) ** 2))
            if distance < min_cen_dis:
                min_cen_dis = distance
                mark = i
        print('最小距离：', min_cen_dis)
        print('此时中心：', mark)
        sample_label.append(mark)
    new_center = []
    for i in range(k):
        new = []
        sum_x = 0.0000000000000
        sum_y = 0.0000000000000
        for j in range(n):
            if sample_label[j] == i:
                sum_x += datasets[j][0]
                sum_y += datasets[j][1]
        new.append(sum_x / n)
        new.append(sum_y / n)
        new_center.append(new)
    return new_center, sample_label







if __name__ == '__main__':

    # X = np.array([[2, 4], [4, 2], [1, 6], [2, 9], [5, 7], [9, 2], [3, 1], [6, 4]])
    X, y = make_blobs(n_features = 2, n_samples = 100, cluster_std = 1, center_box = (20, 20), random_state = 1)
    y_pred = KMeans(n_clusters = 2, init = 'k-means++', n_init = 10).fit_predict(X)

    center = cal_center(X, 2)
    print(center)
    n = X.shape[0]
    new_c = center
    for i in range(1):
        new_c, labels = update_center(X, 2, new_c, n)
        print('第{}次: label : {}'.format(i, labels))


    # plt.figure(figsize = (5, 5))
    # plt.scatter(X[:, 0], X[:, 1], c = y_pred)
    # plt.show()
    print(accuracy_score(y_pred, labels))
    print(y_pred)















