
import numpy as np
from math import sqrt
from collections import Counter



raw_data_X = [[3.393533211, 2.331273381], [3.110073483, 1.781539638], [1.343808831, 3.368360954], [3.582294042, 4.679179110],
              [2.280362439, 2.866990263], [7.423436942, 4.696522875], [5.745051997, 3.533989803], [9.172168622, 2.511101045],
              [7.792783481, 3.424088941], [7.939820817, 0.791637231]]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

#list转数组，数组减数组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

x = np.array([[8.093607318, 3.365731514]])



class KNNClassifier:

    def __init__(self, k):
        self.k = k
        self.__X_train = None
        self.__y_train = None


    def fit(self, X_train, y_train):
        self.__X_train = X_train
        self.__y_train = y_train
        return self

    #预测
    def predict(self, X_pred):
        y_pre = [self.__pred(x) for x in X_pred]
        return np.array(y_pre)


    def __pred(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self.__X_train]
        nearest = np.argsort(distances)
        topk = [self.__y_train[i] for i in nearest[: self.k]]
        votes = Counter(topk)
        print(votes)
        return votes.most_common(1)[0][0]




if __name__ == '__main__':

    knn = KNNClassifier(k = 6)
    knn.fit(X_train, y_train)
    y_pre = knn.predict(x)
    print(y_pre[0])
