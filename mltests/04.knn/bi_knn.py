
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


raw_data_X = [[3.393533211, 2.331273381], [3.110073483, 1.781539638], [1.343808831, 3.368360954], [3.582294042, 4.679179110],
              [2.280362439, 2.866990263], [7.423436942, 4.696522875], [5.745051997, 3.533989803], [9.172168622, 2.511101045],
              [7.792783481, 3.424088941], [7.939820817, 0.791637231]]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

#list转数组，数组减数组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

x = np.array([8.093607318, 3.365731514])

knn_clf = KNeighborsClassifier(n_neighbors = 6)
knn_clf.fit(X_train, y_train)

#变为2维，矩阵
x_pred = x.reshape(1, -1)

print(knn_clf.predict(x_pred))

