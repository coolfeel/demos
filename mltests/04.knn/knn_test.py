
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter



raw_data_X = [[3.393533211, 2.331273381], [3.110073483, 1.781539638], [1.343808831, 3.368360954], [3.582294042, 4.679179110],
              [2.280362439, 2.866990263], [7.423436942, 4.696522875], [5.745051997, 3.533989803], [9.172168622, 2.511101045],
              [7.792783481, 3.424088941], [7.939820817, 0.791637231]]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

#list转数组，数组减数组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c = 'g')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c = 'r')

x = np.array([8.093607318, 3.365731514])

plt.scatter(x[0], x[1], c = 'b')

distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]

nearest = np.argsort(distances)

k = 6

topk = [y_train[i] for i in nearest[: 6]]

votes = Counter(topk)

pred = votes.most_common(1)[0][0]

print(pred)





