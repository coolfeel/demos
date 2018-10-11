
import numpy as np
from sklearn import preprocessing

arr1 = np.array([[1, 2], [1, 2]])


a = preprocessing.Binarizer(threshold = 0.4).transform(arr1)
print(a)