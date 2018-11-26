
import numpy as np


# arr = np.array([[1, 2, 3], [2, 3, 4]])
# print(arr[:, 0, np.newaxis].shape)
# print(arr[np.newaxis, 1].shape)


arr1 = np.array([[1, 2], [2, 1]])
arr1_mat = np.mat(arr1)

arr2 = np.array([[2, 3], [3, 4]])
arr2_mat = np.mat(arr2)

re = arr1_mat * arr2_mat
print(re)

print(arr1.dot(arr2))
print(arr2.dot(arr1))


a = np.zeros(2)
print(a)

