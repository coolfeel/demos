
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random
import time


pdData = pd.read_csv('data.csv')
print(pdData)

positive = pdData[pdData['admitted'] == 1]
negative = pdData[pdData['admitted'] == 0]


plt.scatter(positive['Exam1'], positive['Exam2'], s = 30, c = 'b', marker = 'o', label = 'Admitted')
plt.scatter(negative['Exam1'], negative['Exam2'], s = 30, c = 'r', marker = 'x', label = 'Not Admitted')

plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.show()




def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))       #X是每一行取值来的


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))         #X是每一行取值与theta相乘
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


def gradient(X, y, theta):     #偏导
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()  #每个样本的真实值与预测值之差

    for j in range(len(theta.ravel())):      #属性的长度，几个属性,3个
        term = np.multiply(error, X[:, j])  #每个样本的第j个属性，总共有3个属性，，,j=1时，第1个样本和第1列每个值相乘，第2个样本和第1列没个值相乘，，，最后相加，才为第1个梯度。。。
        grad[0, j] = np.sum(term) / len(X)  #再求和

    return grad    #3个梯度




pdData.insert(0, 'Ones', 1)         #在第1列的位置新加1列，值全为1


orig_data = pdData.as_matrix()
cols = orig_data.shape[1]          #shape显示的是行和列，1取列

X = orig_data[:, 0 : cols - 1]   #取前3列 1 ，exam1，exma2
y = orig_data[:, cols - 1 : cols]  #取最后1列，是否录取

theta = np.zeros([1, 3])   #整行取

print(cost(X, y, theta))   #X和theta都是每一行开始取值的




STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    if type == STOP_ITER: return value > threshold
    elif type == STOP_COST: return abs(value[-1] - value[-2] < threshold)
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshold




def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0 : cols - 1]
    y = data[:, cols - 1 :]
    return X, y



# def decent(data, theta, batchSize, stopType, thresh, alpha):
#     init_time = time.time()
#     i = 0
#     k = 0
#     X, y = shuffleData(data)
#     grad = np.zeros(theta.shape)
#     costs = [cost(X, y, theta)]
#
#     while True:
#         grad = gradient(X[k : k + batchSize], y[k : k + batchSize], theta)
#         k += batchSize
#         if k >= n:
#             k = 0
#             X, y = shuffleData(data)
#
#         theta = theta - alpha * grad
#         costs.append(cost(X, y, theta))
#         i += 1
#
#         if stopType == STOP_ITER:
#             value = i
#         elif stopType == STOP_COST:
#             value = costs
#         elif stopType == STOP_GRAD:
#             value = grad
#         if stopCriterion(stopType, value, thresh):
#             break
#
#     return theta, i - 1, costs, grad, time.time() - init_time



# def runExpe(data, theta, batchSize, stopType, thresh, alpha):
#     theta, iter, costs, grad, dur = decent(data, theta, batchSize, stopType, thresh, alpha)
#     name = 'Original' if (data[:, 1] > 2).sum() > 1 else 'Scaled'
#     name += 'data - learning rate : {} -'.format(alpha)
#     if batchSize == n:
#         strDescType = 'Gradient'
#     elif batchSize == 1:
#         strDescType = 'Stochastic'
#     else:
#         strDescType = 'mini-batch ({})'.format(batchSize)
#




