
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
import numpy as np




num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

np.set_printoptions(suppress = True)
#服从标准差为1的正态分布
features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

#加均值为0，标准差为0.01的正态分布的噪音项
labels += nd.random.normal(scale = 0.01, shape = labels.shape)

#features的每一行和labels的每一行都是向量
#print(features[0])

#
# plt.scatter(features[:, 1].asnumpy(), labels.asnumpy())
# plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #变成list，再洗牌，将indices的元素随机排序
    random.shuffle(indices)
    #一次取10个，使用range
    for i in range(0, num_examples, batch_size):
        #一次切10个
        j = nd.array(indices[i: i + batch_size])
        #take根据索引返回对应元素
        #print(features.take(j), labels.take(j))
        yield features.take(j), labels.take(j)


#线性回归的预测值
def linreg(X, w, b):
    #NDArray， 每一项直接+b
    return nd.dot(X, w) + b


#损失函数
def squared_loss(y_pred, y):
    return (y_pred - y.reshape(y_pred.shape)) ** 2 / 2

#小批量随机梯度下降
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size



if __name__ == '__main__':
    #初始化权重为正态分布随机
    w = nd.random.normal(scale = 0.01, shape = (num_inputs, 1))
    #初始化bias为0
    b = nd.zeros(shape = (1, ))
    #创建梯度，针对w，b求梯度
    w.attach_grad()
    b.attach_grad()

    lr = 0.03
    num_epochs = 300
    batch_size = 10


    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                #针对该表达式对w,b求梯度！！！！
                loss = squared_loss(linreg(X, w, b), y)
            loss.backward()
            #w,b已求完梯度，在sgd中直接调用
            sgd([w, b], lr, batch_size)
        train_loss = squared_loss(linreg(features, w, b), labels)
        #print(train_loss.mean().asnumpy())
    #拟合出来的参数
    print(w, b)





