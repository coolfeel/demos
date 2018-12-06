
from mxnet import autograd, nd
from matplotlib import pyplot as plt
import numpy as np
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon



num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

np.set_printoptions(suppress = True)

features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1]

labels += nd.random.normal(scale = 0.01, shape = (labels.shape))


batch_size = 10
dataset = gdata.ArrayDataset(features, labels)

data_iter = gdata.DataLoader(dataset, batch_size, shuffle = True)


net = nn.Sequential()

#定义线性回归，又即单层神经网络，也即全连接层
net.add(nn.Dense(1))
#初始均值为0，标准差为0.01的正态分布，偏差参数默认初始化为0
net.initialize(init.Normal(sigma = 0.01))
#定义平方损失，又称L2范数损失
loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : 0.03})

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    #print(l.mean().asnumpy())

dense = net[0]
print(dense.weight.data())
print(dense.bias.data())













