
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, nn
import os
import sys




net = nn.Sequential()

net.add(
        #使用较大的11 * 11窗口来捕获物体，同时使用步长为4来较大减少输出高和宽
        nn.Conv2D(96, kernel_size = 11, strides = 4, activation = 'relu'),
        nn.MaxPool2D(pool_size = 3, strides = 2),
        #减少卷积窗口,使用填充为2来使得输入输出高宽一致，且增大通道数
        nn.Conv2D(256, kernel_size = 5, padding = 2, activation = 'relu'),
        nn.MaxPool2D(pool_size = 3, strides = 2),
        #连接3个卷积层，且使用更小的卷积窗口，除了最后的卷积层外，进一步扩大输出通道数
        #前2个卷积层后不适用池化层来减小输入的高和宽
        nn.Conv2D(384, kernel_size = 3, padding = 1, activation = 'relu'),
        nn.Conv2D(384, kernel_size = 3, padding = 1, activation = 'relu'),
        nn.Conv2D(256, kernel_size = 3, padding = 1, activation = 'relu'),
        nn.MaxPool2D(pool_size = 3, strides = 2),
        #全连接
        nn.Dense(4096, activation = 'relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation = 'relu'), nn.Dropout(0.5),
        #输出层,fashion-minst数据集分10类
        nn.Dense(10)
)

#1个单通道 224 * 224的数据样本
X = nd.random.uniform(shape = (1, 1, 227, 227))
#初始化
net.initialize()
#迭代每一层
for layer in net:
    X = layer(X)
    print(layer.name, X.shape)