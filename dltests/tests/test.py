
from mxnet import nd
import numpy as np
from mxnet import autograd


#自动求梯度
x = nd.arange(4).reshape((4, 1))
x.attach_grad()
with autograd.record():
    y = 2 * nd.dot(x.T, x)
y.backward()
print(x.grad)

# w = nd.random.normal(scale = 0.01, shape = (2, 1))
#     #初始化bias为0
# b = nd.zeros(shape = (1, ))
# print(b.grad)
# c = nd.ones((1, 1))
# print(c.grad)













# x = nd.arange(12)
#
#
# X = x.reshape((3, 4))
# print(X)
#
#
# print(type(X))
# #numpy.ndarray与 mxnet.ndarray互换
# print(type(X.asnumpy()))






#
# print(x)
#
# print(type(x))
# print(x.shape, x.size)




#所有元素加起来，但结果是数组
#print(type(X.sum()))
#可以把加起来的数字数组，变成数字
#print(type(X.sum().asscalar()))

# a = nd.array([1])

#仅将(1,)类型转化为数字
# print(a.asscalar())


#判别对位相等与否
# print(X == X)






# y = nd.concat(X, X, dim = 0)
# print(y)



# y = nd.dot(X, X.T)
# print(y)




# Y = nd.zeros((2, 3, 4))
# print(type(Y))
#
#
# Y = nd.array([[1], [1], [1]])
# print(Y.shape)
#
# Y_normal = nd.random.normal(0, 1, shape = (3, 4))
# print(type(Y_normal))

# y = X + X
# print(y)
#
# print(X.exp())


