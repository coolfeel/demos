
import tensorflow as tf





def get_weights(shape, lamda):
    #获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为losses的集合中
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    #集合的名字losses，及内容
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamda)(var))
    return var


x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))

batch_size = 8

#定义每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
#神经网络的层数
n_layers = len(layer_dimension)

#维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x

#当前层的节点个数
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    #layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    #输入不算层，2为输入的，生成当前层中权重的变量，并将这个变量的L2正则化损失加入到计算图上的集合
    weight = get_weights([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape = [out_dimension]))
    #使用ReLU
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    #进入下一层之前将下一层的节点个数更新为当前层节点的个数
    in_dimension = layer_dimension[i]


#在定义NN的前向传播的同时已经将所有的L2正则化损失加入了图熵的集合
#在这里只需要计算刻画模型在训练数据熵表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

#将均方误差损失函数加入到损失集合
tf.add_to_collection('losses', mse_loss)

#get_collection返回一个列表，这个列表是所有这个集合中的元素，在该例子中，就是损失函数的不同部分，将他们加起来得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))





