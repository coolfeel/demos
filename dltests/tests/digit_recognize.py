
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



#MNIST数据集相关的常数
#输入层的节点数，对于MNIST数据集，这个就等于图片的像素
INPUT_NODE = 784
#输出层节点数，这个等于类别的数目
OUTPUT_NODE = 10
#配置神经网络的参数
#隐藏层节点数，这里只使用1个隐藏层的网络结构作为样例，隐藏层有500个节点
LAYER1_NODE = 500
#1个训练batch中的训练数据个数，数字越小，训练过程越接近随机梯度下降。数字越大，越接近梯度下降
BATCH_SIZE = 100

#基础的学习率
LEARNING_RATE_BASE = 0.8
#学习率的衰减率
LEARNING_RATE_DECAY = 0.99
#描述模型复杂度的正则化在损失函数中的系数
REGULARIZATION_RATE = 0.0001
#训练轮数
TRAINING_STEPS = 30000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

#一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果，在
#这里定义1个使用ReLU激活函数的三层全连接NN,通过加入隐藏层实现了多层网络结构，
#通过ReLU激活函数实现了非线性化，在这个函数中也支持传入用于计算参数平均的值的类
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层的前向传播结果，这里使用可ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        #计算输出层的前向传播结果，因为在计算损失函数时会一并计算softmax函数
        #所以这里不需要加入激活函数，而且不加入softmax不会影响预测结果，因为预测时使用的是不同类别对应节点输出值的相对大小
        #有没有softmax层对最后分类结果的计算没有影响，于是在计算整个NN前向传播时可以不加入最后的softmax层
        return tf.matmul(layer1, weights2) + biases2
    else:
        #首先使用avg_class.average函数计算得出变量的滑动平均值
        #然后计算相应的NN前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x_input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y_input')
    #生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))
    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
    #计算在当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均的类为None
    #所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    #定义存储训练轮数的变量。这个变量不需要计算滑动平均值。所以这里指定这个变量为不可训练的变量
    global_step = tf.Variable(0, trainable = False)
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类，给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在所有代表神经网络参数的变量上使用滑动平均，其他辅助变量global_step就不需要了
    #tf.trainable_varibales返回的就是图上集合
    #GraphKeys.TRAINABLE_VARIABLES中的元素，这个集合的元素就是所有没有指定trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())































