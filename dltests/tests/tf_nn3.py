
import tensorflow as tf
import numpy as np



#定义训练数据的batch
batch_size = 8

#定义参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x_input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y_input')

#定义前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义loss
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成1个数据集
rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
np.set_printoptions(suppress = True)

#2个属性加起来小于1为0类，大于1为1类
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_var = tf.initialize_all_variables()
    sess.run(init_var)

    #[[-0.81131822  1.48459876  0.06532937]
    # [-2.4427042   0.0992484   0.59122431]]
    #print(sess.run(w1))

    #[[-0.81131822]
    #[ 1.48459876]
    #[ 0.06532937]]
    #print(sess.run(w2))

    #定义训练的轮数
    STEPS = 5000

    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        #通过选取的样本训练神经网络并更新参数,x,y_为2个placeholder位置
        sess.run(train_step, feed_dict = {x : X[start : end], y_ : Y[start : end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict = {x : X, y_ : Y})
            print(i, total_cross_entropy)

    print(sess.run(w1))
    print(sess.run(w2))





