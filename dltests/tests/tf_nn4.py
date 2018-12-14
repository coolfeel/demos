
import tensorflow as tf
import numpy as np



batch_size = 8

x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x_input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y_input')

w1 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))
y = tf.matmul(x, w1)

#定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), loss_more * (y - y_), loss_less * (y_ - y)))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = np.random.RandomState(1)

dataset_size = 128

np.set_printoptions(suppress = True)

X = rdm.rand(dataset_size, 2)
#定义回归的正确值是2数之和加个噪音
Y = [[x1 + x2 + rdm.rand() / 10 - 0.05] for (x1, x2) in X]

with tf.Session() as sess:
    init_var = tf.initialize_all_variables()
    sess.run(init_var)
    STEP = 5000
    for i in range(STEP):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict = {x : X[start : end], y_ : Y[start : end]})
        print(sess.run(w1))

