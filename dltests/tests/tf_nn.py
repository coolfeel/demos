
import tensorflow as tf
import numpy as np

# weights = tf.Variable(tf.random_normal([2, 3], stddev = 2), name = 'cool')
#
# w2 = tf.constant([2, 3])
#
# with tf.Session() as sess:
    #变量要初始化，常量不用
    # sess.run(weights.initializer)
    # print(sess.run(weights))
    # print(sess.run(w2))


w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1), name = 'cool')
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    #初始化所有的变量
    init_var = tf.initialize_all_variables()
    sess.run(init_var)

    print(sess.run(y))



#拿到所有变量
# a = tf.all_variables()
# print(a)