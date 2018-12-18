
import numpy as np
import tensorflow as tf


# a = [[1, 2], [2, 3], [3, 3]]
#
# print(a[0 : 2])



# v = tf.constant(0.1, shape = [5])
# a = tf.Variable([1, 1, 1, 1, 1])
# with tf.Session() as sess:
#
#     # print(sess.run(v))
#     # a = tf.clip_by_value(v, 2.5, 5.5)
#     # print(sess.run(a))
#     # print(a.eval())
#     print(v.eval())
#     print(a.eval())


#
# v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
# v2 = tf.constant([4.0, 3.0, 2.0, 1.0])
#
# with tf.Session() as sess:
#     print(tf.greater(v1, v2).eval())
#     print(tf.where(tf.greater(v1, v2), v1, v2).eval())

#小批量随机梯度下降
# batch_size = 128
# #STEP =
# x = tf.placeholder(tf.float32, shape = (batch_size, 2), name = 'x_input')
# y_ = tf.placeholder(tf.float32, shape = (batch_size, 1), name = 'y_input')
#
# #loss =
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#
# with tf.Session() as sess:
#     for i in range(STEP):
#         #current_X, current_Y =
#         sess.run(train_step, feed_dict = {x : current_X, y_ : current_Y})



a = tf.constant([1, 1, 2])
b = tf.Variable([2, 2, 3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.add(a, b)))