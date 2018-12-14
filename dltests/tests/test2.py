
import numpy as np
import tensorflow as tf


# a = [[1, 2], [2, 3], [3, 3]]
#
# print(a[0 : 2])



# v = tf.constant([[1.0, 1.0], [2.0, 2.0]])
# with tf.Session() as sess:
#
#     # print(sess.run(v))
#     # a = tf.clip_by_value(v, 2.5, 5.5)
#     # print(sess.run(a))
#     # print(a.eval())
#     print(tf.reduce_mean(v).eval())



v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

with tf.Session() as sess:
    print(tf.greater(v1, v2).eval())
    print(tf.where(tf.greater(v1, v2), v1, v2).eval())