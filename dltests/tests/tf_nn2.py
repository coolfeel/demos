
import tensorflow as tf



w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev = 1))
#定义1个位置,存放数据的地方，避免定义大量常量
x = tf.placeholder(tf.float32, shape = (3, 2), name = 'input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


with tf.Session() as sess:
    init_var = tf.initialize_all_variables()
    sess.run(init_var)
    #定义字典，提供取值
    print(sess.run(y, feed_dict = {x : [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
    print(y)
