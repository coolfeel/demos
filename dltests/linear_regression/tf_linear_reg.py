
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf




num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])


x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# plt.scatter(x_data, y_data, c = 'r')
# plt.show()


w = tf.Variable(tf.random_uniform([1], -1, 1), name = 'w')

b = tf.Variable(tf.zeros([1], name = 'b'))

y = w * x_data + b

loss = tf.reduce_mean(tf.square(y_data - y), name = 'loss')

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss, name = 'train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w), loss.eval())
    #执行20次训练
    for step in range(20):
        sess.run(train)
        print(sess.run(w), sess.run(b), sess.run(loss))
