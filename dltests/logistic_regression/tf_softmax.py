
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../datasets/mnists', one_hot = True)

x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])

w = tf.Variable(tf.zeros(shape = [784, 10]))
b = tf.Variable(tf.zeros([10]))

actv = tf.nn.softmax(tf.matmul(x, w) + b)
#所有样本的平均loss
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), axis = 1))

learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))

accr = tf.reduce_mean(tf.cast(pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_epochs = 50
    batch_size = 100
    display_step = 5
    for epoch in range(training_epochs):
        avg_cost = 0
        num_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #执行优化，梯度下降
            sess.run(optm, feed_dict = {x : batch_xs, y : batch_ys})
            feeds = {x : batch_xs, y : batch_ys}
            #这一小批的平均损失
            avg_cost += sess.run(cost, feed_dict = feeds) / num_batch

            if epoch % display_step == 0:
                feeds_train = {x : batch_xs, y : batch_ys}
                feeds_test = {x : mnist.test.images, y : mnist.test.labels}
                train_acc = sess.run(accr, feed_dict = feeds_train)
                test_acc = sess.run(accr, feed_dict = feeds_test)
                print(epoch, training_epochs, avg_cost, train_acc, test_acc)
