
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



n_input = 784
n_classes = 10
n_hidden1 = 256
n_hidden2 = 128

#input,output
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


#network parameters
stddev = 0.1
weights = {
    'w1' : tf.Variable(tf.random_normal([n_input, n_hidden1], stddev = stddev)),
    'w2' : tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev = stddev)),
    'out' : tf.Variable(tf.random_normal([n_hidden2, n_classes], stddev = stddev))
}

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden2])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
}


def mul_perceptron(_X, _weights, _biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2']))
    return (tf.matmul(layer2, _weights['out']), _biases['out'])


pred = mul_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optm = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))
