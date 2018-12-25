import tensorflow as tf
import numpy as np


#卷积
def conv(layer_name, x, out_channels, kernel_size = [3, 3], stride = [1, 1, 1, 1], is_pretrain = True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(
            name = 'weights',
            trainable = is_pretrain,
            shape = [kernel_size[0], kernel_size[1], in_channels, out_channels],
            initializer = tf.truncated_normal_initializer(stddev = 0.01, dtype = tf.float32)
        )
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(w))
        b = tf.get_variable(
            name = 'biases',
            trainable = is_pretrain,
            shape = [out_channels],
            initializer = tf.constant_initializer(0.0)
        )
        x = tf.nn.conv2d(x, w, stride, padding = 'SAME', name = 'conv')
        x = tf.nn.bias_add(x, b, name = 'bias_add')
        x = tf.nn.relu(x, name ='relu')
        return x


#池化
def pool(layer_name, x, kernel = [1, 2, 2, 1], stride = [1, 2, 2, 1], is_max_pool = True):

    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides = stride, padding = 'SAME', name = layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides = stride, padding = 'SAME', name = layer_name)
    return x


#全连接
def FC_layer(layer_name, x, out_nodes):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable(
            'weights',
            shape = [size, out_nodes],
            initializer = tf.truncated_normal_initializer(stddev = 0.01, dtype = tf.float32)
        )
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0001)(w))
        b = tf.get_variable(
            'biases',
            shape = [out_nodes],
            initializer = tf.constant_initializer(0.0)
        )
        #变成1维
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x



def loss(logits, labels):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = 'cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.add_to_collection('losses', loss)
        losses = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar(scope + '/loss', loss)
        return losses



def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        #Tensor("accuracy/ArgMax:0", shape=(8,), dtype=int64)
        correct = tf.equal(tf.argmax(logits, axis = 1), tf.argmax(labels, axis = 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
    return accuracy



def num_correct_prediction(logits, labels):
    correct = tf.equal(tf.argmax(logits, axis = 1), tf.argmax(labels, axis = 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct



def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        return train_op










