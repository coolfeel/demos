
import tensorflow as tf



n_input = 784
n_output = 10

weights = {
    'wc1' : tf.Variable(tf.random_normal([3, 3, 1, 64], stddev = 0.1)),
    'wc2' : tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.1)),
    'wd1' : tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev = 0.1)),
    'wd2' : tf.Variable(tf.random_normal([1024, n_output], stddev = 0.1))
}

biases = {
    'bc1' : tf.Variable(tf.random_normal([64], stddev = 0.1)),
    'bc2' : tf.Variable(tf.random_normal([128], stddev = 0.1)),
    'bd1' : tf.Variable(tf.random_normal([1024], stddev = 0.1)),
    'bd2' : tf.Variable(tf.random_normal([n_input], stddev = 0.1))
}



def conv_basic(input, w, b, keepratio):
    #input,28 * 28 * 1 ,1通道,第一维是batch_size大小
    input_r = tf.reshape(input, shape = [-1, 28, 28, 1])
    #c1
    conv1 = tf.nn.conv2d(input_r, w['wc1'], strides = [1, 1, 1, 1], padding = 'SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b['bc1']))
    pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    pool_dr1 = tf.nn.dropout(pool1, keepratio)

    #c2
    conv2 = tf.nn.conv2d(pool_dr1, w['wc2'], strides = [1, 1, 1, 1], padding = 'SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b['bc2']))
    pool2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    pool_dr2 = tf.nn.dropout(pool2, keepratio)

    #vectorize, [-1, 7 * 7 * 128] = [-1, 6272]
    dense1 = tf.reshape(pool_dr2, [-1, w['wd1'].get_shape().as_list()[0]])
    #fc1
    fc1 = tf.nn.relu(tf.add(tf.matmul(dense1, w['wd1']), b['bd1']))
    fc_dr1 = tf.nn.dropout(fc1, keepratio)
    #fc2
    out = tf.add(tf.matmul(fc_dr1, w['wd2']), b['bd2'])


