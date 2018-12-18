
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf


#学习率
learning_rate = 1e-4
#训练次数
training_iteration = 10000
dropout = 0.5
batch_size = 50
validation_size = 2000
image_to_out = 10

data = pd.read_csv('../datasets/digit_recognize/train.csv')

#针对元组格式化输出
#print('{0[0]}, {0[1]}'.format(data.shape))
#print('{0}, {1}'.format(2, 3))

#iloc通过行号进行读取
images = data.iloc[:, 1 :].values

#像素从int64转为float64
images = images.astype(np.float)

#print(type(images.values[0][0]))

#归一化
images = np.multiply(images, 1.0 / 255.0)

image_size = images.shape[1]

#向上取整作为图像大小,图像宽和高要为uint
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)


# def display(img):
#     one_image = img.reshape(image_height, image_width)
#     plt.imshow(one_image)
#     plt.show()
#
#
# display(images.values[0])

#所有样本的label
labels_flat = data.values[:, 0]
#所有样本的label种数
label_count = np.unique(labels_flat).shape[0]

#convert to one-hot
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    #偏移量
    index_offset = np.arange(num_labels) * num_classes
    #one-hot
    labels_one_hot = np.zeros((num_labels, num_classes))
    #将42000 × 10拉平，每一个头再加偏移
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot




labels = dense_to_one_hot(labels_flat, label_count).astype(np.uint8)

#print(labels.shape)

#验证集
validation_images = images[: validation_size]
validation_labels = labels[: validation_size]

#训练集
train_images = images[validation_size :]
train_labels = labels[validation_size :]


#权重初始化
def weight_variable(shape):
    initial =  tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

#初始化偏差
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#卷积
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

#池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


#输入
x = tf.placeholder(tf.float32, shape = [None, image_size])
y_ = tf.placeholder(tf.float32, shape = [None, label_count])

#conv1,卷积核5 * 5, 灰度图,单通道，32个feature map
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#tensor reshape   (40000, 784) -> (40000, 28, 28, 1)
image = tf.reshape(x, [-1, image_width, image_height, 1])
#得到40000 * 28 * 28 * 32
h_conv1 = tf.nn.relu(conv2d(image, w_conv1) + b_conv1)
#得到40000 * 14 * 14 * 32
h_pool1 = max_pool_2x2(h_conv1)


#conv2
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#得到40000 * 14 * 14 * 64
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
#得到40000 * 7 * 7 * 64
h_pool2 = max_pool_2x2(h_conv2)


#fc1
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#先flat,40000 * 7 * 7 * 64 -> 40000 * 3136
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
#得到40000 × 1024
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#dropout, 40000 * 1024, 只不过是置0
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax
w_fc2 = weight_variable([1024, label_count])
b_fc2 = bias_variable([label_count])

#40000 * 10
y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)


#loss
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#优化
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predict = tf.argmax(y, axis = 1)

#stochastic training
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]


def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        #finish
        epochs_completed += 1
        #shuffle data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        #再随机取数据
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        #start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples

    end = index_in_epoch
    return train_images[start : end], train_labels[start : end]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # train_accu = []
    # validation_accu = []
    # x_range = []

    display_step = 1
    for i in range(training_iteration):
        #get new batch
        batch_xs, batch_ys = next_batch(batch_size)
        if i % display_step == 0 or (i + 1) == training_iteration:
            train_accu = accuracy.eval(feed_dict = {x : batch_xs, y_ : batch_ys, keep_prob : 1.0})
            if validation_size:
                validation_accu = accuracy.eval(feed_dict = {
                    x : validation_images[0 : batch_size],
                    y_ : validation_labels[0 : batch_size],
                    keep_prob : 1.0
                })
                print(train_accu, validation_accu, i)
                #validation_accu.append(validation_accu)
            else:
                print(train_accu, i)
            #train_accu.append(train_accu)
            #x_range.append(i)
            if i % (display_step * 10) == 0 and i:
                display_step *= 10
        sess.run(train_step, feed_dict = {x : batch_xs, y_ : batch_ys, keep_prob : dropout})


    test_images = pd.read_csv('../datasets/digit_recognize/test.csv').values
    test_images = test_images.astype(np.float)

    test_images = np.multiply(test_images, 1.0 / 255.0)
    #print(test_images.shape)

    predicted_labels = np.zeros(test_images.shape[0])
    for i in range(0, test_images.shape[0] // batch_size):
        predicted_labels[i * batch_size : (i + 1) * batch_size] = predict.eval(
            feed_dict = {x : test_images[i * batch_size : (i + 1) * batch_size],
                         keep_prob : 1.0}
        )

    np.savetxt('result.csv',
               np.c_[range(1, len(test_images) + 1), predicted_labels],
               header = 'ImageId,Label',
               delimiter = ',',
               comments = '',
               fmt = '%d'
    )



















