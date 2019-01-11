
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf





TRAIN_DIR = '../datasets/dog_vs_cat/train'
TEST_DIR = '../datasets/dog_vs_cat/test'

IMAGE_SIZE = 150
CHANNEL = 3
pixel_depth = 255.0

TRAINING_AND_VALIDATION_SIZE_DOGS = 1000
TRAINING_AND_VALIDATION_SIZE_CATS = 1000
TRAINING_AND_VALIDATION_SIZE_ALL = 2000
TRAINING_SIZE = 1600
VALID_SIZE = 400
TEST_SIZE_ALL = 500

image_size = IMAGE_SIZE
num_labels = 2
num_channels = 3




#提取训练集文件
train_images = [i for i in os.listdir(TRAIN_DIR)]
#提取出训练集中的狗图片文件
train_dogs = [i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
#提取处训练集中的猫图片文件
train_cats = [i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
#提取测试集文件
test_images = [i for i in os.listdir(TRAIN_DIR)]

#制作训练集
train_images = train_dogs[: TRAINING_AND_VALIDATION_SIZE_DOGS] + train_cats[: TRAINING_AND_VALIDATION_SIZE_CATS]
train_labels = np.array((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + (['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS))

#制作测试集
test_images = test_images[: TEST_SIZE_ALL]
test_labels = np.array(['unknowclass'] * TEST_SIZE_ALL)




#读取图片，并改变尺寸
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #将图像的长或宽最大限制在150，并按照原图的比例，设置另一方，如果不够150，则补0
    if img.shape[0] >= img.shape[1]:
        resizeto = (IMAGE_SIZE, int(round(IMAGE_SIZE * (float(img.shape[1]) / img.shape[0]))))
    else:
        resizeto = (int(round(IMAGE_SIZE * (float(img.shape[0]) / img.shape[1]))), IMAGE_SIZE)
    #使用双三次插值cv2.INTER_CUBIC，（宽，高）(150, 103, 3)
    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation = cv2.INTER_CUBIC)
    #copyMakeBorder给原图像增加边界，（src,top, bottom, left, right ,borderType,value），(150, 150, 3)
    img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)
    # cv2.imshow('cool', img3)
    # cv2.waitKey(0)
    #(150,150,3)
    return img3[:, :, : : -1]



#预处理数据
def prep_data(images):
    count = len(images)
    #初始化全部为0的(2000, 150, 150, 3)的numpy.ndarray
    data = np.ndarray((count, IMAGE_SIZE,IMAGE_SIZE, CHANNEL), dtype = np.float32)
    #遍历图片list, 0 dog.11823.jpg，1 dog.12430.jpg
    for i, image_file in enumerate(images):
        image = read_image(TRAIN_DIR + '/' + image_file)
        image_data = np.array(image, dtype = np.float32)
        image_data[:, :, 0] = (image_data[:, :, 0].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:, :, 1] = (image_data[:, :, 1].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:, :, 2] = (image_data[:, :, 2].astype(float) - pixel_depth / 2) / pixel_depth
        data[i] = image_data
        if i % 250 == 0:
            print('Processed {} of {}'.format(i, count))
    return data



#数据集打乱顺序
def randomize(dataset, labels):
    #顺序打乱，但不影响原来数组,0-1999乱序
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels



#将labels设置成one-hot编码
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    #将dog设为0， cat设为1
    labels = (labels == 'cats').astype(np.float32)
    #[0 1] == [1]  变成  [False, True] 变成 [0, 1] 对应 [狗， 猫]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels



#模型
def model(data):
    parameters = []
    # conv1
    with tf.name_scope('conv1_1') as scope:
        conv = tf.nn.conv2d(data, kernel_conv1, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases_conv1)
        # 先conv，再加bias后经过relu再池化
        conv1_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel_conv1, biases_conv1]
    # pool1
    pool1 = tf.nn.max_pool(
        conv1_1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool1'
    )
    # conv2
    with tf.name_scope('conv2_1') as scope:
        conv = tf.nn.conv2d(pool1, kernel_conv2, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases_conv2)
        conv2_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel_conv2, biases_conv2]
    # pool2
    pool2 = tf.nn.max_pool(
        conv2_1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool2'
    )
    with tf.name_scope('conv3_1') as scope:
        conv = tf.nn.conv2d(pool2, kernel_conv3, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases_conv3)
        conv3_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel_conv3, biases_conv3]
    # pool3
    pool3 = tf.nn.max_pool(
        conv3_1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool3'
    )
    # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool3.get_shape()[1:]))
        pool3_flat = tf.reshape(pool3, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)
        parameters += [fc1w, fc1b]
    # fc3
    with tf.name_scope('fc3') as scope:
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        parameters += [fc2w, fc2b]
    return fc2l



#准确率
def accuracy(prediction, labels):
    return (100.0 * np.sum((np.argmax(prediction, axis = 1) == np.argmax(labels, axis = 1)).astype(np.float32)) / prediction.shape[0])




if __name__ == '__main__':
    #read_image(TRAIN_DIR + '/' + 'cat.25.jpg')
    #(2000, 150, 150, 3)
    train_normalized = prep_data(train_images)
    #(500, 150, 150, 3)
    test_normalized = prep_data(test_images)

    np.random.seed(133)
    #打乱训练集
    train_dataset_rand, train_labels_rand = randomize(train_normalized, train_labels)
    #打算测试集 (500, 150, 150, 3) (500,)
    test_dataset, test_labels = randomize(test_normalized, test_labels)
    #划分验证集 (400, 150, 150, 3) (400,)
    valid_dataset = train_dataset_rand[: VALID_SIZE, :, :, :]
    valid_labels = train_labels_rand[: VALID_SIZE]
    #剩下的是拿来训练的 (1600, 150, 150, 3) (1600,)
    train_dataset = train_dataset_rand[VALID_SIZE :, :, :, :]
    train_labels = train_labels_rand[VALID_SIZE :]

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)


    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64

    graph = tf.Graph()
    with graph.as_default():
        #输入数据
        tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        #变量
        kernel_conv1 = tf.Variable(
            tf.truncated_normal([3, 3, 3, 32], dtype = tf.float32, stddev = 1e-1),
            name = 'weitghts_conv1'
        )
        biases_conv1 = tf.Variable(
            tf.constant(0.0, shape = [32], dtype = tf.float32),
            trainable = True,
            name = 'biases_conv1'
        )
        kernel_conv2 = tf.Variable(
            tf.truncated_normal([3, 3, 32, 32], dtype = tf.float32, stddev = 1e-1),
            name = 'weights_conv2'
        )
        biases_conv2 = tf.Variable(
            tf.constant(0.0, shape = [32], dtype = tf.float32),
            trainable = True,
            name = 'biases_conv2'
        )
        kernel_conv3 = tf.Variable(
            tf.truncated_normal([3, 3, 32, 64], dtype = tf.float32, stddev = 1e-1),
            name = 'weights_conv3'
        )
        biases_conv3 = tf.Variable(
            tf.constant(0.0, shape = [64], dtype = tf.float32),
            trainable = True,
            name = 'biases_conv3'
        )
        fc1w = tf.Variable(
            tf.truncated_normal([23104, 64], dtype = tf.float32, stddev = 1e-1),
            name = 'weights'
        )
        fc1b = tf.Variable(
            tf.constant(1.0, shape = [64], dtype = tf.float32),
            trainable = True,
            name = 'biases'
        )
        fc2w = tf.Variable(
            tf.truncated_normal([64, 2], dtype = tf.float32, stddev = 1e-1),
            name = 'weights'
        )
        fc2b = tf.Variable(
            tf.constant(1.0, shape = [2], dtype = tf.float32),
            trainable = True,
            name = 'biases'
        )


        #训练,预测值
        logits = model(tf_train_dataset)
        #损失,预测值与真实值的交叉熵
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf_train_labels))
        #优化
        optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(loss)
        #pred
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

        #迭代次数
        num_steps = 1001
        with tf.Session(graph = graph) as session:
            tf.initialize_all_variables().run()
            print('Init')
            for step in range(num_steps):
                #偏移的索引
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                #在偏移的基础上，进行批处理
                batch_data = train_dataset[offset : (offset + batch_size), :, :, :]
                batch_labels = train_labels[offset : (offset + batch_size), :]
                #填充批处理的数据
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                _, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
                if step % 50 == 0:
                    print('minibatch loss at step', step, ':', l)
                    print('minibatch accu : %.1f%%' % accuracy(prediction, batch_labels))
                    print('validation accu : %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            #print('test : %.1f%%' % accuracy(test_prediction.eval(), test_labels))