
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img





datasets_path = 'yangben/'
tfrecords_path = 'tfrecords/'
#所有种类,按顺序
#'bajiaojinpan_', 'tao_', 'changchunteng_', 'wucaisu_', 'luhui_',
# 'luwei_', 'dongqing_', 'yueji_', 'yinxing_', 'gangzhu_', 'yulan_',
# 'chuiliu_', 'zonglv_', 'jizhuaqi_', 'guangyulan_', 'shuishan_
classes = os.listdir('yangben')
#分类数目
classes_num = len(classes)


#样本扩充
def data_aug():
    print(classes)
    datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale = 1.0 / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest',
    )
    #每张图片经过变换扩充为10个
    for yangben_classes in classes:
        for pic in os.listdir(datasets_path + '/' + str(yangben_classes)):
            img = load_img(datasets_path + str(yangben_classes) + '/' + pic)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for _ in datagen.flow(x, batch_size = 1, save_format = 'jpg', save_to_dir = datasets_path + str(yangben_classes)):
                i += 1
                if i > 10:
                    break

#划分数据集
def split_datasets():
    train_sets = []
    test_sets = []
    #将图片分成训练集和测试集
    for index, label in enumerate(classes):
        print(index, label)
        imgs = os.listdir('yangben/' + str(label))
        #打乱顺序
        np.random.shuffle(imgs)
        #该类总图片数
        imgs_nums = len(imgs)
        ratio = 0.8
        train_nums = int(imgs_nums * ratio)
        train = imgs[: train_nums]
        test = imgs[train_nums :]
        train_sets.append(train)
        test_sets.append(test)
    return train_sets, test_sets


def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


#数据集转为tfrecord格式文件
def datasets_to_tfrecords(datasets, datasets_name):
    writer = tf.python_io.TFRecordWriter(tfrecords_path + datasets_name)
    for index, sets in enumerate(datasets):
        classes_path = datasets_path + classes[index]
        for img_name in sets:
            img_path = classes_path + '/' + img_name
            print(index, img_path)
            img = Image.open(img_path, 'r')
            size = img.size
            #将图片转为二进制
            img_raw = img.tobytes()
            example = tf.train.Example(features = tf.train.Features(feature = {
                "label": _int64_feature(index),
                "img_raw": _bytes_feature(img_raw),
                'img_width': tf.train.Feature(int64_list = tf.train.Int64List(value = [size[0]])),
                'img_height': tf.train.Feature(int64_list = tf.train.Int64List(value = [size[1]]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


#读取tfrecords文件
def read_tfrecords(filename, batch_size, shuffle):
    # 获取文件名列表
    data_files = tf.gfile.Glob(tfrecords_path + filename)
    print(data_files)
    # 文件名列表生成器
    filename_queue = tf.train.string_input_producer(data_files)
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 取出包含image和label的feature对象
    features = tf.parse_single_example(serialized_example,
                                       features = {
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64)
                                       })
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    height = tf.cast(features['img_height'], tf.int32)
    channel = 3
    #裁剪和归一化
    image = tf.reshape(image, [height, width, channel])
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    image = tf.cast(image, tf.float32) * (1. / 255)

    # 组合batch
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        #随机抽取
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size = batch_size,
            num_threads = 64,
            capacity = capacity,
            min_after_dequeue = min_after_dequeue)
    else:
        #按顺序抽取
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size = batch_size,
            num_threads = 64,
            capacity = capacity)
    # 转成one-hot编码形式
    label_batch = tf.reshape(label_batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    label_batch = tf.sparse_to_dense(
        tf.concat(values = [indices, label_batch], axis = 1),
        [batch_size, classes_num], 1.0, 0.0)
    return image_batch, label_batch



#测试
if __name__ == '__main__':
    #扩充
    #data_aug()
    #划分
    #train_sets, test_sets = split_datasets()
    #转成tfrecords
    #datasets_to_tfrecords(train_sets, 'train_tfrecords')
    #datasets_to_tfrecords(test_sets, 'test_tfrecords')
    #测试
    # read_tfrecords('test_tfrecords', 10, True)
    pass