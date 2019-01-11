
import os
import math
import numpy as np
import tensorflow as tf
import preprocessing_datasets
import VGG
import tools
import matplotlib.pyplot as plt




#图片宽
IMG_W = 224
#图片高
IMG_H = 224
#种类数
N_CLASSES = 16
#一批量的图片数量
BATCH_SIZE = 10
#测试时一批量的图片数量
TEST_BATCH_SIZE = 20
#学习率
learning_rate = 0.0001
#最大迭代次数
MAX_STEP = 6000
#是否是trainable
IS_PRETRAIN = True


#训练
def train():
    train_data_dir = 'train_tfrecords'
    #训练生成文件路径
    train_log_dir = 'logs/train/'

    with tf.name_scope('input'):
        #读取train_tfrecords
        tra_image_batch, tra_label_batch = preprocessing_datasets.read_tfrecords(
            filename = train_data_dir,
            batch_size = BATCH_SIZE,
            shuffle = True
        )
        print(tra_image_batch, tra_label_batch)

    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape = [BATCH_SIZE, N_CLASSES])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            # print(tra_images.shape, tra_labels.shape)
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], feed_dict = {x: tra_images, y_: tra_labels})

            print('Step: %d, loss: %.5f, accuracy: %.5f%%' % (step, tra_loss, tra_acc))

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)

            # plt.imshow(tra_images[step, :, :, :])
            # plt.show()
    except tf.errors.OutOfRangeError:
        print('out of range')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()



#测试
def test():
    with tf.Graph().as_default():

        log_dir = 'logs/train/'
        test_data_dir = 'test_tfrecords'
        n_test = 500

        #读取test_tfrecords
        test_image_batch, test_label_batch = preprocessing_datasets.read_tfrecords(
            filename = test_data_dir,
            batch_size = TEST_BATCH_SIZE,
            shuffle = False
        )

        logits = VGG.VGG16N(test_image_batch, N_CLASSES, False)
        correct = tools.num_correct_prediction(logits, test_label_batch)
        saver = tf.train.Saver(tf.global_variables())


        with tf.Session() as sess:
            print(log_dir)
            ckpt = tf.train.get_checkpoint_state(log_dir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('global_step %s' % global_step)
            else:
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            try:
                num_step = int(math.floor(n_test / TEST_BATCH_SIZE))
                num_sample = num_step * TEST_BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    # print(sess.run(logits))
                    # print(sess.run(tf.argmax(logits, axis = 1)), sess.run(tf.argmax(test_label_batch, axis=1)))
                    step += 1
                print('samples_nums:%d' % num_sample)
                print('acc_nums:%d' % total_correct)
                print('acc: %.5f%%' % (100 * total_correct / num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)



if __name__ == '__main__':
    #train()
    test()