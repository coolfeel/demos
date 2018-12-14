

import gluonbook as gb
from mxnet.gluon import data as gdata
import time
import ipykernel
import matplotlib.pyplot as plt
import sys





def get_fashion_minst_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
                   'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_minst(images, labels):
    gb.use_svg_display()
    #返回axes对象组成的数组figs
    _, figs = gb.plt.subplots(1, len(images), figsize = (12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    #plt.show()








if __name__ == '__main__':
    #每一个数据是一个元组，第一个元素是28 * 28 * 1的图像矩阵，第2个是标签
    minst_train = gdata.vision.FashionMNIST(train=True)
    minst_test = gdata.vision.FashionMNIST(train=False)

    X, y = minst_train[0 : 9]
    show_fashion_minst(X, get_fashion_minst_labels(y))

    #小批量
    batch_size = 256
    transformer = gdata.vision.transforms.ToTensor()
    #Gluon中的DataLoader设置4个进程读取数据
    num_workers = 4
    #通过ToTensor类将图像数据从 uint8 格式变换成 32 位浮点数格式，
    # 并除以 255 使得所有像素的数值均在 0 到 1 之间。
    # ToTensor类还将图像通道从最后一维移到最前一维来方便之后介绍的卷积神经网络计算。
    # 通过数据集的transform_first函数，将ToTensor的变换应用在每个数据样本（图像和标签）的第一个元素，即图像之上
    train_iter = gdata.DataLoader(minst_train.transform_first(transformer), batch_size, shuffle = True, num_workers = num_workers)
    test_iter = gdata.DataLoader(minst_test.transform_first(transformer), batch_size, shuffle = False, num_workers = num_workers)

