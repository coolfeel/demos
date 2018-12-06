

import gluonbook as gb
from mxnet.gluon import data as gdata
import time
import ipykernel




minst_train = gdata.vision.FashionMNIST(train = True)
minst_test = gdata.vision.FashionMNIST(train = False)


#feature, label = minst_train[1]

def get_fashion_minst_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
                   'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_minst(images, labels):
    gb.use_svg_display()
    figs =gb.plt.subplots(1, len(images), figsize = (12, 12))
    print(figs)
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape())









if __name__ == '__main__':
    X, y = minst_train[0 : 9]
    show_fashion_minst(X, get_fashion_minst_labels(y))