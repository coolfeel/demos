
import gluonbook as gb
from mxnet import autograd, nd


batch_size = 256
train_iter, test_iter  = gb.load_data_fashion_mnist(batch_size)

