
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

#转成one-hot使用
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau





np.random.seed(2)
sns.set(style = 'white', context = 'notebook', palette = 'deep')
#42000 * 785
train = pd.read_csv('../datasets/digit_recognize/train.csv')
#28000 * 784
test = pd.read_csv('../datasets/digit_recognize/test.csv')

Y_train = train['label']
#在列上删去为label的列
X_train = train.drop(labels = ['label'], axis = 1)

#画处计数的柱状图
# g = sns.countplot(Y_train)
# plt.show()
#计算每种数的个数
#print(Y_train.value_counts())


#print(X_train.isnull().any().describe())
#print(test.isnull().any().describe())

#归一化
X_train = X_train / 255.0
test = test / 255.0

#不省略的输出,设置阀值
#np.set_printoptions(threshold = np.inf)


#reshape,将784变成28 * 28 * 1,取values时变成ndarray
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

#变成one-hot编码
Y_train = to_categorical(Y_train, num_classes = 10)


random_seed = 2
#划分成训练集和验证集,(37800, 28, 28, 1) (4200, 28, 28, 1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = random_seed)

#(28, 28, 1)
#print(X_train[0].shape)
#print(X_train[0])
#(28, 28)
# print(X_train[0][:, :, -1].shape)
# print(X_train[0][:, :])

# g = plt.imshow(X_train[0][:, :, 0])
# plt.show()


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

#rho梯度滑动平均衰减因子,decay学习率衰减
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])


#学习率衰减,factor每次减少学习率的因子lr = lr * factor,
#patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发

learing_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)


epochs = 1
batch_size = 86


datagen = ImageDataGenerator(
    #设置输入的均值为0
    featurewise_center = False,
    #设置每个样本的均值为0
    samplewise_center = False,
    #输入除以数据集的标准差进行标准化
    featurewise_std_normalization = False,
    #每个样本除以它的标准差进行标准化
    samplewise_std_normalization = False,
    #白化
    zca_whitening = False,
    rotation_range = 10,
    #图像放大
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    #对图像进行翻转
    horizontal_flip = False,
    vertical_flip = False
)


datagen.fit(X_train)


history = model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size = batch_size),
    epochs = epochs,
    validation_data = (X_val, Y_val),
    verbose = 2,
    steps_per_epoch = X_train.shape[0] // batch_size,
    callbacks = [learing_rate_reduction]
)














