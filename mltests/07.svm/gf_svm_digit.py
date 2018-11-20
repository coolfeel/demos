

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import numpy as np
from scipy import stats   #统计有关




digits = datasets.load_digits()

np.set_printoptions(threshold = np.inf)

images_and_labels = list(zip(digits.images, digits.target))

# for index, (image, label) in enumerate(images_and_labels[: 4]):
#     plt.subplot(1, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1)) #1764 * 8 * 8 转为 1764 * 64

classifier = svm.SVC(gamma = 0.001)
classifier.fit(data[: n_samples // 2], digits.target[: n_samples // 2])

expected = digits.target[n_samples // 2 :]
pred = classifier.predict(data[n_samples // 2 :])

#混淆矩阵
print(metrics.confusion_matrix(expected, pred))

#报告
print(classifier, metrics.classification_report(expected, pred))

images_and_pred_labels = list(zip(digits.images[n_samples // 2 :], pred))

print(images_and_pred_labels)

for index, (image, prediction) in enumerate(images_and_pred_labels[: 4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap = plt.cm.gray_r, interpolation = 'nearest')
    plt.title(prediction)

plt.show()