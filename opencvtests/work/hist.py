# *-* encoding:utf-8 *-*

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('../imgs/dog4.jpeg', 0)

np.set_printoptions(threshold = np.inf)
print(img)

ret, fp = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('fp', np.hstack([img, fp]))

equ = cv2.equalizeHist(img)

gray_lap = cv2.Laplacian(img, -1, ksize = 3)

plt.figure()
plt.subplot(2, 1, 1)
ori = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(ori)

plt.subplot(2, 1, 2)
cur = cv2.calcHist([equ], [0], None, [256], [0, 256])
plt.plot(cur)

plt.show()

cv2.imshow('cool', np.hstack([img, equ, gray_lap]))

cv2.waitKey()