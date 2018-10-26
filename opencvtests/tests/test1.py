import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt




img = cv2.imread('../imgs/dog3.png')

# roi = img[0 : 100, 0 : 100]
# img[300 : 400, 300 : 400] = roi
#
# cv2.imshow('cool', img)



kernel_3X3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

res = cv2.filter2D(img, -1, kernel_3X3)

plt.imshow(res)
plt.imsave('res2.png', res)

cv2.waitKey()