
import cv2



img = cv2.imread('../imgs/dog4.jpeg')

gray_lap = cv2.Laplacian(img, -1, ksize = 3)

cv2.imshow('cool', gray_lap)

cv2.waitKey()