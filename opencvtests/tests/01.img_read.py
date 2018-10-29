
import cv2
import matplotlib.pyplot as plt



img = cv2.imread('../imgs/dog4.jpeg', 0)

cv2.imshow('img', img)
k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('cool.jpeg', img)
    cv2.destroyAllWindows()


img2 = cv2.imread('../imgs/dog1.jpeg', 0)
plt.imshow(img2, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()
