
import numpy as np
import cv2



img = np.zeros((512, 512, 3), np.uint8)

# np.set_printoptions(threshold = np.inf)

cv2.line(img, (0, 0), (260, 260), (255, 0, 0), 5)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img2 = np.zeros((512, 512, 3), np.uint8)

cv2.rectangle(img2, (350, 0), (500, 128), (0, 255, 0), 3)
cv2.circle(img2, (425, 63), 63, (0, 0, 255), -1)
cv2.line(img2, (0, 0), (128, 128), (255, 0, 0), 10)
cv2.ellipse(img2, (256, 256), (100, 50), 0, 0, 100, 255, -1)

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
print(pts)
pts = pts.reshape((-1, 1, 2))
print(pts)
cv2.polylines(img2, [pts], True, (0, 255, 255))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img2, 'cooooooool', (10, 500), font, 2, (255, 255, 255), 5, cv2.LINE_AA)


cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()