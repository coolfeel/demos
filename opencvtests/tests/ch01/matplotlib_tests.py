
from PIL import Image
from pylab import *
import numpy as np




im = Image.open('../../imgs/empire.jpg')



img = np.array(im)

imshow(img)
print('1')
x = ginput(3)

print(x)

show()





#转灰度图像数组
# im = np.array(im.convert('L'))
# print(im.shape)


#创建1个图像
# figure()

#不用颜色信息
# gray()

#在远点的左上角显示轮廓图像
# contour(im, origin = 'image')
#
# figure()

#数组压平成1维，128个直方图的小区间
# hist(im.flatten(), 128)
#
# show()








#绘制像点线
# imshow(im)

# x = [100, 100, 400, 400]
# y = [200, 500, 200, 500]

#pylab库命令颜色，线型格式，标记格式
# plot(x, y, 'r:*')
#plot(x[: 2], y[: 2])
# show()







#图像转矩阵
#im_mat = np.array(im)
#R, G, B
# t1 = im_mat[:, :, 0]
# t2 = im_mat[:, :, 1]
# t3 = im_mat[:, :, 2]

#R, G, B
#T1, T2, T3 = im.split()

#PIL.Image
#print(type(T2))

#单纯的数组
#print(type(t2))



#数组变成PIL.Image
# zeros = Image.fromarray(np.zeros((t1.shape), dtype = 'uint8'))
# print(zeros)

#数组转PIL.Image
# t11 = Image.fromarray(t1)
# t22 = Image.fromarray(t2)
# t33 = Image.fromarray(t3)
#
# print(type(t11), t22)
# t11.show()
# t22.show()
# t33.show()

#merge接收PIL.Image类型,不接收数组类型
# tr = Image.merge('RGB', (zeros, T2, zeros))
# tr.show()