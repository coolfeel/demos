
from PIL import Image


#读取
pil_im = Image.open('../../imgs/empire.jpg')
pil_im.show()


#读取并转成灰度图
pil_im2 = pil_im.convert('L')
pil_im2.show()


#将原图变成略缩图,创建最长变为128像素的略缩图
# pil_im.thumbnail((120, 120))
# print(pil_im)
# pil_im.show()


#裁剪出指定区域
#box = (100, 100, 400, 400)
#region = pil_im.crop(box)
#旋转180°
#region = region.transpose(Image.ROTATE_180)
#在指定位置粘贴图片
#pil_im.paste(region, box)
#pil_im.show()
#region.show()


#调整尺寸
out = pil_im.resize((128, 128))
#旋转45°
out = out.rotate(45 )
out.show()



