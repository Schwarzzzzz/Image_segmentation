import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('input.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(type(img_gray))
print(img_gray.shape)
print(img_gray)
# hist是256x1数组，每个值对应于该图像中具有相应像素值的像素数
hist = cv.calcHist([img_gray],[0],None,[256],[0,256])
# 绘制直方图
plt.plot(hist)
plt.show()

retVal, a_img = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
print("retVal：" + str(retVal))
cv.imwrite('output.jpg', a_img)
cv.imshow("a_img",a_img)
cv.waitKey()
