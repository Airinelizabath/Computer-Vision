#lect3
#RGBTOGRAY
import cv2
from PIL import Image
Img = Image.open('F:/projects/Computer Vision/tinytan.jpg')
import matplotlib.pyplot as plt
plt.imshow(Img)
import numpy as np
print(np.shape(Img))
img1=np.asarray(Img)

grayscale=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1=np.asarray(grayscale)
print(img1.shape)
plt.imshow(img1,cmap='gray')

ret,thresh1=cv2.threshold(img1,127,255,cv2.THRESH_BINARY) #threshold value=127,  THRESH_BINARY-simple manual thresholding
img2=np.hstack((img1,thresh1))

hist,bins=np.histogram(img1.flatten(),256,[0,256])
plt.plot(hist,color='b')

th2=cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) #try diff threshold
th3=cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
img2=np.hstack((img1,th2,th3))
plt.imshow(img2,cmap='gray')
plt.imshow(th2,cmap='gray')
plt.imshow(th3,cmap='gray')

ret2,th4=cv2.threshold(img1,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
ret2

blur=cv2.GaussianBlur(img1,(5,5),0)
ret3,th5=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img2=np.hstack((img1,th4,th5))
plt.imshow(img2,cmap='gray')
plt.imshow(th3,cmap='gray')
plt.imshow(th4,cmap='gray')
