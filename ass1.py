# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
print(cv2.__version__)
from PIL import Image
Img = Image.open('F:/projects/Computer Vision/tinytan.jpg')
import matplotlib.pyplot as plt
plt.imshow(Img)


import numpy as np
print(np.shape(Img))

img1=np.asarray(Img)
plt.imshow(img1[:,:,1],cmap='gray')
plt.imshow(img1[:,:,0],cmap='Purples_r')
img1[1:10,1:10,1]

#see each channel separately
r,g,b=Img.split()
img1=np.asarray(r)
plt.imshow(Img)
plt.imshow(img1)

from skimage.color import rgb2hsv
hsvimg=rgb2hsv(Img)
print(img1[1:10,1:10])
print(hsvimg[1:10,1:10,0])
print(hsvimg[1:10,1:10,1])
print(hsvimg[1:10,1:10,2])
plt.imshow(hsvimg)

def histogram(im):
    h = np.zeros(255)
    for row in im.shape[0]:
        for col in im.shape[1]:
            val = im[row,col]
            h[val] +=1
#Homework
#view color channels
color=['b','g','r']

#separate the colors and plot the histogram
for i,col in enumerate(color):
    hist = cv2.calcHist([Img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.show()
            
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
Img = cv2.imread('F:/projects/Computer Vision/tinytan.jpg')
plt.imshow(Img)
histogram_image = cv2.calcHist([Img],[0],None,[256],[0,256])
hist,bins=np.histogram(Img.ravel(),256,[0,256])
np.shape(hist)
hist[1:10]
#flatten the histogram
plt.hist(Img.ravel(),256,[0,256])
plt.show()

np.mean(Img)

from scipy import stats
stats.mode(Img)   #mode calculated over column

np.median(Img)
np.std(Img)
np.amin(Img,1)
np.amax(Img,1)

np.unique(Img, return_counts=True)


#2
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
Img = Image.open('F:/projects/Computer Vision/tinytan.jpg')
plt.imshow(Img)
print(np.shape(Img))
img1=np.asarray(Img)  #conv to numpy array
#RGB TO GRAY
import cv2
grayscale=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1=np.asarray(grayscale)
print(img1.shape)
plt.imshow(img1,cmap='gray')

#histogram equilization
equ=cv2.equalizeHist(img1)
img2=np.hstack((img1,equ))  #plotting both images
plt.imshow(img2,cmap='gray')
img1[1:5,1:5]
equ[1:5,1:5]  #statistical properties

#cumulative distributive function
hist,bin=np.histogram(img1.flatten(),256,[0,256])
cfd=hist.cumsum()
cfd_normalized = cfd*hist.max()/cfd.max()
plt.plot(cfd_normalized,color='b')

hist1,bin1=np.histogram(equ.flatten(),256,[0,256])
cfd1=hist1.cumsum()
cfd_normalized1 = cfd1*hist1.max()/cfd1.max()
plt.plot(cfd_normalized1,color='b')

plt.hist(img1.flatten(),256,[0,256],color='r')
plt.hist(equ.flatten(),256,[0,256],color='b')

#guassian filter  blurry effect , change the sigma values
kernel = np.ones((5,5),np.float32)/25
kernel
filtered =cv2.GaussianBlur(img1,(5,5),0)
img2=np.hstack((img1,filtered))
plt.imshow(img2,cmap='gray')

#fourier transform 2d of image   global effect
fftofimg=np.fft.fft2(img1)
img1[0:4,0:4]
fftofimg[0:4,0:4]
fftofimg[0:4,0:4]=0 
ifftofimg=np.fft.ifft2(fftofimg)
img2=np.hstack((img1,ifftofimg))
plt.imshow(np.real(img2),cmap='gray')

fshift =np.fft.fftshift(fftofimg)
magnitude_spectrum=20*np.log(np.abs(fshift))
plt.imshow(magnitude_spectrum,cmap='gray')
