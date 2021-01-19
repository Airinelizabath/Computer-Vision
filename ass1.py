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
            
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
Img = cv2.imread('F:/projects/Computer Vision/tinytan.jpg')
histogram_image = cv2.calcHist([Img],[0],None,[256],[0,256])
hist,bins=np.histogram(Img.ravel(),256,[0,256])
np.shape(hist)
hist[1:10]
#flatten the histogram
plt.hist(Img.ravel(),256,[0,256])
plt.show()
#view color channels
color=['b','g','r']
#separate the colors and plot the histogram
for i,col in enumerate(color):
    hist = cv2.calcHist([Img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.show()

np.mean(img1)

from scipy import stats
stats.mode(img1)   #mode calculated over column

np.median(img1)
np.std(img1)
np.amin(img1,1)
np.amax(img1,1)

np.unique(img1, return_counts=True)
