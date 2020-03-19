# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:49:45 2019

@author: karan
"""

#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


#img1 = cv2.imread('face_good.bmp', cv2.IMREAD_COLOR)
img1 = cv2.imread(r'faces\image_0001.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread(r'faces\image_0001.jpg', 0)
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

#H = [0, 50], S = [0.20, 0.68] and V = [0.35, 1.0] 
#cv2.imshow('gu',img1)
#cv2.waitKey(1)
plt.imshow(hsv)

plt.show()

lower = np.array([0, 70, 150], dtype = "uint8")
upper = np.array([20, 190, 255], dtype = "uint8")
mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(hsv,hsv, mask= mask)
plt.imshow(mask)
plt.show()

kernel1 = np.ones((50, 50), np.uint8) 
cmask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel1) 
#res = cv2.bitwise_and(hsv,hsv, mask= opening)

plt.imshow(cmask)
plt.show()
res = cv2.bitwise_and(hsv,hsv, mask= cmask)
final = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
plt.imshow(final)

plt.show()
'''
plt.imshow(hsv)
plt.show()
chans = cv2.split(img1)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
 

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

plt.hist(img.ravel(),256,[0,256]); plt.show()

lower = np.array([150, 10, 90], dtype = "uint8")
upper = np.array([255, 190, 200], dtype = "uint8")
mask = cv2.inRange(hsv, lower, upper)

plt.imshow(mask)
plt.show()

kernel = np.ones((40,40), np.uint8) 
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
kernel1 = np.ones((40,40), np.uint8) 
cmask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel1)

#res = cv2.bitwise_and(hsv,hsv, mask= opening)
res = cv2.bitwise_and(hsv,hsv, mask= cmask)
final = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
plt.imshow(final)
plt.show()'''
cv2.imwrite('mask.jpg',cmask)
new = cv2.imread('mask.jpg', 0)
arr=[]
arr1=[]
for i in range(img2.shape[0]):
    arr.append(sum(mask[i][:img2.shape[1]])) 
column=arr.index(max(arr))
for j in range(img2.shape[1]):
    arr1.append(sum(mask[:img2.shape[0],j])) 
row=arr1.index(max(arr1))
newimg = final[column-120:column+170,row-200:row+160]
plt.imshow(newimg)
plt.show()    
cv2.destroyAllWindows()        

