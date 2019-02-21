# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 05:24:18 2018

@author: vignajeeth
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
#
#def threshold(sobel_out,t=50):
#    sobel_out=np.abs(sobel_out)
#    for i in range(h):
#        for j in range(w):
#            if sobel_out[i,j]<t:
#                sobel_out[i,j]=0
#            else:
#                sobel_out[i,j]=1
#    return(sobel_out)

image = cv2.imread('hough.jpg',0)
h,w=image.shape
sobel_out=np.zeros((h,w))
sobel=[[-1,0,1],[-2,0,2],[-1,0,1]]
sobel=np.array(sobel)

for i in range(1,h-1):
    for j in range(1,w-1):
        sobel_out[i,j]+=np.sum(sobel*image[i-1:i+2,j-1:j+2])

for _ in range(2):    
    for i in range(1,h-1):
        for j in range(1,w-1):
            sobel_out[i,j]+=np.sum(sobel.T*image[i-1:i+2,j-1:j+2])

#sobel_out=threshold(sobel_out)
sobel_out=np.abs(sobel_out)
sobel_out[sobel_out<400]=0
plt.imshow(sobel_out)

votmat=np.zeros((h,w))
rad=22
for i in tqdm(range(rad+5,h-rad-5)):
    for j in range(rad+5,w-rad-5):
        if sobel_out[i,j]!=0:
            for k in range(360):
                votmat[i+(rad*np.cos(k*np.pi/180)),j+(rad*np.sin(k*np.pi/180))]+=1
temp=np.array(votmat)
#votmat=np.array(temp)

#
#for i in range(np.count_nonzero(votmat)):
#    a,b=np.unravel_index(np.argmax(votmat, axis=None), votmat.shape)
#    cv2.circle(votmat,(b,a),22,(0,225,0),2)
plt.imshow(votmat,cmap='gray')

hist=np.zeros((np.max(votmat)+1))
for i in range(h):
    for j in range(w):
        hist[votmat[i,j]]+=1

for i in range(h):
    for j in range(w):
        if votmat[i,j]<73:
            votmat[i,j]=0

best=np.array(votmat)
votmat=np.array(best)



newmat=np.zeros((h,w))
center=[]
for i in tqdm(range(rad+5,h-rad-5)):
    for j in range(rad+5,w-rad-5):
        d=0
        f=True
        for x,y in center:
            if ((x-i)**2)+((y-j)**2)**0.5<70:
                f=False
                break
        if (votmat[i,j]!=0) and f:
            center.append((i,j))
            for k in range(360):
                newmat[i+(rad*np.cos(k*np.pi/180)),j+(rad*np.sin(k*np.pi/180))]=1
            
plt.imshow(image,cmap='gray')

for i in range(h):
    for j in range(w):
        if newmat[i,j]!=0:
            image[i,j]=255



