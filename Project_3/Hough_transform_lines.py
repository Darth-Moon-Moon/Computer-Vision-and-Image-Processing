# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:39:31 2018

@author: vignajeeth
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import copy


image = cv2.imread('hough.jpg',0)
h,w=image.shape
sobel_out=np.zeros((h,w))
sobel=[[-1,0,1],[-2,0,2],[-1,0,1]]
sobel=np.array(sobel)

for _ in range(3):    
    for i in range(1,h-1):
        for j in range(1,w-1):
            sobel_out[i,j]+=np.sum(sobel*image[i-1:i+2,j-1:j+2])
plt.imshow(sobel_out)

sobel_out=np.abs(sobel_out)
sobel_out[sobel_out<600]=0
#sobel_out=threshold(sobel_out)


D=int((((h-1)**2+(w-1**2))**0.5))
nrho=(2*D)+1
ac_mat=np.zeros((nrho,360))

for i in tqdm(range(1,h-1)):
    for j in range(1,w-1):
        if sobel_out[i,j]!=0:
            for k in range(360):
                r=j*np.cos((k-90)*3.141592654/180)+i*np.sin((k-90)*3.141592654/180)
                ac_mat[int(r),k]+=1

ac_mat_copy=np.array(ac_mat)
plt.imshow(ac_mat,aspect='auto',cmap='gray')
#ac_mat=np.array(ac_mat_copy)

def perplines(degree=90,linesdrawn=10):
    #degree=90    
    image = cv2.imread('hough.jpg',0)
    large=ac_mat[:,degree-2:degree+2]
    maxima=[]
    #np.where()
    for i in range(linesdrawn):         #Works for 8
        rho,theta=np.unravel_index(np.argmax(large, axis=None), large.shape)
        large[rho,theta]=0    
        theta=theta+degree-2-90
    #    ac_mat[rho,theta+85]=0
        maxima.append([rho,theta])
    
    for rho,theta in maxima:
        theta=(theta)*3.141592654/180
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    plt.imshow(image,cmap='gray')

perplines(90,10)












#----------------------------------------------------------------


import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import copy

image = cv2.imread('hough.jpg',0)
h,w=image.shape
sobel_out=np.zeros((h,w))
sobel=[[-1,0,1],[-2,0,2],[-1,0,1]]
sobel=np.array(sobel)


for _ in range(1):    
    for i in range(1,h-1):
        for j in range(1,w-1):
            sobel_out[i,j]+=np.sum(sobel.T*image[i-1:i+2,j-1:j+2])
plt.imshow(sobel_out)

sobel_out=np.abs(sobel_out)
sobel_out[sobel_out>150]=0
sobel_out[sobel_out<20]=0



D=int((((h-1)**2+(w-1**2))**0.5))
nrho=(2*D)+1
ac_mat=np.zeros((nrho,180))

for i in tqdm(range(1,h-1)):
    for j in range(1,w-1):
        if sobel_out[i,j]!=0:
            for k in range(180):
                r=j*np.cos((k-90)*3.141592654/180)+i*np.sin((k-90)*3.141592654/180)
                ac_mat[int(r),k]+=1

ac_mat_copy=np.array(ac_mat)
plt.imshow(ac_mat,aspect='auto',cmap='gray')
#ac_mat=np.array(ac_mat_copy)
#def nonperplines(degree=55,linesdrawn=10):

degree=55
image = cv2.imread('hough.jpg',0)
large=ac_mat[:,degree-1:degree+1]
maxima=[]
#plt.imshow(large,aspect='auto',cmap='gray')
#large=np.array(large_temp)
temp=copy.deepcopy(large)
#np.where()
for i in range(40):         #Works for 8
    rho,theta=np.unravel_index(np.argmax(large, axis=None), large.shape)
#    large[rho-3:rho+4,theta-1:theta+2]=0    
    large[rho,theta]=0    
    theta=theta+degree-1-90
    maxima.append([rho,theta])



for rho,theta in maxima:
    theta=(theta)*3.141592654/180
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

plt.imshow(image,cmap='gray')

#large=copy.deepcopy(temp)

# Reference: 
# Opencv








