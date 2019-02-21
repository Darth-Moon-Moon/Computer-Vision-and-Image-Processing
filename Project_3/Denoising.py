# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 01:40:11 2018

@author: vignajeeth
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

def normalise(image):
    image=image/np.max(image)
    image=image.astype(int)
    return(image)

def dilation(image,st_el):
#    st_el=np.ones((3,3))        #Comment this
    fin_image=np.zeros((h,w))
    for i in range(1,h-1):
        for j in range(1,w-1):
            if image[i,j]==1:
                fin_image[i-1:i+2,j-1:j+2]=st_el#*image[i-1:i+2,j-1:j+2]
    return(fin_image)

def erosion(image,st_el):
    new_image=np.invert(image)
    new_image+=(np.min(new_image)*-1)
    new_st_el=np.fliplr(np.flipud(st_el))
    new_fin=dilation(new_image,new_st_el)
    fin_image=np.invert(new_fin.astype(int))
    return(fin_image+np.min(fin_image)*-1)

def opening(image,st_el):
    return(dilation(erosion(image,st_el),st_el))

def closing(image,st_el):
    return(erosion(dilation(image,st_el).astype(int),st_el))

image = cv2.imread('noise.jpg',0)
h,w=image.shape
image=normalise(image)
st_el=np.ones((3,3)).astype(int)

clopen=opening(closing(image,st_el).astype(int),st_el)
opeclo=closing(opening(image,st_el).astype(int),st_el)

cv2.imwrite('res_noise1.jpg',clopen*255)
cv2.imwrite('res_noise2.jpg',opeclo*255)

b1=clopen-erosion(clopen.astype(int),st_el)
b2=opeclo-erosion(opeclo.astype(int),st_el)
#plt.imshow(image-opeclo,cmap='gray')
cv2.imwrite('res_bound1.jpg',b1*255)
cv2.imwrite('res_bound2.jpg',b2*255)

#plt.imshow(b2)



