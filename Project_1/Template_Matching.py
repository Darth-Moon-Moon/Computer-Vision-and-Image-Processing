# -*- coding: utf-8 -*-
"""
Created on Mon Oct 1 09:48:15 2018

@author: vignajeeth
"""

import numpy as np
import cv2
import imutils


imag=cv2.imread('/home/vignajeeth/python/Graduate_Codes/CVIP/task3/neg_10.jpg',0)

template=cv2.imread('/home/vignajeeth/python/Graduate_Codes/CVIP/task3/template.png',0)

after_gauss=cv2.GaussianBlur(imag,(3,3),1)
img=cv2.Laplacian(after_gauss,cv2.CV_32F)

arr=np.linspace(0.5,1,num=20)

ANS=[]
maxi=-100
maxi_loc=(0,0)

for i in arr:
    tempi=imutils.resize(template,width=int(template.shape[1]*i))    
    aftergausstemp=cv2.GaussianBlur(tempi,(3,3),1)
    temp=cv2.Laplacian(aftergausstemp,cv2.CV_32F)
    res = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if  max_val>maxi:
        maxi=max_val
        maxi_loc=max_loc
        scale=i
    #ANS.append([min_val, max_val, min_loc, max_loc])
    #maxi.append(max_val)

Oimg=cv2.imread('/home/vignajeeth/python/Graduate_Codes/CVIP/task3/neg_10.jpg')
cv2.rectangle(Oimg,maxi_loc,(maxi_loc[0]+int(template.shape[0]*scale),maxi_loc[1]+int(template.shape[1]*scale)),(0,0,255),2)

cv2.imwrite("negImage10.png",Oimg)
#cv2.imshow("G",template)

#cv2.waitKey(0)
