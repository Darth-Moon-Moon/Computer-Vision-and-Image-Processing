# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 22:43:40 2018

@author: vignajeeth
"""

import cv2
import numpy as np

def Zeromatrix(finalylen,finalxlen):
    fin_img=[]
    for i in range(finalylen):
        fin_img.append([0]*finalxlen)
    return(fin_img)

img=cv2.imread('/home/vignajeeth/python/Graduate_Codes/CVIP/task1.png',0)
Image=np.asarray(img)
ImageVertP=Zeromatrix(Image.shape[0],Image.shape[1])
ImageHorzP=Zeromatrix(Image.shape[0],Image.shape[1])
ImageVertN=Zeromatrix(Image.shape[0],Image.shape[1])
ImageHorzN=Zeromatrix(Image.shape[0],Image.shape[1])
ImageF=Zeromatrix(Image.shape[0]+2,Image.shape[1]+2)

for i in range(Image.shape[0]):
    for j in range(Image.shape[1]):
        ImageF[i+1][j+1]=Image[i][j]
        
for i in range(0,Image.shape[0]):
    for j in range(0,Image.shape[1]):
        ImageVertP[i][j]=ImageF[i-1][j-1]+(2*ImageF[i][j-1])+ImageF[i+1][j-1]-ImageF[i-1][j+1]-(2*ImageF[i][j+1])-ImageF[i+1][j+1]
        ImageHorzP[i][j]=ImageF[i-1][j-1]+(2*ImageF[i-1][j])+ImageF[i-1][j+1]-ImageF[i+1][j-1]-(2*ImageF[i+1][j])-ImageF[i+1][j+1]
        ImageVertN[i][j]=-ImageF[i-1][j-1]-(2*ImageF[i][j-1])-ImageF[i+1][j-1]+ImageF[i-1][j+1]+(2*ImageF[i][j+1])+ImageF[i+1][j+1]
        ImageHorzN[i][j]=-ImageF[i-1][j-1]-(2*ImageF[i-1][j])-ImageF[i-1][j+1]+ImageF[i+1][j-1]+(2*ImageF[i+1][j])+ImageF[i+1][j+1]

mIVP=np.min(ImageVertP)
MIVP=np.max(ImageVertP)
mIHP=np.min(ImageHorzP)
MIHP=np.max(ImageHorzP)
mIVN=np.min(ImageVertN)
MIVN=np.max(ImageVertN)
mIHN=np.min(ImageHorzN)
MIHN=np.max(ImageHorzN)

f1=Zeromatrix(Image.shape[0],Image.shape[1])
f2=Zeromatrix(Image.shape[0],Image.shape[1])
f3=Zeromatrix(Image.shape[0],Image.shape[1])
f4=Zeromatrix(Image.shape[0],Image.shape[1])


for i in range(Image.shape[0]):
    for j in range(Image.shape[1]):
        f1[i][j]=(ImageVertP[i][j]-mIVP)/(MIVP-mIVP)
        f2[i][j]=(ImageHorzP[i][j]-mIHP)/(MIHP-mIHP)
        f3[i][j]=(ImageVertN[i][j]-mIVN)/(MIVN-mIVN)
        f4[i][j]=(ImageHorzN[i][j]-mIHN)/(MIHN-mIHN)

f1=np.asarray(f1)
f2=np.asarray(f2)
f3=np.asarray(f3)
f4=np.asarray(f4)

ImageVertP=np.asarray(ImageVertP)
ImageHorzP=np.asarray(ImageHorzP)
ImageVertN=np.asarray(ImageVertN)
ImageHorzN=np.asarray(ImageHorzN)


cv2.imwrite("Vertical Edge Pos.png",ImageVertP)
#cv2.waitKey(0)
cv2.imwrite("Horizontal Edge Pos.png",ImageHorzP)
#cv2.waitKey(0)

cv2.imwrite("Vertical Edge Neg.png",ImageVertN)
#cv2.waitKey(0)
cv2.imwrite("Horizontal Edge Neg.png",ImageHorzN)
#cv2.waitKey(0)


cv2.imshow("f1",f1)
cv2.waitKey(0)
cv2.imshow("f2",f2)
cv2.waitKey(0)

cv2.imshow("f3",f3)
cv2.waitKey(0)
cv2.imshow("f4",f4)
cv2.waitKey(0)

cv2.destroyAllWindows()

#cv2.imwrite('f3.jpeg',f1*255)

#cv2.destroyAllWindows()
