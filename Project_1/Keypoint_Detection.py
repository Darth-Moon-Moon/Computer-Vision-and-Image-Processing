# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 02:47:45 2018

@author: vignajeeth
"""
#a=[[1,2,3,4,5],[11,12,13,14,15],[21,22,23,24,25],[31,32,33,34,35],[41,42,43,44,45]]
#b=[[]]
#b[0].append(1)

import cv2
import numpy as np


def Zeromatrix(finalylen,finalxlen):
    fin_img=[]
    for i in range(finalylen):
        fin_img.append([0]*finalxlen)
    return(fin_img)

def Create_Gaussian(dims=7,sigma=1):
    arr=[]
    u=int(dims/2)
    for i in range(u,u-dims,-1):
        a=[]
        for j in range(-u,dims-u,1):
            a.append(j**2+i**2)
        arr.append(a)
    
    arr=np.asarray(arr)
    gauss=(np.exp(-arr/(2*(sigma**2))))/(2*3.141592654*(sigma**2))
    return(gauss)

def GaussianFilter(ini_img,gauss,x,y):
    sums=0
    for i in range(-3,4):
        for j in range(3,-4,-1):
            sums=(ini_img[x+i][y+j]*gauss[i][j])+sums
    return(sums)
            
    
#--------------MAIN-----------------------


img=cv2.imread('/home/vignajeeth/python/Graduate_Codes/CVIP/task2.jpg',0)
Imagf=cv2.imread('/home/vignajeeth/python/Graduate_Codes/CVIP/task2.jpg')

Image=np.asarray(img)
Y_SIZE=len(Image)
X_SIZE=len(Image[0])

#----Octave 1------

Image1=Zeromatrix(Y_SIZE+6,X_SIZE+6)

ImageFin11=Zeromatrix(Y_SIZE,X_SIZE)
ImageFin12=Zeromatrix(Y_SIZE,X_SIZE)
ImageFin13=Zeromatrix(Y_SIZE,X_SIZE)
ImageFin14=Zeromatrix(Y_SIZE,X_SIZE)
ImageFin15=Zeromatrix(Y_SIZE,X_SIZE)

k=2**0.5

gauss11=Create_Gaussian(7,2**(-0.5))
gauss12=Create_Gaussian(7,k*2**(-0.5))
gauss13=Create_Gaussian(7,k*k*2**(-0.5))
gauss14=Create_Gaussian(7,k*k*k*2**(-0.5))
gauss15=Create_Gaussian(7,k*k*k*k*2**(-0.5))

for i in range(Y_SIZE):
    for j in range(X_SIZE):
        Image1[i+3][j+3]=Image[i][j]
        
for i in range(Y_SIZE):
    for j in range(X_SIZE):
        ImageFin11[i][j]=GaussianFilter(Image1,gauss11,i,j)
        ImageFin12[i][j]=GaussianFilter(Image1,gauss12,i,j)
        ImageFin13[i][j]=GaussianFilter(Image1,gauss13,i,j)
        ImageFin14[i][j]=GaussianFilter(Image1,gauss14,i,j)
        ImageFin15[i][j]=GaussianFilter(Image1,gauss15,i,j)


Image1=np.asarray(Image1)
ImageFin11=np.asarray(ImageFin11)
ImageFin12=np.asarray(ImageFin12)
ImageFin13=np.asarray(ImageFin13)
ImageFin14=np.asarray(ImageFin14)
ImageFin15=np.asarray(ImageFin15)

cv2.imwrite("Image11.png",ImageFin11)
cv2.imwrite("Image12.png",ImageFin12)
cv2.imwrite("Image13.png",ImageFin13)
cv2.imwrite("Image14.png",ImageFin14)
cv2.imwrite("Image15.png",ImageFin15)



#-----------Octave 2--------------

Imageby2=[]
for i in range(0,Y_SIZE,2):
    Imageby2.append(Image[i][::2])
Imageby2=np.asarray(Imageby2)

Y_SIZE_2=len(Imageby2)
X_SIZE_2=len(Imageby2[0])

Image2=Zeromatrix(Y_SIZE_2+6,X_SIZE_2+6)

ImageFin21=Zeromatrix(Y_SIZE_2,X_SIZE_2)
ImageFin22=Zeromatrix(Y_SIZE_2,X_SIZE_2)
ImageFin23=Zeromatrix(Y_SIZE_2,X_SIZE_2)
ImageFin24=Zeromatrix(Y_SIZE_2,X_SIZE_2)
ImageFin25=Zeromatrix(Y_SIZE_2,X_SIZE_2)

k=2**0.5

gauss21=Create_Gaussian(7,(k**2)*2**(-0.5))
gauss22=Create_Gaussian(7,(k**3)*2**(-0.5))
gauss23=Create_Gaussian(7,(k**4)*2**(-0.5))
gauss24=Create_Gaussian(7,(k**5)*2**(-0.5))
gauss25=Create_Gaussian(7,(k**6)*2**(-0.5))

for i in range(Y_SIZE_2):
    for j in range(X_SIZE_2):
        Image2[i+3][j+3]=Imageby2[i][j]
        
for i in range(Y_SIZE_2):
    for j in range(X_SIZE_2):
        ImageFin21[i][j]=GaussianFilter(Image2,gauss21,i,j)
        ImageFin22[i][j]=GaussianFilter(Image2,gauss22,i,j)
        ImageFin23[i][j]=GaussianFilter(Image2,gauss23,i,j)
        ImageFin24[i][j]=GaussianFilter(Image2,gauss24,i,j)
        ImageFin25[i][j]=GaussianFilter(Image2,gauss25,i,j)


Image2=np.asarray(Image2)
ImageFin21=np.asarray(ImageFin21)
ImageFin22=np.asarray(ImageFin22)
ImageFin23=np.asarray(ImageFin23)
ImageFin24=np.asarray(ImageFin24)
ImageFin25=np.asarray(ImageFin25)

cv2.imwrite("Image21.png",ImageFin21)
cv2.imwrite("Image22.png",ImageFin22)
cv2.imwrite("Image23.png",ImageFin23)
cv2.imwrite("Image24.png",ImageFin24)
cv2.imwrite("Image25.png",ImageFin25)






#-----------Octave 3--------------

Imageby4=[]
for i in range(0,int(Y_SIZE/2),2):
    Imageby4.append(Imageby2[i][::2])
Imageby4=np.asarray(Imageby4)

Y_SIZE_3=len(Imageby4)
X_SIZE_3=len(Imageby4[0])

Image3=Zeromatrix(Y_SIZE_3+6,X_SIZE_3+6)

ImageFin31=Zeromatrix(Y_SIZE_3,X_SIZE_3)
ImageFin32=Zeromatrix(Y_SIZE_3,X_SIZE_3)
ImageFin33=Zeromatrix(Y_SIZE_3,X_SIZE_3)
ImageFin34=Zeromatrix(Y_SIZE_3,X_SIZE_3)
ImageFin35=Zeromatrix(Y_SIZE_3,X_SIZE_3)

k=2**0.5

gauss31=Create_Gaussian(7,(k**4)*2**(-0.5))
gauss32=Create_Gaussian(7,(k**5)*2**(-0.5))
gauss33=Create_Gaussian(7,(k**6)*2**(-0.5))
gauss34=Create_Gaussian(7,(k**7)*2**(-0.5))
gauss35=Create_Gaussian(7,(k**8)*2**(-0.5))

for i in range(Y_SIZE_3):
    for j in range(X_SIZE_3):
        Image3[i+3][j+3]=Imageby4[i][j]
        
for i in range(Y_SIZE_3):
    for j in range(X_SIZE_3):
        ImageFin31[i][j]=GaussianFilter(Image3,gauss31,i,j)
        ImageFin32[i][j]=GaussianFilter(Image3,gauss32,i,j)
        ImageFin33[i][j]=GaussianFilter(Image3,gauss33,i,j)
        ImageFin34[i][j]=GaussianFilter(Image3,gauss34,i,j)
        ImageFin35[i][j]=GaussianFilter(Image3,gauss35,i,j)


Image3=np.asarray(Image3)
ImageFin31=np.asarray(ImageFin31)
ImageFin32=np.asarray(ImageFin32)
ImageFin33=np.asarray(ImageFin33)
ImageFin34=np.asarray(ImageFin34)
ImageFin35=np.asarray(ImageFin35)

cv2.imwrite("Image31.png",ImageFin31)
cv2.imwrite("Image32.png",ImageFin32)
cv2.imwrite("Image33.png",ImageFin33)
cv2.imwrite("Image34.png",ImageFin34)
cv2.imwrite("Image35.png",ImageFin35)




#-----------Octave 4--------------

Imageby8=[]
for i in range(0,int(Y_SIZE/4),2):
    Imageby8.append(Imageby4[i][::2])
Imageby8=np.asarray(Imageby8)

Y_SIZE_4=len(Imageby8)
X_SIZE_4=len(Imageby8[0])

Image4=Zeromatrix(Y_SIZE_3+6,X_SIZE_3+6)

ImageFin41=Zeromatrix(Y_SIZE_4,X_SIZE_4)
ImageFin42=Zeromatrix(Y_SIZE_4,X_SIZE_4)
ImageFin43=Zeromatrix(Y_SIZE_4,X_SIZE_4)
ImageFin44=Zeromatrix(Y_SIZE_4,X_SIZE_4)
ImageFin45=Zeromatrix(Y_SIZE_4,X_SIZE_4)

k=2**0.5

gauss41=Create_Gaussian(7,(k**6)*2**(-0.5))
gauss42=Create_Gaussian(7,(k**7)*2**(-0.5))
gauss43=Create_Gaussian(7,(k**8)*2**(-0.5))
gauss44=Create_Gaussian(7,(k**9)*2**(-0.5))
gauss45=Create_Gaussian(7,(k**10)*2**(-0.5))

for i in range(Y_SIZE_4):
    for j in range(X_SIZE_4):
        Image4[i+3][j+3]=Imageby8[i][j]
        
for i in range(Y_SIZE_4):
    for j in range(X_SIZE_4):
        ImageFin41[i][j]=GaussianFilter(Image4,gauss41,i,j)
        ImageFin42[i][j]=GaussianFilter(Image4,gauss42,i,j)
        ImageFin43[i][j]=GaussianFilter(Image4,gauss43,i,j)
        ImageFin44[i][j]=GaussianFilter(Image4,gauss44,i,j)
        ImageFin45[i][j]=GaussianFilter(Image4,gauss45,i,j)


Image4=np.asarray(Image4)
ImageFin41=np.asarray(ImageFin41)
ImageFin42=np.asarray(ImageFin42)
ImageFin43=np.asarray(ImageFin43)
ImageFin44=np.asarray(ImageFin44)
ImageFin45=np.asarray(ImageFin45)

cv2.imwrite("Image41.png",ImageFin41)
cv2.imwrite("Image42.png",ImageFin42)
cv2.imwrite("Image43.png",ImageFin43)
cv2.imwrite("Image44.png",ImageFin44)
cv2.imwrite("Image45.png",ImageFin45)

DOG11=ImageFin11-ImageFin12
DOG12=ImageFin12-ImageFin13
DOG13=ImageFin13-ImageFin14
DOG14=ImageFin14-ImageFin15


DOG21=ImageFin21-ImageFin22
DOG22=ImageFin22-ImageFin23
DOG23=ImageFin23-ImageFin24
DOG24=ImageFin24-ImageFin25


DOG31=ImageFin31-ImageFin32
DOG32=ImageFin32-ImageFin33
DOG33=ImageFin33-ImageFin34
DOG34=ImageFin34-ImageFin35


DOG41=ImageFin41-ImageFin42
DOG42=ImageFin42-ImageFin43
DOG43=ImageFin43-ImageFin44
DOG44=ImageFin44-ImageFin45


def Comp(left,mid,right):
    L=np.max(left)
    R=np.max(right)
    M=np.max(mid)
    T=np.max([L,R,M])
    l=np.min(left)
    r=np.min(right)
    m=np.min(mid)
    t=np.min([l,r,m])
        
    if (mid[1][1]==t) or (mid[1][1]==T):
        return(True)
    else:
        return(False)


KP11=[]
KP12=[]
KP21=[]
KP22=[]
KP31=[]
KP32=[]
KP41=[]
KP42=[]

for i in range(Y_SIZE-3):
    for j in range(X_SIZE-3):
        if(Comp(DOG11[i:i+3,j:j+3],DOG12[i:i+3,j:j+3],DOG13[i:i+3,j:j+3])):
            KP11.append((i+1,j+1))
        if(Comp(DOG12[i:i+3,j:j+3],DOG13[i:i+3,j:j+3],DOG14[i:i+3,j:j+3])):
            KP12.append((i+1,j+1))


for i in range(Y_SIZE_2-3):
    for j in range(X_SIZE_2-3):
        if(Comp(DOG21[i:i+3,j:j+3],DOG22[i:i+3,j:j+3],DOG23[i:i+3,j:j+3])):
            KP21.append((i+1,j+1))
        if(Comp(DOG22[i:i+3,j:j+3],DOG23[i:i+3,j:j+3],DOG24[i:i+3,j:j+3])):
            KP22.append((i+1,j+1))


#for i in range(Y_SIZE_3-3):
 #   for j in range(X_SIZE-3):
#        if(Comp(DOG31[i:i+3,j:j+3],DOG32[i:i+3,j:j+3],DOG33[i:i+3,j:j+3])):
 #           KP31.append((i+1,j+1))
#        if(Comp(DOG32[i:i+3,j:j+3],DOG33[i:i+3,j:j+3],DOG34[i:i+3,j:j+3])):
 #           KP32.append((i+1,j+1))


for i in range(Y_SIZE_4-3):
    for j in range(X_SIZE_4-3):
        if(Comp(DOG41[i:i+3,j:j+3],DOG42[i:i+3,j:j+3],DOG43[i:i+3,j:j+3])):
            KP41.append((i+1,j+1))
        if(Comp(DOG42[i:i+3,j:j+3],DOG43[i:i+3,j:j+3],DOG44[i:i+3,j:j+3])):
            KP42.append((i+1,j+1))


for i,j in KP11:
    if(i>6) and (j>6):
        Imagf[i][j]=255

for i,j in KP12:
    if(i>6) and (j>6):
        Imagf[i][j]=255


for i,j in KP21:
    if(i>6) and (j>6):
        Imagf[i*2][j*2]=255
        Image2[i][j]=255

for i,j in KP22:
    if(i>6) and (j>6):
        Imagf[i*2][j*2]=255
        Image2[i][j]=255

for i,j in KP31:
    if(i>6) and (j>6):
        Imagf[i*4][j*4]=255

for i,j in KP32:
    if(i>6) and (j>6):
        Imagf[i*4][j*4]=255


for i,j in KP41:
    if(i>6) and (j>6):
        Imagf[i*8][j*8]=255

for i,j in KP42:
    if(i>6) and (j>6):
        Imagf[i*8][j*8]=255

#Image2=Image[::2]

#cv2.imwrite("DOG31.png",DOG31)
#cv2.imwrite("DOG32.png",DOG32)
#cv2.imwrite("DOG33.png",DOG33)
#cv2.imwrite("DOG34.png",DOG34)
#cv2.waitKey(0)

#cv2.imwrite("Octave2.png",Image2)

#cv2.waitKey(0)

#cv2.destroyAllWindows()




