# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 04:22:18 2018

@author: vignajeeth
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import timeit
#import tqdm
UBIT = 'vignajee'
np.random.seed(sum([ord(c) for c in UBIT]))

def dist(a,b):
    ans=0
    for i in range(len(a)):
        ans=ans+(a[i]-b[i])**2
    return(ans**0.5)

def baboon_clustering(clusters):
    k=clusters
    img=cv2.imread('baboon.jpg')
    mat=np.reshape(img,(512*512,3))
    centre=np.random.randint(255,size=(k,3))
    
    distance=np.zeros((k))
    mat=np.asarray(mat)
    centre=np.asarray(centre)
    clas=[0]*k
    newmat=np.zeros((262144,3))
    
    
    t1=timeit.default_timer()
    for _ in range(20):    
        classification=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(512*512):
            for j in range(k):
                distance[j]=(dist(mat[i],centre[j]))
            classification[np.argmin(distance)].append(mat[i])
        for i in range(k):
            clas[i]=np.asarray(classification[i])# Sorts into classes
            centre[i]=np.mean(clas[i],axis=0)# Finds the new center
    
    t2=timeit.default_timer()
    print((t2-t1)/60)
    
    for i in range(512**2):
        for j in range(k):
            distance[j]=(dist(mat[i],centre[j]))
            newmat[i]=centre[np.argmin(distance)]
    
    img2=np.reshape(newmat,(512,512,3))
    savefilename='task3_baboon_'+str(k)+'.jpg'    
    cv2.imwrite(savefilename,img2)
    return(img2,centre)
#    plt.imshow(img2,),plt.show()

    
#    plt.savefig(savefilename, bbox_inches='tight')


img3,centre3=baboon_clustering(3)
img5,centre5=baboon_clustering(5)
img10,centre10=baboon_clustering(10)
img20,centre20=baboon_clustering(20)










