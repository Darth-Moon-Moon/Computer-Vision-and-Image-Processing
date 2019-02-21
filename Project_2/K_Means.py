# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 04:22:18 2018

@author: vignajeeth
"""

import numpy as np
from matplotlib import pyplot as plt

def dist(a,b):
    ans=(((a[0]-b[0])**2)+((a[1]-b[1])**2))**0.5
    return(ans)


k=3

X=[[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5,3],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6,3]]
centre=[[6.2,3.2],[6.6,3.7],[6.5,3]]
distance=np.zeros((k))
#centre=np.zeros((3,2))
X=np.asarray(X)
centre=np.asarray(centre)
clas=[0]*k

for _ in range(10):    
    classification=[[],[],[]]
    for i in range(10):
        for j in range(k):
            distance[j]=(dist(X[i],centre[j]))
        classification[np.argmin(distance)].append(X[i])
    for i in range(k):
        clas[i]=np.asarray(classification[i])# Sorts into classes
        centre[i]=np.mean(clas[i],axis=0)# Finds the new center


plt.scatter(clas[0][:,0],clas[0][:,1],marker='^',c='r',edgecolors='face')
plt.scatter(clas[1][:,0],clas[1][:,1],marker='^',c='g',edgecolors='face')
plt.scatter(clas[2][:,0],clas[2][:,1],marker='^',c='b',edgecolors='face')

plt.scatter(centre[0,0],centre[0,1],marker='o',c='r',edgecolors='face')
plt.scatter(centre[1,0],centre[1,1],marker='o',c='g',edgecolors='face')
plt.scatter(centre[2,0],centre[2,1],marker='o',c='b',edgecolors='face')
for xy in zip(clas[0][:,0], clas[0][:,1]):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
for xy in zip(clas[1][:,0], clas[1][:,1]):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
for xy in zip(clas[2][:,0], clas[2][:,1]):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
for xy in zip(centre[:,0], centre[:,1]):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')



