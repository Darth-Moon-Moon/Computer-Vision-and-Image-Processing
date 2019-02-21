

import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


def threshold(sobel_out,t=200):
    sobel_out=np.abs(sobel_out)
    h,w=sobel_out.shape
    for i in range(h):
        for j in range(w):
            if sobel_out[i,j]<t:
                sobel_out[i,j]=0
            else:
                sobel_out[i,j]=1
    return(sobel_out)

def segment():
    Oimage = cv2.imread('segment.jpg',0)
    image=np.array(Oimage)
    h,w=image.shape
    m=np.max(image)+1
    density=[0]*m
    
    for i in range(h):
        for j in range(w):
            density[image[i,j]]+=1
    
    image[image<206]=0
    
    cv2.imwrite('output.png',image)
    cv2.rectangle(Oimage,(164,125),(199,161),255)
    cv2.rectangle(Oimage,(253,77),(297,204),255)
    cv2.rectangle(Oimage,(334,23),(362,284),255)
    cv2.rectangle(Oimage,(389,42),(419,248),255)
    
    cv2.rectangle(image,(164,125),(199,161),255)
    cv2.rectangle(image,(253,77),(297,204),255)
    cv2.rectangle(image,(334,23),(362,284),255)
    cv2.rectangle(image,(389,42),(419,248),255)
    cv2.imwrite('output_box.png',image)
    cv2.imwrite('orig_box.png',Oimage)#,cmap='gray')

def point():
    image = cv2.imread('turbine-blade.jpg',0)
    h,w=image.shape
    
    point=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    point_out=np.zeros((h,w))
    for i in range(1,h-1):
        for j in range(1,w-1):
            point_out[i,j]=np.sum(point*image[i-1:i+2,j-1:j+2])
    
    
    thres_out=threshold(point_out,2000)
    ans=np.unravel_index(np.argmax(thres_out,axis=None),thres_out.shape)
    ans=(ans[1],ans[0])
    plt.annotate('(%s, %s)' % ans, xy=ans, textcoords='data')
    plt.imshow(thres_out)


point()
segment()





