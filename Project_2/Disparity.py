# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 01:43:29 2018

@author: vignajeeth
"""


from matplotlib import pyplot as plt
import numpy as np
import cv2
#plt.imshow(img1,),plt.show()
UBIT = 'vignajee'
np.random.seed(sum([ord(c) for c in UBIT]))

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def Task211():
    img1 = cv2.imread('tsucuba_left.png',0)
    img2 = cv2.imread('tsucuba_right.png',0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = sift.detect(img1,None)
    kp2 = sift.detect(img2,None)
    img1=cv2.drawKeypoints(img1,kp1,img1)
    cv2.imwrite('task2_sift1.jpg',img1)
    img2=cv2.drawKeypoints(img2,kp2,img2)
    cv2.imwrite('task2_sift2.jpg',img2)


def Task212():
    img1 = cv2.imread('tsucuba_left.png',0) 
    img2 = cv2.imread('tsucuba_right.png',0)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)#check
    search_params = dict(checks=50)#check
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)#check
    
    matchesMask = [[0,0] for i in range(len(matches))]
    arr=[]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
            arr.append(m)
    
    
    draw_params = dict(matchColor = (0,255,0),flags = 0,singlePointColor = (255,0,0))
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,arr,None,**draw_params)#check
    cv2.imwrite('task2_matches_knn.jpg',img3)
    #    plt.imshow(img3,),plt.show()



def Task23():
    img1 = cv2.imread('tsucuba_left.png',0)
    img2 = cv2.imread('tsucuba_right.png',0)
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp2, des2 = sift.detectAndCompute(img2,None)
    kp1, des1 = sift.detectAndCompute(img1,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print(F)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    #--------
    newpts1=pts1[np.random.choice(192,10)]
    newpts2=pts2[np.random.choice(192,10)]
    
    lines1 = cv2.computeCorrespondEpilines(newpts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,newpts1,newpts2)
    lines2 = cv2.computeCorrespondEpilines(newpts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,newpts2,newpts1)
    #plt.subplot(121),plt.imshow(img5)
    #plt.subplot(122),plt.imshow(img3)
    #plt.show()
    cv2.imwrite('task2_epi_right.jpg',img3)
    cv2.imwrite('task2_epi_left.jpg',img5)


def Task25():
    imgL = cv2.imread('tsucuba_left.png',0)
    imgR = cv2.imread('tsucuba_right.png',0)
    
    stereo = cv2.StereoSGBM_create(minDisparity=-16,
    numDisparities=64,blockSize=10,P1=0,P2=0,disp12MaxDiff=0,
    preFilterCap=100,uniquenessRatio=15,speckleWindowSize=50,speckleRange=2)
    
    disparity = stereo.compute(imgL,imgR)
    #plt.imshow(disparity,'gray')
    #plt.show()
#    cv2.imwrite('task2_disparity.jpg',disparity)
    #plt.savefig('task2_disparityppp.jpg',disparity,c=plt.cm.gray)
    plt.imsave('task2_disparity.jpg',disparity,cmap=plt.cm.gray)


Task211()
Task212()
Task23()
Task25()


