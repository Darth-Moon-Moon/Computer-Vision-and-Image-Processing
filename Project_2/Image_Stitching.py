# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 01:43:29 2018

@author: vignajeeth
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
UBIT = 'vignajee'
np.random.seed(sum([ord(c) for c in UBIT]))


def Task11():
    img1 = cv2.imread('mountain1.jpg',0)
    img2 = cv2.imread('mountain2.jpg',0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = sift.detect(img1,None)
    kp2 = sift.detect(img2,None)
    img1=cv2.drawKeypoints(img1,kp1,img1)
    cv2.imwrite('task1_sift1.jpg',img1)
    img2=cv2.drawKeypoints(img2,kp2,img2)
    cv2.imwrite('task1_sift2.jpg',img2)


def Task12():
    img1 = cv2.imread('mountain1.jpg',0) 
    img2 = cv2.imread('mountain2.jpg',0)
    
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
    cv2.imwrite('task1_matches_knn.jpg',img3)
#    plt.imshow(img3,),plt.show()


def Task145():
    img1 = cv2.imread('mountain1.jpg',0) 
    img2 = cv2.imread('mountain2.jpg',0)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good[::24] ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good[::24] ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    print(M)
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
#    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,flags = 2)
    
    img3=cv2.drawMatches(img1,kp1,img2,kp2,good[::24],None,**draw_params)
    
    cv2.imwrite('task1_matches.jpg',img3)
#    plt.imshow(img3),plt.show()

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, M)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

    result = cv2.warpPerspective(img1, Ht.dot(M), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2
    cv2.imwrite("task1_pano.jpg",result)


Task11()
Task12()
Task145()




