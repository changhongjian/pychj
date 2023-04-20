# -*- coding:utf-8 -*
import numpy as np
import cv2

def gen_rand_img(w=512, h=512, b=64):
    img=np.zeros( (w, h, 3), np.uint8 )
    for i in range(h//b):
        for j in range(w//b):
            c = np.random.randint(0, 256, 3) #.tolist()
            img[ i*b:(i+1)*b, j*b:(j+1)*b ] = c[None, None, :]
    return img
#https://stackoverflow.com/questions/36637400/how-to-normalize-opencv-sobel-filter-given-kernel-size
def img_gradient(img, type, dx, dy, ksize):
    deriv_filter = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=True)
    return cv2.sepFilter2D(img, type, deriv_filter[0], deriv_filter[1])

def img_gradient_xy(img, ksize=3):
    dIx = img_gradient(img, cv2.CV_64F, 1, 0, ksize)
    dIy = img_gradient(img, cv2.CV_64F, 0, 1, ksize)
    
    return dIx, dIy


def img_gradient_xy32F(img, ksize=3):
    dIx = img_gradient(img, cv2.CV_32F, 1, 0, ksize)
    dIy = img_gradient(img, cv2.CV_32F, 0, 1, ksize)

    return dIx, dIy

def isblur(frame,thed=400,thd2=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r1 = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    r2=numpy.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))
    
    #if r1> 400 and r2> 100: return false
    if r2> 120: return false
    return true
   
# @2021-3-13
def show_color_idx(_COLORS, col, b, fsize=1, fweight=1):
    NC = len(_COLORS)
    row = (NC-1) // col + 1
    img=np.zeros( (row*imgc, col*imgc, 3) , np.uint8)

    for i in range(row):
        for j in range(col):
            idx = i*col+j
            if idx>=NC: break
            img[ i*imgc:(i+1)*imgc, j*imgc:(j+1)*imgc ] = _COLORS[idx]
            pt = (int( (j+0.2)*imgc ), int( (i+0.6)*imgc) )
            cv2.putText(img,str(idx),pt,cv2.FONT_HERSHEY_COMPLEX,fsize, (0,0,0),fweight)
    return img

