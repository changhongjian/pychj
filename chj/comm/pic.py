# -*- coding:utf-8 -* 
import os ; import sys 
#os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

import re
import numpy 
import numpy as np
import cv2 as cv
import cv2
from PIL import Image
import scipy
import matplotlib.pyplot as plt
try: 
    import torch
except ImportError:
    pass
import glob
import math
from tqdm import tqdm
import random
import copy
import pathlib
from pathlib import Path
import multiprocessing
import time
from easydict import EasyDict as edict
import chj.speed.split_run as chj_split_run
from chj.base.sys import exec_cmd
from chj.base.file import readlines, chdir
import chj.base.file as chj_file


def np2th(e): return torch.from_numpy(e) if type(e)==np.ndarray else e
def np2th_gpu(e): return torch.from_numpy(e).cuda()
def th2np(e): return e.detach().cpu().numpy()

def ps(dt): print(dt.shape if hasattr(dt, "shape") else len(dt) )
def p(*info): print(*info)

def showimg(img,nm="pic",wait=0):
    cv2.imshow(nm,img)
    return cv2.waitKey(wait)
def showImg(img,nm="pic",waite=0):
    cv2.imshow(nm,img)
    return cv2.waitKey(waite)
def showimg_task(img,nm="pic", isexits=True):
    key=-1
    while key!=32:
        cv2.imshow(nm,img)
        key=cv2.waitKey()
        if key==27: 
            if isexits: exit()
            else: return key

def drawLine(img, pts, color=(0,255,0), thickness=2, lineType=1):
    for i in range(len(pts)-1):
        ptStart, ptEnd = pts[i], pts[i+1]
        cv.line(img, tuple( ptStart ), tuple( ptEnd ), color, thickness, lineType)

def drawCircle(img,x,y,color=(0,255,0),size=2,shift=0):
    for id in range(len(x)):
        if shift==0:
            cv2.circle(img,(int(x[id]),int(y[id])),1,color, size)
        else:
            scale=2**shift
            cv2.circle(img,(int(x[id]*scale),int(y[id]*scale)),1*scale,color,size,shift=shift)

def drawCirclev2(img,xy,color=(0,255,0),size=2,shift=0):
    drawCircle(img, xy[:,0],xy[:,1], color, size,shift)

def drawRect(img,rect,color=(255,0,0)):
    r=[ int(x) for x in rect ]
    cv2.rectangle(img,(r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color,1)
    
def drawRectXY(img,rect,color=(255,0,0), size=1):
    cv2.rectangle(img,(int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), color,size)    
    
def drawIds(img,x,y,color=(0,0,255), baseid=0):
    for id in range(len(x)):
        cv2.putText( img, str(id+baseid),(int(x[id]),int(y[id])), 1,0.5,color, 1);

def drawIdsv2(img,xy,color=(0,0,255), baseid=0, fontsz=1, fontbold=1):
    x, y = xy.T.astype(np.int32)
    for id in range(len(x)):
        cv2.putText( img, str(id+baseid),(x[id],y[id]), 1, fontsz, color, fontbold)

def drawTxt(img,txt,xy,color=(0,0,255), baseid=0, fontsz=1, fontbold=1):
    x, y = xy
    cv2.putText( img, str(txt),(x,y), 1, fontsz, color, fontbold)

def loop_show(func_gain_img,nm="pic"):
    while True:
        img=func_gain_img()
        if img is None: break
        cv2.imshow(nm,img)
        key=cv2.waitKey(1)
        if key==27:break
        

