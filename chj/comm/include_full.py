# -*- coding:utf-8 -* 
import os, sys
import platform
import re
import numpy 
import numpy as np
import scipy
import scipy.io as scio
import cv2 as cv
import cv2
from PIL import Image
from numpy import random
import time
import random
import math
import struct
from skimage import exposure
from functools import reduce
import time
import ntpath
import threading
import PIL
import shutil
import subprocess
import copy
# pytorch 需要
import torch
from easydict import EasyDict as edict
import chj.speed.split_run as chj_split_run
from chj.base.sys import exec_cmd
from chj.base.file import readlines, chdir
import chj.base.file as chj_file


def np2th(e): return torch.from_numpy(e) if type(e)==np.ndarray else e
def np2th_gpu(e): return torch.from_numpy(e).cuda()
def th2np(e): return e.detach().cpu().numpy()

def pst(dt): print(dt.shape, dt.dtype)
def ps(dt): print(dt.shape if hasattr(dt, "shape") else len(dt) )
def p(*info): print(*info)

true=True
false=False
NULL=None

def LOG_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def GPU_usage():
    info = os.popen(" nvidia-smi | grep MiB ")
    info=[x.split("|")[2] for x in info]
    ss = [x.strip() for x in info]
    for i,line in enumerate(ss):
        if len(line.strip())<2: print("GPU",i+1,line)
    return ss

def exitinfo(ss, execode=0):
    print(ss)
    exit(execode)
