# -*- coding:utf-8 -* 

import numpy as np

import ctypes
from ctypes import *
def p(*info): print(*info)
'''
@4-20 整理
@5-17 为了传torch gpu又增加一些函数
'''


def torch_c_ptr(thmat):
    state_ptr = thmat.data_ptr()
    return ctypes.c_void_p(state_ptr)

class cls_speed:
    def __init__(self):
        self.cdll = None
        self.check_type=0 # 检查类型，我一开始不希望使用float32 后来为了速度才采用
        
    def load_cdll(self, fcdll):
        p("use dll", fcdll)
        self.cdll = ctypes.cdll.LoadLibrary(fcdll)
        return self
    def set_from_dict(self, mp):
        for k, v in mp.items(): 
            if type(v) is np.ndarray: self.set_mp(k, v)
            else: self.set_mp_torch(k, v)
    def set_mp(self, nm, npmat):
        if self.check_type==1:
            if npmat.dtype==np.float32: p(nm,"type is float32")
            if npmat.dtype==np.int64: p(nm,"type is int64")

        pm = npmat.ctypes.data_as(c_void_p)
        # bt=(c_char * 100)().value
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, pm)
       
    # ctype type is ok,# numpy  may has problem
    def get_mp(self, nm, shape, ctype):
        #if(ctype==np.int32): ctype= types.c_int32
        ss = bytes(nm, encoding="ascii")
        #self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int32, shape=(3,4))
        self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctype, shape=shape)
        npmat=self.cdll.get_mp(ss)
        return npmat

    def set_mp_ext(self, nm, ctype_ptr):
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, ctype_ptr)

    # cpu/gpu都可以
    def set_mp_torch(self, nm, thmat):
        self.set_mp_ext(nm, torch_c_ptr(thmat))

