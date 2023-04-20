
import numpy as np
class Sel_key_frame:
    def __init__(self, drop_energy=0.05, thd=0.1):
        self.drop_energy = drop_energy
        self.thd = thd
        self.arr_v=[]
        self.m=None
        self.EET=None
        self._loss=0

    def judge(self, v):
        if len(self.arr_v)<1: return True
        _v = v - self.m
        _v2 = self.EET.dot(_v)
        #p(self.EET)
        loss = np.power(_v-_v2, 2).sum()
        self._loss = loss  
        if loss<self.thd: return False
        else: return True
    
    def _add_v(self, v):
        arr_v=self.arr_v+[v]
        
        if len(arr_v)==1:
            self.m = v.copy()
            self.EET = np.eye(len(v)) * 0
            self.arr_v = arr_v
            return None

        vs=np.array(arr_v) #n, 32

        m = vs.mean(0)
        vs -= m

        M = vs.T.dot(vs)
        s, E = np.linalg.eigh(M)
       
        _sum = s[s>0].sum()
        c =0
        # 竟然是降序
        for i, e in enumerate( s[::-1] ):    
            c+=e
            if (c/_sum) > (1-self.drop_energy): 
                E = E[:, len(s)-1-i:]
                break

        #p("use", 1+i)
        self.m = m
        self.EET = E.dot(E.T)
        self.arr_v = arr_v
        
    def query(self, v):
        if self.judge(v):
            self._add_v(v)
            return True
        return False


