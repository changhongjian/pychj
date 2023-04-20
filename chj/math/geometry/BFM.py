
import numpy as np
import cv2 
import cv2 as cv 
from scipy.spatial.transform import Rotation as sciR

def calc_weak_proj_TS_batch(sRt, S, isrev=False):
    so3, t2d, s = sRt[:, :3], sRt[:, 3:5], sRt[:, 5:6]
    rotR = sciR.from_rotvec( so3 ).as_matrix()
    if isrev:
        TS = S.copy()
        TS[..., :2] -= t2d[:, None]
        TS /= s[:,None]
        TS = np.einsum('bac,bcd->bad',TS, rotR) 
    else:
        R = rotR * s[:,None]
        TS = np.einsum('bac,bdc->bad',S, R) 
        TS[..., :2] += t2d[:, None]

    return TS


def calc_weak_proj_TS(sRt, S, isrev=False):
    so3 = sRt[:3]
    t2d = sRt[3:5]
    s = sRt[5]
    if isrev:
        TS = S.copy()
        TS[:, :2] -= t2d
        TS /= s
        TS = TS.dot( cv.Rodrigues(so3)[0] )
    else:
        R = cv.Rodrigues(so3)[0] * s
        TS = S.dot(R.transpose())
        TS[:, :2] += t2d

    return TS

def calc_proj_TS(Rt, S):
    so3 = Rt[:3]
    t3d = Rt[3:]
    R = cv.Rodrigues(so3)[0]  

    TS = S.dot(R.transpose())
    #TS += t3d[np.newaxis]
    TS += t3d

    return TS


class BFM:
    def set_pca(self, Vm, Ese):
        self.Vm = Vm.reshape(-1)
        nV = self.Vm.shape[0]
        self.Ese = Ese.reshape(nV, -1)

        self.nV = nV

    def set_by_sel_ids(self, sel_ids):
        sel_ids = (sel_ids * 3).repeat(3).reshape(-1, 3)
        sel_ids[:, 1] += 1
        sel_ids[:, 2] += 2
        sel_ids = sel_ids.reshape(-1)

        #self.key_Vm = self.Vm[sel_ids].copy()
        self.Vm = self.Vm[sel_ids].copy()
        #Es = model[1][sel_ids].copy()
        #Ee = model[2][sel_ids].copy()
        #self.key_Ese = np.hstack((Es, Ee))
        #self.key_Ese = self.Ese[sel_ids].copy()
        self.Ese = self.Ese[sel_ids].copy()


    def get_lm3d_from_param(self, pm):
        sRt = pm[:6]
        xse = pm[6:]

        S = self.get_S(xse)
        TS = self.get_TS(sRt, S)

        return TS

    def get_S(self, xse):
        V = self.Vm + self.Ese.dot(xse).reshape(-1)

        return V.reshape(-1, 3)

    def get_TS(self, sRt, S):
        return calc_weak_proj_TS(sRt, S)

    def get_TS_Rt3d(self, Rt, S):
        return calc_proj_TS(Rt, S)

