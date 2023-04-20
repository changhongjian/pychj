import numpy as np
import math
from scipy.spatial.transform import Rotation as sciR
import scipy
# 还有角度变换等

class Cls_ldmk_table:
    def init(self, fldmk_table):
        ldmk_table = np.load(fldmk_table)
        # 这里的min max 是角度, 按照ogl , yaw, pitch
        self.n_y_p = ldmk_table["n_y_p"]
        self.min_v = ldmk_table["min_v"]
        self.max_v = ldmk_table["max_v"]
        self.vids = ldmk_table["vids"]
        self.wets = ldmk_table["wets"]

        # 为了加速运算, 里面都按照弧度来进行
        self.min_v = angle_to_radius(self.min_v)
        self.max_v = angle_to_radius(self.max_v)
        return self

    def wets_to_th(self, iscuda=False):
        import torch
        wets=torch.from_numpy(self.wets)
        if iscuda: wets = wets.cuda()
        self.wets = wets

    # R 的唯一性
    # https://math.stackexchange.com/questions/105264/3d-rotation-matrix-uniqueness/105380#105380
    def get_lm_ids_by_R(self, R, oldid=-1, tp_mask=(1, 1, 1)):
        #yaw, pitch = get_yaw_pitch_from_Rzxy(R)
        yaw, pitch = get_face_yaw_pitch_from_R(R)


        idx = self._get_rect_id([yaw, pitch],
                                self.n_y_p, self.min_v, self.max_v)
        if idx == oldid: return [None, None, [idx]]

        res=[]
        if tp_mask[0]: res.append( self.vids[idx] )
        if tp_mask[1]: res.append(self.wets[idx])
        if tp_mask[2]: res.append( [idx, yaw, pitch] )
        return res


    def _get_id(self, val, n, v_min, v_max):
        if val <= v_min: return 0
        if val >= v_max: return n - 1
        if n <= 1: return 0
        rate = (v_max - v_min) / (n - 1)
        m = int((val - v_min) / rate + 0.5)
        return m

    def _get_rect_id(self, y_p, n_y_p, min_v, max_v):
        x = self._get_id(y_p[0], n_y_p[0], min_v[0], max_v[0])
        y = self._get_id(y_p[1], n_y_p[1], min_v[1], max_v[1])
        idx = y * n_y_p[0] + x
        return idx

    def get_only_vids_by_y_p(self, yaw, pitch, use_wet=False):
        idx = self._get_rect_id([yaw, pitch], self.n_y_p, self.min_v, self.max_v)
        if use_wet:
            return self.vids[idx], self.wets[idx]
        return self.vids[idx]

    def get_only_vids_by_R(self, R, isocv, isdbg=False, use_wet=False):
        #yaw, pitch = get_yaw_pitch_from_Rzxy(R, isocv=isocv)
        yaw, pitch = get_face_yaw_pitch_from_R(R, isocv=isocv)
        #print(yaw/np.pi*180, pitch/np.pi*180)

        idx = self._get_rect_id([yaw, pitch], self.n_y_p, self.min_v, self.max_v)
        if isdbg:
            yaw, pitch = radius_to_angle([yaw, pitch]) # int(180 * yaw / np.pi)
            return self.vids[idx], self.wets[idx], [idx, yaw, pitch]
        else:
            if use_wet:
                return self.vids[idx], self.wets[idx]
            return self.vids[idx]


def Rzxy_to_eulr(R, degrees=False):
    return sciR.from_dcm(R).as_eulr("zxy", degrees=degrees)

def Eulr_zxy_to_R(eulr, degrees=False):
    return sciR.from_euler('zxy', eulr, degrees=degrees).as_dcm()

def radius_to_angle(x):
    return np.degrees(x)


def angle_to_radius(x):
    return np.radians(x)
    #math.radians(e)

def get_face_yaw_pitch_from_R(R, isocv):
    if isocv:
        R=R.copy()
        R[1:] *=-1

    if abs(math.degrees(math.atan2(-R[0, 1], R[1, 1]))) > 15:
        yaw, pitch = math.atan2(-R[2, 0], R[2, 2]), math.asin(-R[1, 2])
    else:
        yaw, pitch = math.atan2(R[0, 2], R[2, 2]), math.asin(-R[1, 2])
    return yaw, pitch

# @19-10-8 这样似乎也是错误的
# 先做z,然后x, y https://en.wikipedia.org/wiki/Euler_angles 可以分析出
def get_yaw_pitch_from_Rzxy(R, isocv=False, is_col3=False):
    if is_col3:
        col3=R
    else:
        col3=R[:, 2]
    if isocv:
        yaw = math.atan2(col3[0], -col3[2])
        pitch = math.asin(col3[1])
    else:
        yaw = math.atan2(col3[0], col3[2])
        pitch = math.asin(-col3[1])

    return yaw, pitch
