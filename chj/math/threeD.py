import numpy as np
from scipy.spatial.transform import Rotation as sciR

def normalize( vec ):
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

# 给定三个点
def plane_normal(pts):
    a, b = pts[1] - pts[0], pts[2] - pts[0]
    n = np.cross(a, b) 
    return n / np.linalg.norm(n)

def rot_from_vector(a,b, ismat=False, precision=1e-18, isbatch=False):
    if not isbatch: a, b = [ np.expand_dims(e,0) for e in [a,b] ]
    B, D = a.shape
    va, vb = [ (e/(np.linalg.norm(e))[...,np.newaxis] + precision ) for e in [a, b] ]
    v = np.cross(va, vb)
    c = (va*vb).sum(-1)
    s = np.linalg.norm(v, axis=-1) + precision
    v /= s[...,np.newaxis]
    theta = np.arcsin(s)
    rot = v*theta[..., np.newaxis]
    #rot[:]=[0,-0.5,0]
    if ismat: rot = rotR = sciR.from_rotvec( rot.reshape(-1, D) ).as_matrix()
    if not isbatch: rot= rot[0]
    return rot 

def get_aug_mat44(mat, t):
    if len(mat.shape) == 1: 
        mat = sciR.from_rotvec( mat ).as_matrix()
    T = np.zeros((4,4,), dtype=mat.dtype)
    T[:3,:3] = mat
    T[:3,-1] = t
    T[-1,-1] = 1
    return T

