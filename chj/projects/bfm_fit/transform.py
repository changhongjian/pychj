import numpy as np

def rot_90_clockwise(img, lm):
    img = img.transpose(1, 0, 2)[:, ::-1].copy()
    lm = lm[:, ::-1].copy()
    lm[:, 0] = img.shape[1] - lm[:, 0]
    return img, lm

def solve_Orthogonal3D_2D_sR_t2d_v2(src, dst, wet=None):
    N = src.shape[0]

    A = np.zeros( (N*2, 8) )
    b = dst.reshape(-1).copy() 

    for i in range(N):
        A[i*2, :3] = src[i]
        A[i*2+1, 4:-1] = src[i]
        A[i*2, 3] = 1
        A[i*2+1, 7] = 1


    if wet is not None:
        wet= wet.repeat(2)

        b*=wet
        A*=wet[:, None]

    x = np.linalg.lstsq(A, b, rcond=None)[0]

    x = x.reshape(2, 4)

    t2d = x[:, -1]
    x = x[:, :3] #.copy()
    s1 = np.linalg.norm(x[0])
    s2 = np.linalg.norm(x[1])

    #s = np.sqrt(s1*s2)
    s = (s1+s2)/2
    x[0] /= s1
    x[1] /= s2

    r3 = np.cross(x[0], x[1])
    R = np.zeros((3, 3))
    R[:2] = x
    R[2] = r3

    u, w, vt = np.linalg.svd(R)
    R = u.dot(vt)
    if np.linalg.det(R) < 0:
        R[-1, :] = -R[-1, :]


    return [ e.astype(src.dtype) for e in [ s, R, t2d ] ]
    #return  s, R, t2d 

def solve_Orthogonal3D_2D_sR_t2d(src, dst):
    N = src.shape[0]
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)

    src_ = src - np.tile(src_mean.reshape((1, -1)), (N, 1))
    dst_ = dst - np.tile(dst_mean.reshape((1, -1)), (N, 1))
    
    # https://zhuanlan.zhihu.com/p/35901184
    # u, w, vt = np.linalg.svd(src_, full_matrices=False)
    u, w, vt = np.linalg.svd(src_)
    R = dst_.T.dot(u[:, 0:vt.shape[0]]).dot(np.diag(1. / w)).dot(vt)
    u, w, vt = np.linalg.svd(R)

    # 这里有些不一样的地方
    u = np.array([[u[0, 0], u[0, 1], 0], [u[1, 0], u[1, 1], 0], [0, 0, 1]])
    R = u.dot(vt)
    if np.linalg.det(R) < 0:
        R[-1, :] = -R[-1, :]

    sR = w.mean() * R
    t2d = dst_mean - src_mean.dot(sR[0:2].T)
    # return sR, t2d
    return w.mean(), R, t2d


def transform_Similarity3D_sR_t2d(src, sR, t2d):
    N = src.shape[0]
    dst = src.dot(sR.T)
    dst[:, :2] += t2d
    return dst


def transform_Similarity3D_sR_t3d(src, sR, t3d):
    dst = src.dot(sR.T) + t3d
    return dst


def solve_Similarity3D_sR_t3d(src, dst):
    N = src.shape[0]
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)

    src_ = src - np.tile(src_mean.reshape((1, -1)), (N, 1))
    dst_ = dst - np.tile(dst_mean.reshape((1, -1)), (N, 1))

    # https://zhuanlan.zhihu.com/p/35901184
    # u, w, vt = np.linalg.svd(src_, full_matrices=False)
    u, w, vt = np.linalg.svd(src_)
    R = dst_.T.dot(u[:, 0:vt.shape[0]]).dot(np.diag(1. / w)).dot(vt)
    u, w, vt = np.linalg.svd(R)
    R = u.dot(vt)

    sR = w.mean() * R
    t3d = dst_mean - src_mean.dot(sR.T)
    return w.mean(), R, t3d


