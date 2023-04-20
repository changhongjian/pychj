# -*- coding:utf-8 -* 


import numpy as np
import numpy
import math
from ..solver import *

def fit3d_sets(set1, set2):   
    # 从 set1到 set2的变换

    # 求mean 其实就是质心，然后将数据变为 0均值
    mean1=set1.mean(axis=0)
    set1=set1-mean1
    mean2=set2.mean(axis=0)
    set2=set2-mean2

    # 求scale
    # 本意应该是每个点的模长平方和
    S1=np.sum(set1*set1)
    S2=np.sum(set2*set2)
    s =np.sqrt(S2/S1)

    # 求 R 最难， 说明之前自己的想法和这个不太一样，求R时可以不考虑s
    # 论文中的矩阵M, 这个看公式要小心转置，和迹函数的性质
    M=set2.transpose().dot(set1) # 不是对称的
    #如果R非奇异， R=M(M^TM)^{-1/2}
    MTM=M.transpose().dot(M)
    # 平方先不急着做，最后只对奇异值进行sqrt就行了
    u, _s, vh = np.linalg.svd(MTM)
    #_s=np.power(_s, -0.5)
    #B=u.dot(np.diag(_s).dot(vh))
    #R = M.dot(B)
    R = u.dot(vh)
    #print(R, np.linalg.norm(R))

    #t= (mean2 - s*R.dot( mean1 )).transpose()
    t= mean2 - s*R.dot( mean1 ) # 行向量
    return s,R,t

'''
很奇怪，第一次写的竟然有问题

w,v = np.linalg.eig(MTM)  # w[i] 对应 v[:,i]
    w=np.sqrt(w)
    v=np.mat(v)
    S=  w[0]*v[:,0].dot( v[:,0].transpose() ) + \
        w[1]*v[:,1].dot( v[:,1].transpose() ) + \
        w[2]*v[:,2].dot( v[:,2].transpose() )

    S=np.array(S)
    R=M.dot( np.linalg.pinv(S) ) # pinv

'''

# fg2018-competition
'''
d, Z, tform = procrustes(np.array(grundtruth_landmark_points), np.array(predicted_mesh_landmark_points),
                             scaling=True, reflection='best')
# Use tform to transform all vertices in predicted_mesh_vertices to the ground truth reference space:
predicted_mesh_vertices_aligned = []
for v in predicted_mesh_vertices:
    s = tform['scale']
    R = tform['rotation']
    t = tform['translation']
    transformed_vertex = s * np.dot(v, R) + t
    predicted_mesh_vertices_aligned.append(transformed_vertex)
'''

# 应该是 Nx3
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform

#ogl, image size
def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T

# @19-6-1
def fit3d_to_2d(S3d, lm2d, weight=None):
    S3d = S3d.reshape(-1, 3)
    lm2d = lm2d.reshape(-1, 2)
    
    nV = S3d.shape[0]
    assert nV == lm2d.shape[0], "ERROR in fit3d_to_2d: dimention not true"
    
    
    A = np.zeros( (2*nV,8) )
    b = lm2d.reshape(-1)
    
    for i in range(nV):
        A[i*2,0:3] = S3d[i]
        A[i * 2, 3] = 1
        A[i * 2+1, 4:7] = S3d[i]
        A[i * 2 + 1, 7] = 1
    
    At=A.transpose()
    if weight is not None:
        W=self.Wcs[ self.wetsid ]
        At=At.dot( W ) # 考虑加权
    A=At.dot(A)
    b=At.dot(b)
    x=solve(A,b)
    
    sRt1= x[0:3]
    sRt2 = x[4:7]
    sTx = x[3]
    sTy = x[7]
    s = np.sqrt( norm(sRt1)*norm(sRt2) )
    #s = 0.5 * (norm(sRt1) + norm(sRt2))
    r1 = sRt1 / norm(sRt1)
    r2 = sRt2 / norm(sRt2)
    r3 = np.cross(r1, r2)
    
    R = np.zeros( (3, 3) )
    
    R[0, :] = r1
    R[1, :] = r2
    R[2, :] = r3
    t1 = sTx #/ s
    t2 = sTy #/ s
    
    u, s_, vh = np.linalg.svd(R, full_matrices=False)
    R_ortho = u.dot(vh)
    if np.linalg.det(R_ortho) < 0:
        R_ortho[2, :] = -R_ortho[2, :]
    R[...] = R_ortho[...]
    
    return s, R, [t1, t2]

