# -*- coding:utf-8 -*

from scipy.linalg import solve
import numpy
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve

# Ax=b
def solve_minsqure_Axb(A,b, W=None):
    At = A.transpose()
    if W is not None: At=At.dot(W)
    left = At.dot(A)
    right = At.dot(b)
    x = solve(left, right)
    return x

# || b - Ax ||^2_2 + w*|| x ||_sigma2 the ||^2 x ||_sigma2  is sqrt( xt*diag(sigma2)*x )
# (AtA + w*diage(sigma2))x = Atb
def solve_minsqure_Axb_and_regularize(A,b, w=0, sigma=None):
    if w==0: return solve_minsqure_Axb(A,b)
    At = A.transpose()
    left = At.dot(A)
    left_add=0
    if sigma is None: left_add = np.eye( len(At) )
    #else: left += w * np.diag( sigma )
    else: left_add =  sigma  # 矩阵，如果是对角阵则自己制作
    if w!=1: left_add *= w
    left += left_add
    right = At.dot(b)
    #p(np.linalg.norm(left), np.linalg.norm(right))
    x = solve(left, right)
    return x

# 其实就是上面那个的简化版本
def solve_minsqure_Axb_like_lm(A,b, W):
    At = A.transpose()
    left = At.dot(A) + W
    right = At.dot(b)
    return solve(left, right)


# || b - Ax ||^2_C + w*|| x ||_sigma2 the || x ||_sigma2  is sqrt( xt*diag(sigma2)*x )
# (AtA + w*diage(sigma2))x = Atb
def solve_minsqure_Axb_with_weight(A,b, Weight, w=0, sigma=None):
    if w==0: return solve_minsqure_Axb(A,b, Weight)
    At = A.transpose()
    if W is not None: At = At.dot(W)
    left = At.dot(A)
    if sigma is None: left += w * np.eye( len(At) )
    else: left += w * sigma
    right = At.dot(b)
    #p(np.linalg.norm(left), np.linalg.norm(right))
    x = solve(left, right)
    return x

