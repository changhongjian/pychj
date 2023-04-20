# -*- coding:utf-8 -* 


import numpy as np
import numpy
import math

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

#https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# 这个是按照 R=RzRyRx 来决定的
def rotationMatrixToEulerAngles(R):
    #assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def rot_Rodrigues2euler(rotVects):
    R = cv2.Rodrigues(rotVects)[0]
    return rotationMatrixToEulerAngles(R)


# 所谓的left 其实基本说明它的欧拉角是按照局部坐标系定义的
# 尝试了一下这个应该是没问题的，和lie_group也对应
# Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
def Euler2RotationMatrix(eulerAngles):
    rotation_matrix=np.zeros( (3,3),dtype=float )
    # 据说单个计算sin的话 numpy 比 math的快
    s1 = np.sin(eulerAngles[0])
    s2 = np.sin(eulerAngles[1])
    s3 = np.sin(eulerAngles[2])

    c1 = np.cos(eulerAngles[0])
    c2 = np.cos(eulerAngles[1])
    c3 = np.cos(eulerAngles[2])

    ''' 检查一下，是对的 matlab的有问题
    R_x = np.array([ [1,0,0],[0,c1,s1],[0,-s1,c1] ])
    R_y = np.array([ [ c2,0, s2],[ 0, 1, 0], [-s2,0, c2] ])
    R_z = np.array([ [c3, s3,0], [-s3, c3,0],[ 0, 0, 1] ])

    R = R_x.dot( R_y)
    R=R.dot(R_z)
    return R
    '''

    rotation_matrix[0, 0] = c2 * c3
    rotation_matrix[0, 1] = -c2 * s3
    rotation_matrix[0, 2] = s2
    rotation_matrix[1, 0] = c1 * s3 + c3 * s1 * s2
    rotation_matrix[1, 1] = c1 * c3 - s1 * s2 * s3
    rotation_matrix[1, 2] = -c2 * s1
    rotation_matrix[2, 0] = s1 * s3 - c1 * c3 * s2
    rotation_matrix[2, 1] = c3 * s1 + c1 * s2 * s3
    rotation_matrix[2, 2] = c1 * c2

    return rotation_matrix

def RotationMatrix2Euler(rotation_matrix):
    q0 = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2.0
    q1 = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4.0 * q0)
    q2 = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4.0 * q0)
    q3 = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4.0 * q0)

    #t1 = 2.0 * (q0 * q2 + q1 * q3)

    yaw = np.arcsin(2.0 * (q0 * q2 + q1 * q3))
    pitch = np.arctan2(2.0 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    roll = np.arctan2(2.0 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)

    return np.array([pitch, yaw, roll])    
    
# 也不知道谁有问题，反正这个对于李子青他们的那个是正确的    
def Euler2RotationMatrix_v2(eulerAngles):
    rotation_matrix=numpy.zeros( (3,3),dtype=float )
    # 据说单个计算sin的话 numpy 比 math的快
    s1 = np.sin(eulerAngles[0])
    s2 = np.sin(eulerAngles[1])
    s3 = np.sin(eulerAngles[2])

    c1 = np.cos(eulerAngles[0])
    c2 = np.cos(eulerAngles[1])
    c3 = np.cos(eulerAngles[2])

    #p(s1,s2,s3,c1,c2,c3)
    '''
    R_x = np.array([ [1,0,0],[0,c1,s1],[0,-s1,c1] ])
    R_y = np.array([ [ c2,0, -s2],[ 0, 1, 0], [s2,0, c2] ])
    R_z = np.array([ [c3, s3,0], [-s3, c3,0],[ 0, 0, 1] ])

    R = R_x.dot( R_y)
    R=R.dot(R_z)
    return R
    '''
    #http://www.eefocus.com/wdw1/blog/15-12/375573_5dea4.html
    rotation_matrix[0, 0] = c2 * c3
    rotation_matrix[0, 1] = c2 * s3
    rotation_matrix[0, 2] = -s2
    rotation_matrix[1, 0] = -c1 * s3 + c3 * s1 * s2
    rotation_matrix[1, 1] = c1 * c3 + s1 * s2 * s3
    rotation_matrix[1, 2] = c2 * s1
    rotation_matrix[2, 0] = s1 * s3 + c1 * c3 * s2
    rotation_matrix[2, 1] = -c3 * s1 + c1 * s2 * s3
    rotation_matrix[2, 2] = c1 * c2

    return rotation_matrix

#http://blog.csdn.net/lql0716/article/details/72597719

# 这里也可简化用不了那么麻烦
def RotationMatrix2Euler_v2(rotation_matrix):
    r23=rotation_matrix[1,2]
    r33=rotation_matrix[2,2]
    r13 = rotation_matrix[0, 2]
    r12=rotation_matrix[0, 1]
    r11=rotation_matrix[0, 0]
    pitch=np.arctan2( r23,r33 )
    yaw=np.arctan2(-r13,np.sqrt( r23**2+r33**2 ))
    roll=np.arctan2( r12,r11 )
    return np.array([pitch, yaw, roll])

    
    
def get_yaw_from_RotationMatrix(rotation_matrix):
    # 不写了，感觉不统一
    return None


# #2018-3-15 未整理
def quat_2_rvec(quat):
    #p(quat)
    theta_half = np.arccos(quat[-1])
    theta = 2*theta_half #w

    #p(theta,theta_half,np.sin(theta_half), "----")
    if(theta!=0.):

        axis = quat[0:3]/np.sin(theta_half)
        n=np.sum(axis*axis)
        #p(n, "----" )
        #assert ( np.abs( n-1 )<0.0001 )
    else:
        axis = [0, 0, 0]

    return theta, axis


