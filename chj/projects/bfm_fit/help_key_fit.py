import cv2
import numpy as np
from . import transform

class Cls_help_fit:
    def __init__(self, bfm):
        self.bfm = bfm
        self.dim_s = bfm.m[1].shape[-1]
        self.dim_e = bfm.m[2].shape[-1]
    
    #r = sciR.from_rotvec(so3).as_quat()
    def get_lm2d(self, x):
        lm = self.bfm.get_S(x[6:6+self.dim_s], x[6+self.dim_s:], 1)
        so3, t2d, s = x[:3], x[3:5], x[5]
        R = cv2.Rodrigues(so3)[0] 
        lm = transform.transform_Similarity3D_sR_t2d( lm, s*R, t2d)
        return lm, R

    def fit_key_f(self, arr_pm, arr_t_lm, n_it=5, wets=[1, 0.2, -1], arr_xs=None):
        x_s = fit_arr_t_lm( self.bfm, arr_t_lm, arr_pm, n_it, wets, arr_xs )
        return x_s

    def cmp_S(self, x_s):
        return self.bfm.get_S(x_s, None, model_type=None)

def ck_fit_res( cls_bfm, x_s, x_e, sR, t2d, t_lm, a ):
    lm = cls_bfm.get_S(x_s, x_e, 1)
    lm = transform.transform_Similarity3D_sR_t2d( lm, sR, t2d)[:, :2]
    return np.power( (t_lm - lm).reshape(-1), 2 ).sum() + a*np.power(x_s, 2).sum()

    
# wets, 包括 reg_s, e, c 
def fit_arr_t_lm(cls_bfm, arr_t_lm, init_pm, n_it=5, wets=[1, 0.2, -1], arr_xs=None):
    solve_sRt = transform.solve_Orthogonal3D_2D_sR_t2d
    x_s = None
    #arr_t_lm = arr_t_lm[:, 32:]
    #m_org = cls_bfm.get_model(model_type=None, vids=cls_bfm.lm106_vids_comm[32:])
    m_org = cls_bfm.get_model(1)
    dim_s = m_org[1].shape[-1]
    dim_e = m_org[2].shape[-1]


    n_len, nV, _dim = arr_t_lm.shape
    assert _dim==2

    arr_pm = [ init_pm[0], init_pm[1], init_pm[-1] ]
    for i in range(n_len):
        arr_pm[0][i] *= init_pm[2][i]

    m_meanS = m_org[0].reshape(-1)
    m_As = m_org[1].reshape(-1, dim_s)
    m_Ae = m_org[2].reshape(-1, dim_e)
    for i in range(n_it):
        # 因为有init_pm 先集体求一个 shape -> 先算出lm2d无关, 然后 对 lm 取平均
        mean_tlm = 0
        arr_infos = [0, 0, 0]
        for j in range(n_len):
            R2r = arr_pm[0][j, :2]
            lm = arr_t_lm[j] - arr_pm[1][j] 
            _V = ( m_meanS + m_org[2].dot( arr_pm[2][j] ) ).reshape(-1, 3).dot(R2r.T)
            lm -= _V
            #lm3d = lm.dot( np.linalg.pinv(R2r).T ).reshape(-1)
            #lm3d -= m_org[2].dot( arr_pm[2][j] ) - m_meanS
            #arr_t_lm3d.append( lm3d )

            #p( np.allclose( get_bfmA_by_R( m_org[1], R2r ), get_bfmA_by_R_2( m_org[1], R2r )  ) )
            #exit()

            A = get_bfmA_by_R( m_org[1], R2r )
            b = lm.reshape(-1)
            AtA = A.T.dot(A)
            Atb = A.T.dot(b)
            arr_infos[0] += Atb / n_len
            arr_infos[1] += AtA / n_len

            #x_s = solve_reg_At_b(A, b, wets[0])
            #res1 =  ck_fit_res(cls_bfm, x_s, arr_pm[2][j], arr_pm[0][j], arr_pm[1][j], arr_t_lm[j], wets[0])
            #x_s = arr_xs[j]
            #res2 =  ck_fit_res(cls_bfm, x_s, arr_pm[2][j], arr_pm[0][j], arr_pm[1][j], arr_t_lm[j], wets[0])
            #p(res1, res2)

        
            #arr_infos[2] += solve_reg_squa_A_b(AtA, Atb, wets[0]) / n_len
            #arr_infos[2] += solve_reg_At_b(AtA, Atb, wets[0]) / n_len
        #mean_tlm = np.array(arr_t_lm3d).mean(axis=0)
        #x_s = solve_reg_At_b(m_As, mean_tlm, wets[0])
        
        # 验证是一样的
        x_s = solve_reg_squa_A_b(arr_infos[1], arr_infos[0], wets[0])
        #x_s = arr_infos[2]
        #return x_s
        #
        lm3d_x_s = m_meanS + m_As.dot(x_s)

        # 先求pose, 再求 x_e
        for j in range(n_len):
            t_lm = arr_t_lm[j]
            V = lm3d_x_s + m_Ae.dot( arr_pm[2][j] )
            s, R, t2d = solve_sRt(V.reshape(nV, 3), t_lm)
            
            arr_pm[0][j] = s*R
            arr_pm[1][j] = t2d
            _dst = (t_lm - t2d) / s

            #lm3d = lm.dot( np.linalg.pinv(R[:2]).T ).reshape(-1)
            #lm3d -= lm3d_x_s
            #x_e = solve_reg_At_b(m_Ae, lm3d, wets[1])
            R2r = arr_pm[0][j, :2]
            lm = arr_t_lm[j] - arr_pm[1][j] 
            lm -= lm3d_x_s.reshape(-1, 3).dot(R2r.T)
            A = get_bfmA_by_R( m_org[2], R2r )
            AtA = A.T.dot(A)
            Atb = A.T.dot(lm.reshape(-1))
            x_e = solve_reg_squa_A_b(AtA, Atb, wets[1])
            arr_pm[2][j] = x_e

    return x_s

def get_bfmA_by_R_2(A, R):
    dim = A.shape[-1]
    A = A.reshape(-1, 3, dim)
    B=A[:, :2].copy()
    for i in range(dim):
        B[:, :, i] = A[:, :, i].dot(R.T)
    return B.reshape(-1, dim)

def get_bfmA_by_R(A, R):
    E = A.reshape(-1, 3, A.shape[-1]).transpose(1, 0, 2)
    return R.dot(E.reshape(3, -1)).reshape(-1, E.shape[1], A.shape[-1]).transpose(1, 0, 2).reshape(-1, A.shape[-1])

def solve_reg_squa_A_b(A, b, reg):
    A[ np.diag_indices_from(A) ] += reg
    x = np.linalg.inv(A).dot(b)
    return x

# 两个是一样的
def solve_reg_At_b_(A, b, reg):
    reg_b = np.zeros(A.shape[-1])
    reg = np.eye(A.shape[-1]) * reg

    A = np.concatenate((A, reg), 0)
    b = np.concatenate((b, reg_b))
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x

def solve_reg_At_b(A, b, reg):
    AtA = A.T.dot(A)
    Atb = A.T.dot(b)
    AtA[ np.diag_indices_from(AtA) ] += reg
    x = np.linalg.inv(AtA).dot(Atb)
    return x

def solve_arr_reg_At_b(arr_A, arr_b, reg):
    assert len(reg) == len(arr_A)+1
    AtA, Atb = 0, 0
    for i in range( len(reg)-1 ):
        #ps( arr_A[i] )
        #ps( arr_b[i] )
        AtA += arr_A[i].T.dot( arr_A[i] ) * reg[i]
        Atb += arr_A[i].T.dot( arr_b[i] ) * reg[i]
    AtA[ np.diag_indices_from(AtA) ] += reg[-1]
    x = np.linalg.inv(AtA).dot(Atb)
    return x

def solve_arr_At_b(arr_A, arr_b, wets):
    AtA, Atb = 0, 0
    for i in range( len(arr_A) ):
        AtA += arr_A[i].T.dot( arr_A[i] ) * wets[i]
        Atb += arr_A[i].T.dot( arr_b[i] ) * wets[i]
    #p(AtA)
    #p(Atb)
    x = np.linalg.inv(AtA).dot(Atb)
    return x

