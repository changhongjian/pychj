# -*- coding:utf-8 -* 

from chj.comm.include_full import *
from chj.math.geometry.lie_algebra import so3_exp
from chj.math.type_convert import npcoo_to_torch

import torch.nn.functional as thF

# CHJ_INFO: task related

def compute_so3_to_R_no_batch(so3):
    return so3_exp(so3.unsqueeze(0)).squeeze(0)

# batch
def compute_so3_to_R(so3):
    return so3_exp(so3)

# R=so3, batch
def compute_RT(R, T, V3d, isso3=True):
    if isso3: R = compute_so3_to_R(R)
    return V3d.bmm(R.transpose(1, 2)) + T.unsqueeze(-2).expand_as(V3d)


def compute_sRT_S(xes, V3d, isbatch=True, ratio_t=-1):
    if isbatch:
        n_b = xes.size(0)
    else:
        xes = xes.view(1, -1)
        V3d = V3d.view(1, -1, 3)
        n_b = 1

    # 我这里就直接使用弱透视投影了
    p_Rso3 = xes[:, :3]
    p_T = xes[:, 3:5]
    p_s = xes[:, 5:6]
    p_Rot3x3 = so3_exp(p_Rso3)

    sR_t = p_Rot3x3.transpose(2, 1) * p_s.view(-1,1,1).expand_as( p_Rot3x3 )

    V3d_n3 = V3d.view(n_b, -1, 3)
    Vc3d = V3d_n3.bmm(sR_t)

    if ratio_t>0:
        p_T = p_T * ratio_t

    Vc3d[:, :, :2] += p_T.unsqueeze(1)  # .expand(batch_num, nV, 2)

    return Vc3d

def sel_landmarks(Vs, ids):
    nb = Vs.size(0)
    lms = Vs.view(nb, -1, 3)[:, ids, :2].clone()
    return lms


# batch [ceofs1, ceof2], [mean, base1, base2 ]
# def compute_PCA(arr_xse, pca):
#     n_b = arr_xse[0].size(0)
#     nV3 = pca[0].shape[0]
#     y_mean = pca[0].expand(n_b, nV3).clone() #.contiguous()
#     for i in range( len(arr_xse) ):
#         #p( n_b )
#         #p( pca[i+1].unsqueeze(0).expand(n_b, nV3, -1).size() )
#         #p( y_mean.size() )
#         #p( arr_xse[i].unsqueeze(2).size() )
#         y_mean += pca[i+1].unsqueeze(0).expand(n_b, nV3, -1).bmm(arr_xse[i].unsqueeze(2)).squeeze(2)
#     return y_mean

def compute_PCA(arr_xse, pca, use_ids=None):
    n_b = arr_xse[0].size(0)
    nV = pca[0].view(-1, 3).size(0)
    pca_m = pca[0].view(-1)
    nV3 = pca_m.shape[0]

    if use_ids is not None:
        _ids1, _ids2 = use_ids
        pca_m = pca_m.view(nV, 3).unsqueeze(0).expand(n_b, nV, 3)[_ids1, _ids2]
        pca_m = pca_m.view(n_b, -1)
        new_pca=[]
        for i in range(1, len(pca)):
            E = pca[i].view(nV, 3, -1).unsqueeze(0).expand(n_b, nV, 3, -1)[_ids1, _ids2]
            dim = E.size(-1)
            new_pca.append( E.view(n_b, -1, dim) )

        y = 0
        for i in range(len(arr_xse)):
            y += new_pca[i].bmm(arr_xse[i].unsqueeze(-1)).squeeze(-1)
        return y + pca_m

    else:
        #y = pca[1].unsqueeze(0).expand(n_b, nV3, -1).bmm(arr_xse[0].unsqueeze(2)).squeeze(2)
        y=0
        for i in range(len(arr_xse)):
            #y += pca[i + 1].unsqueeze(0).expand(n_b, nV3, -1).bmm(arr_xse[i].unsqueeze(2)).squeeze(2)
            y += pca[i + 1].view(nV3, -1).unsqueeze(0).expand(n_b, nV3, -1).bmm(arr_xse[i].unsqueeze(2)).squeeze(2)
        return y + pca_m.unsqueeze(0).expand(n_b, nV3)



# F 是同一个F
def compute_batch_norm(V, thF, prec=1e-9):

    va = V[:,thF[:,0],:]
    vb = V[:,thF[:,1],:]
    vc = V[:,thF[:,2],:]
    ab = vb - va
    ac = vc - va
    Vn = ab.cross(ac) # compute normal for each face

    #thF = torch.from_numpy(F)
    # 大体就是[dim0, dim1, dim2], 然后 i, j, k 遍历,其中指明的那个dim 是 out[ i, id[i][j][k], k ] = in [i, j ,k]
    # 解释下中间的 thF, 就是要生成id, 然后内部的查询i,j,k不由其管
    _vn = V.new(V.size()).zero_()
    vn1 = _vn.clone().scatter_add_(1, thF[:, 0].view(1, -1, 1).repeat(V.size(0), 1, 3), Vn)
    vn2 = _vn.clone().scatter_add_(1, thF[:, 1].view(1, -1, 1).repeat(V.size(0), 1, 3), Vn)
    vn3 = _vn.clone().scatter_add_(1, thF[:, 2].view(1, -1, 1).repeat(V.size(0), 1, 3), Vn)
    vn = vn1 + vn2 + vn3

    norm = torch.sqrt(torch.clamp((vn * vn).sum(-1), min=prec))
    norm = norm.unsqueeze(-1).expand_as(vn)
    return vn / norm

#
def compute_SH(N):
    nb, nV, _ = N.size()
    _ones = torch.ones(nb, nV).float().cuda()
    nx = N[:, :, 0]
    ny = N[:, :, 1]
    nz = N[:, :, 2]

    arrH = []
    arrH.append(_ones)
    arrH.append(nx)
    arrH.append(ny)
    arrH.append(nz)
    arrH.append(nx * ny)
    arrH.append(nx * nz)
    arrH.append(ny * nz)
    arrH.append(nx.pow(2) - ny.pow(2))
    arrH.append(3 * nz.pow(2) - 1)

    #p( [ x.size() for x in arrH ]  )

    H = torch.stack(arrH, 2)
    return H

#nb, nV, 9;  nb, 9, 3 -> nb, nV, 3
#def compute_light(SH, illu): return SH.bmm(illu)

def compute_light(illu, Vs, facemodel):
    Ns = compute_batch_norm(Vs, facemodel)
    SH = compute_SH(Ns)
    #p( SH.size(), illu.size() )
    return SH.bmm(illu) # return

# must have the save size
def compute_tex(R, lighting):
    #p( R.size(), lighting.size() )
    return R*lighting

# 按照最后一维
def f_Normalize(x, prec = 1e-9):
    norm = x.pow(2).sum(-1).clamp(min=prec).sqrt()
    return x/norm.unsqueeze(-1).expand_as(x)


# light_phong
class LightPhong:
    def __init__(self, shiness = 20, ks=0.6):
        self.shiness = shiness
        #self.specularIntensity = 0.6
        # 我瞎写的
        self.Ks = ks
        self.Kd = 1
        self.Ka = 0.1
        self.Ke = 0 # emissive: 自发光物体有这个

    # https://www.cnblogs.com/bluebean/p/5299358.html
    # https://blog.csdn.net/Lyn_B/article/details/89852600
    # https://blog.csdn.net/goteet/article/details/8004192
    # 注意这个 direct 是指向眼睛的
    # batch 的和以前的不同
    def compute_phong_light(self, V, F, view_direct, light_p, light_c, amb_c):
        nb = V.size(0)
        N = compute_batch_norm(V, F)

        # specular: L-2N(N^t*L) 其中L = (Vn - lp)
        L = f_Normalize(V - light_p)

        #reflect_L = L - 2*N*N.view(-1, 1, 3).bmm(L.view(-1, 3, 1)).view(-1, 1).expand_as(N)
        reflect_L = L - 2 * N * (N*L).sum(-1).view(nb, -1, 1).expand_as(N)

        specularFactor = (reflect_L*view_direct).sum(-1).clamp(min=0.0).pow(self.shiness)
        specular = specularFactor.view(nb, -1, 1).bmm(light_c.view(nb, 1, 3)) * self.Ks

        # diffusion
        lightDirection = -L
        diffuseFactor = (N*lightDirection).sum(-1).clamp(min=0.0)
        diffuse = diffuseFactor.view(nb, -1, 1).bmm(light_c.view(nb,1, 3))

        # ambient
        ambient = amb_c.view(nb, 1, 3).expand_as(N)

        shading = ambient + diffuse + specular
        return shading

