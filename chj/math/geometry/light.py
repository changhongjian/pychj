
import numpy as np

PI=np.pi
sqrt=np.sqrt
'''  这里其实可以优化，有很多常数的计算 '''
def HarmonicBasis_4(basis, xyz):
    x, y, z = xyz
    basis[0] = 0.5 / sqrt(PI)
    basis[1] = 0.5 * sqrt(3.0 / PI) * y
    basis[2] = 0.5 * sqrt(3.0 / PI) * z
    basis[3] = 0.5 * sqrt(3.0 / PI) * x


def HarmonicBasis_9(basis, xyz):
    x, y, z = xyz
    HarmonicBasis_4(basis, xyz)

    basis[4] = 1.0 / 2.0 * sqrt(15.0 / PI) * x * y
    basis[5] = 1.0 / 2.0 * sqrt(15.0 / PI) * y * z
    basis[6] = 1.0 / 4.0 * sqrt(5.0 / PI) * (-x * x - y * y + 2 * z * z)
    basis[7] = 1.0 / 2.0 * sqrt(15.0 / PI) * z * x
    basis[8] = 1.0 / 4.0 * sqrt(15.0 / PI) * (x * x - y * y)

def HarmonicBasis_16(basis, xyz):
    x, y, z = xyz
    HarmonicBasis_9(basis, xyz)

    basis[9] = 1.0 / 4.0 * sqrt(7.0 / PI) * z*(2*z*z-3*x*x-3*y*y * y)
    basis[10] = 1.0 / 4.0 * sqrt(21.0 / 2 / PI) * x*(5*z*z-1)
    basis[11] = 1.0 / 4.0 * sqrt(21.0 / 2 / PI) * y*(5*z*z-1)
    basis[12] = 1.0 / 4.0 * sqrt(105.0/PI) * z*(x*x-y*y)
    basis[13] = 1.0 / 2.0 * sqrt(105.0/PI) * x*y*z
    basis[14] = 1.0 / 4.0 * sqrt(35.0/2/PI) * x*(x*x-3*y*y)
    basis[15] = 1.0 / 4.0 * sqrt(35.0/2/PI) * y*(3*x*x-y*y)
    

# 这些 vs 在我这里是法向量
def mesh_Harmonic( vs, dim=9):
    HarmonicBasis=None
    if dim==9: HarmonicBasis = HarmonicBasis_9
    elif dim==4: HarmonicBasis = HarmonicBasis_4
    elif dim==16: HarmonicBasis = HarmonicBasis_16
    else: exit(" mesh_Harmonic dim error ")
    basises=np.zeros( (dim, vs.shape[0]) )
    HarmonicBasis( basises, (vs[:, 0], vs[:, 1], vs[:, 2]) )
    return basises.T

# 法向量
def mesh_Harmonic_RGB(vs, dim=9):
    basises = mesh_Harmonic( vs, dim)

    basises_rgb = np.zeros((vs.shape[0], dim*3*3))

    basises_rgb[:, 0:dim]=basises
    basises_rgb[:, dim*4:dim*4+dim] = basises
    basises_rgb[:, dim * 8:dim * 8 + dim] = basises

    basises_rgb = basises_rgb.reshape( -1, 3*dim )
    return basises_rgb

# t_type, 0: np, 1: th, 2:th_gpu
def mesh_SH_basis(vn, is_rgb=1, dim=9):
    if dim==9: HarmonicBasis = HarmonicBasis_9
    elif dim==4: HarmonicBasis = HarmonicBasis_4
    elif dim==16: HarmonicBasis = HarmonicBasis_16
    else: exit(" mesh_Harmonic dim error ")
    
    a_dim = dim*3*3 if is_rgb else dim

    if type(vn)==np.ndarray:
        basises=np.zeros( (a_dim, vn.shape[0]), dtype=vn.dtype )
    else:
        import torch
        basises=torch.zeros( (a_dim, vn.size(0)), device=vn.device, dtype=vn.dtype )
    
    HarmonicBasis( basises[0:dim], ( vn[:, 0], vn[:, 1], vn[:, 2] ) )

    if is_rgb:
        basises[dim*4:dim*4+dim] = basises[0:dim]
        basises[dim*8:dim*8+dim] = basises[0:dim]
        basises =  basises.T.reshape(-1, 3*dim)
        
        return basises
    else:    
        return basises.T
    

# 后面我自己实现了一个 pytorch 的
# 参考pjt的　c++
class LightPong:

    def __init__(self):

        self.Amb = np.array([.5, .5, .5]) * 255
        self.Dif = np.array([.5, .5, .5]) * 255
        self.Ks = 0 * 256
        self.ns = 20
        self.light = np.array([0, 0, -1])
        
        pass

    # 需要传入顶点即可
    def get_light_no_tex(self, vn):
        vn = vn.reshape(-1, 3)
        nV, _ = vn.shape

        I = vn.copy()
        
        light = self.light
        Amb = self.Amb
        Dif = self.Dif
        
        view = np.array([0, 0, 1])
        for i in range(nV):
            dot = light.dot(vn[i]) 
            reflect = np.array([
                2 * dot * vn[i][0] - light[0],
                2 * dot * vn[i][1] - light[1],
                2 * dot * vn[i][2] - light[2]])
 
            dotr = reflect.dot(view)
            spec = 0 if dotr <= 0 else Ks * np.power(dotr, ns) 
            dot = 0 if dot <= 0 else dot
            
            I[3 * i]     = Amb[0] + Dif[0] * (dot + spec);
            I[3 * i + 1] = Amb[1] + Dif[1] * (dot + spec);
            I[3 * i + 2] = Amb[2] + Dif[2] * (dot + spec);
   
        return I

