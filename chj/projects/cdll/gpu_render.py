from chj.speed.cdll import cls_speed
from chj.math.geometry.points import bilinear_interpolate_numpy
from chj.comm.pic import *

class Cls_UV_render_gpu:
    def __init__(self): pass

    def init_render_TS(self, batch_size, fcdll, imgw, imgh, F, tp ):
        nV=F.max()+1
        nF=len(F)
        nb = batch_size
        self.dims=np.array( [nb, nF, tp], np.int32 )
        self.gpu_dims=np2th_gpu( np.array( [nV, nF, imgw, imgh ], np.int32 ) )
        self.deepmask=np2th_gpu( np.zeros( (nb, imgh, imgw ), np.float32 ) )
        self.xy_Fid=np2th_gpu( np.zeros( (nb, imgh, imgw  ), np.int32 ) )
        self.xy_bc=np2th_gpu( np.zeros( (nb, imgh, imgw , 3 ), np.float32 ) )
        if isinstance(fcdll, str):
            self.speed = cls_speed().load_cdll( fcdll )
        else: self.speed = fcdll 
        F = np2th_gpu(F)
        self.F=F
        # 小心如果cdll 被改写这样会有问题的
        mp={"dims": self.dims, "gpu_dims":self.gpu_dims, "F": F, "deepmask": self.deepmask, 
                "imgs_fid": self.xy_Fid, "imgs_b_c": self.xy_bc}
        self.speed.set_from_dict(mp)
        return self

    def run_render_get_info(self, TS, deepmask_val):
        self.deepmask.fill_(deepmask_val)
        self.speed.set_mp_torch("V", TS)
        self.speed.cdll.D3F_Render_B_F_pixinfo()
        self.deepmask_val = deepmask_val

    def get_valid_xyids(self):
        dm_val = self.deepmask_val
        isocv = self.isocv
        return self.deepmask < dm_val if isocv else self.deepmask > dm_val

    def calc_valid_Rs(self, R, valid_xyids):
        s_bcs = self.xy_bc[valid_xyids]
        s_fids = self.xy_Fid[valid_xyids]
        s_vids = self.F[s_fids.long()]
        s_vcs = R[ s_vids.view(-1).long() ].view(-1, 3, 3)
        s_pix = torch.einsum('nij,ni->nj', s_vcs, s_bcs)
        return s_pix

    # 渲染之后获得信息，计算原图的 uv
    def calc_valid_uvs(self, uv, valid_xyids):
        #if valid_xyids is None: valid_xyids = self.get_valid_xyids()
        
        s_bcs = self.xy_bc[valid_xyids]
        s_fids = self.xy_Fid[valid_xyids]
        s_vids = self.F[s_fids.long()]
        s_uvs = uv[ s_vids.view(-1).long() ].view(-1, 3, 2)
        s_uvs = torch.einsum('nij,ni->nj', s_uvs, s_bcs)
        return s_uvs

