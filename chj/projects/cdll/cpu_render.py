from chj.speed.cdll import cls_speed
from chj.math.geometry.points import bilinear_interpolate_numpy
import numpy as np

class Cls_UV_render:
    def __init__(self): pass

    def init_render_TS(self, fcdll, imgw, imgh, F, isocv, tp=3 ):
        nV=F.max()+1
        nF=len(F)
        self.dims=np.array( [nV, nF, imgw, imgh, isocv, tp ], np.int32 )
        self.deepmask=np.zeros( ( imgh, imgw ), np.float32 ) 
        self.xy_Fid=np.zeros( ( imgh, imgw  ), np.int32 )
        self.xy_bc=np.zeros( ( imgh, imgw , 3 ), np.float32 )
        if isinstance(fcdll, str):
            self.speed = cls_speed().load_cdll( fcdll )
        else: self.speed = fcdll 
        self.F=F
        mp={"dims": self.dims, "F": F, "deepmask": self.deepmask, 
                "xy_tri_id": self.xy_Fid, "xy_tri_bc": self.xy_bc}
        self.speed.set_from_dict(mp)
        #exit()
        return self

    def run_render_get_info(self, TS, deepmask_val):
        self.deepmask.fill(deepmask_val)
        #print( self.deepmask.min(), self.deepmask.max())
        self.speed.set_mp("V", TS)
        self.speed.cdll.Render_get_info()
        #print( self.deepmask.min(), self.deepmask.max() )
        self.deepmask_val = deepmask_val

    def init_uv(self, uv, uvimg ):
        self.uvimg = uvimg
        self.uv = uv

    def get_valid_xyids(self):
        dm_val = self.deepmask_val
        isocv=self.dims[4]
        #print( self.dims, self.deepmask.min() )
        #exit()
        return self.deepmask < dm_val if isocv else self.deepmask > dm_val

    # 渲染之后获得信息，计算原图的 uv
    def calc_valid_uvs(self, uv, valid_xyids=None):
        if valid_xyids is None: valid_xyids = self.get_valid_xyids()
        s_bcs = self.xy_bc[valid_xyids]
        s_fids = self.xy_Fid[valid_xyids]
        s_vids = self.F[s_fids]
        s_uvs = uv[ s_vids.reshape(-1) ].reshape(-1, 3, 2)
        s_uvs = np.einsum('nij,ni->nj', s_uvs, s_bcs)
        return s_uvs

