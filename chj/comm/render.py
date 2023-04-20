
from chj.comm.include_full import th2np, np, np2th_gpu, torch

# nnet 中其实有个 render的

class cls_visual:
    def __init__(self):
        self.imgh = 0
        self.imgw = 0
        self.nb = 1
        self.isogl=True

    def set_fcdll(self, fcdll):
        from chj.speed.cdll import cls_speed
        speed = cls_speed()
        speed.load_cdll(fcdll)
        self.speed = speed

    def set_F(self, F):
        if type(F)==np.ndarray:
            F = np2th_gpu(F)
        F = F.int().cuda()
        self.nF =F.size(0)
        self.gpu_F = F
        self.speed.set_mp_torch("F", self.gpu_F)

    def set_render_info(self, imgh, imgw, nb=1):
        if nb == self.nb and imgh == self.imgh and imgw == self.imgw: return
        self.nb = nb
        self.imgh = imgh
        self.imgw = imgw

        self.img_Fid = torch.zeros((nb, imgh, imgw)).int().cuda()
        self.img_deep = torch.zeros((nb, imgh, imgw)).cuda()

    def run_render(self, TS):
        if TS.size(0)!=self.nb: 
            print("render TS must have batch: nb, TS.size(0)", self.nb, TS.size(0))
            exit()
        nV = TS.size(1)
        speed = self.speed
        self.dims = np.array([self.nb, self.nF], np.int32)
        dims = np.array([
            nV, self.nF, self.imgh, self.imgw
        ], np.int32)
        self.gpu_dims = torch.from_numpy(dims).cuda()

        self.img_Fid.fill_(-1)
        if self.isogl:
            self.img_deep.fill_(-1e9)
        else:
            self.img_deep.fill_(1e9)

        speed.set_mp("dims", self.dims)
        speed.set_mp_torch("gpu_dims", self.gpu_dims)
        speed.set_mp_torch("V", TS)
        speed.set_mp_torch("xy_tri_id", self.img_Fid)
        speed.set_mp_torch("deepmask", self.img_deep)
        if self.isogl:
            speed.cdll.D3F_batch_nF_render_info_gpu_v2() # 注意内部需要考虑 y 轴的问题
        else:
            speed.cdll.D3F_batch_nF_render_info_gpu_ocv()

    # batch
    def render_add_mask(self, imgs, org_rate=0.7, max_color=255):
        img_Fid = th2np(self.img_Fid)
        nb = img_Fid.shape[0]
        arr_newimg=[]
        for i in range(nb):
            # 只用计算个mask就行了
            # 继续改进
            img_fid = img_Fid[i]
            v_pix_ids = img_fid >= 0
    
            fmask=np.zeros(img_fid.shape, np.uint8)
            fmask[ v_pix_ids ] = 1
    
            nimg = imgs[i].copy()
            nimg[v_pix_ids] = imgs[i][v_pix_ids]*org_rate + max_color*(1-org_rate)

            arr_newimg.append(nimg)
    
        return arr_newimg
