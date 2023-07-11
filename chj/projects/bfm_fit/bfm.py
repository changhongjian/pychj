from .ldmk_table import *
import chj.base.file as chj_file
import os

# 主要是关于 bfm
class Cls_bfm_np:
    def __init__(self):
        self.use_c = False
        self.has_reg_val=False

    def init(self, fmodel, fcomm_vids=None):
        assert os.path.exists(fmodel)
        m = chj_file.load_np_mats(fmodel)
        self.m = m

        if fcomm_vids is None: return  self
        if isinstance(fcomm_vids, str): vids = np.load(fcomm_vids)
        else: vids = fcomm_vids

        idx = get_repeat_vids(vids)
        Vm = m[0][vids].copy()
        Es = m[1][idx].copy()
        Ee = m[2][idx].copy()
    
        # 这个106其实也是泛指
        self.lm106_m_comm = [Vm, Es, Ee]
        self.lm106_vids_comm = vids
        return self

    # 依据106点
    def init_5points(self, vids=[74, 77, 46, 84, 90]):
        if type(vids == list):
            vids = np.array(vids, np.int32)
        m = self.lm106_m_comm
        idx = get_repeat_vids(vids)
        Vm = m[0][vids]
        Es = m[1][idx]
        Ee = m[2][idx]

        self.lm5_m = [Vm, Es, Ee]
        self.lm5_ids = vids
        self.lm5_vids = self.lm106_vids_comm[vids]
        return self

    def init_ldmk_table(self, fldmk_table):
        self.ldmk_table = Cls_ldmk_table().init(fldmk_table)
        return self

    # 注意这个没有看明白代码不要轻易用, 因为这个是后来加
    def init_middle(self, fmodel_middle):
        m_middle = np.load(fmodel_middle)
        nV, dim_c = m_middle.shape
        self.use_c = True
        #m_c = np.hstack( (m_middle, m_middle, m_middle) ).reshape(-1,  dim_c)
        m_c = m_middle

        dim_s = self.m[1].shape[1]
        dim_e = self.m[2].shape[1]

        self.m_E_dims = [dim_s, dim_e, dim_c]
        self.m_has_c =[self.m[0], self.m[1], self.m[2], m_c]
        return self

    def init_reg_val(self, tp=0, vs=None, ve=None, vc=None, v_all=None, is_sd=True):
        self.has_reg_val = True
        arr_regs = [None, None, None]
        m = self.m
        dim_s = m[1].shape[1]
        dim_e = m[2].shape[1]

        if tp==1:
            if is_sd:
                v_all = np.power(v_all, 2)

            v_all = 1 / v_all
            vs = v_all[:dim_s]
            ve = v_all[dim_s: dim_s+dim_e]
            if self.use_c:
                #if v_all.shape[0] > (dim_e+dim_s):
                vc = v_all[dim_s + dim_e:]

        arr_regs[0] = vs
        arr_regs[1] = ve
        if self.use_c: arr_regs[2] = vc

        self.arr_regs = arr_regs

    def get_model_has_c(self, vids=None):
        m = self.m_has_c
        if vids is not None:
            m2 = [0, 0, 0, 0]
            m2[0] = m[0][vids]
            for i in range(3):
                dim = self.m_E_dims[i]
                if i == 2:
                    m2[1 + i] = m[1 + i].reshape(-1, dim)[vids].reshape(-1, dim)
                else:
                    m2[1+i] = m[1+i].reshape(-1, 3, dim)[vids].reshape(-1, dim)
            m = m2
        return m

    def get_model(self, model_type=None, vids=None):
        if model_type is  not None:
            if model_type == 0:
                return self.m
            if model_type == 1:
                return self.lm106_m_comm
            if model_type == 2:
                return self.lm5_m
        elif vids is not None:
            m = self.m
            m2 = [0, 0, 0]
            nV = m[0].shape[0]
            nV2 = len(vids)
            m2[0] = m[0][vids]
            m2[1] = m[1].reshape(nV, 3, -1)[vids].reshape(nV2 * 3, -1)
            m2[2] = m[2].reshape(nV, 3, -1)[vids].reshape(nV2 * 3, -1)
            return m2
        return self.m

    def get_S_has_c(self, xes, vids=None):
        m = self.get_model_has_c(vids)
        V = m[0].copy()
        for i, x in enumerate(xes):
            if x is not None:
                if i == 2:
                    V += m[i + 1].dot(x.reshape(-1, 3)) # 注意这里很特别
                else:
                    V += m[i+1].dot(x).reshape(-1, 3)
        return V

    def get_S(self, xs, xe, model_type=None, vids=None, base_V=None):
        m = self.get_model(model_type, vids)
        if base_V is None:
            V = m[0].copy()
        else: V = base_V
        if xs is not None:
            V += (m[1].dot(xs)).reshape(-1, 3)
        if xe is not None:
            V += (m[2].dot(xe)).reshape(-1, 3)

        return V

    def solve_param_has_c(self, dst, wets, T_trans=None, vids=None, isres_cmp_S=False, lm_wets=None):
        m = self.get_model_has_c(vids)

        if wets[3] > 0:
            Ec = np.kron(m[3], np.eye(3))
            m[3] = Ec

        nV, dim_data = dst.shape
        if T_trans is not None:
            Rt = T_trans[:dim_data].T
            m1 = m
            m = [None, None, None, None]
            # nV = len(m1[0])

            m[0] = m1[0].dot(Rt)

            for i in range(3):
                if wets[i+1]>0:
                    dim = self.m_E_dims[i]
                    if i==2: dim*=3
                    m[i+1] = np.zeros((dim_data * nV, dim), m[0].dtype)
                    for j in range(dim):
                        m[i+1][:, j] = m1[i+1][:, j].reshape(-1, 3).dot(Rt).reshape(-1)

        arr_reg=[]
        arr_E=[]
        for i in range(3):
            if wets[i + 1] > 0:
                dim = self.m_E_dims[i]
                if i == 2: dim *= 3
                arr_E.append(m[i+1])
                if self.has_reg_val:
                    _reg = self.arr_regs[i]
                    if i == 2: _reg = _reg.repeat(3)

                    arr_reg.append(_reg * wets[i + 1] / wets[0])
                else:
                    arr_reg.append(np.ones( dim, dtype=np.float32) * wets[i+1] / wets[0] )

        reg = np.hstack(arr_reg)
        W = np.hstack(arr_E)

        reg_b = np.zeros(reg.shape[0])
        reg = np.diag(reg)

        if lm_wets is None:
            A = np.concatenate((W, reg), 0)
            b = np.concatenate(((dst - m[0]).reshape(-1), reg_b))
        else:
            _, n2 = W.shape
            W = W.reshape((nV, -1)) * lm_wets[:, None]
            b1 = (dst - m[0]) * lm_wets[:, None]
            A = np.concatenate((W.reshape(-1, n2), reg), 0)
            b = np.concatenate((b1.reshape(-1), reg_b))

        x = scipy.linalg.lstsq(A, b, rcond=None)[0]

        arr_x =[None] * 3
        _cnt=0
        for i in range(3):
            if wets[i + 1] > 0:
                dim = self.m_E_dims[i]
                if i == 2: dim *= 3

                arr_x[i] = x[_cnt:_cnt+dim].astype(m[0].dtype)
                _cnt+=dim

        V = None
        if isres_cmp_S: V = self.get_S_has_c(arr_x, vids)

        return arr_x, V


    # 其实对于这种线性求解的还可以自己写一个, 这样使得矩阵小些, 求解速度会快些
    # model_type 和 vids 二选一, T_trans
    # 改成了 2d, 3d 都支持的形式了
    def solve_param(self, dst, wets, model_type=None, T_trans=None, vids=None,  isres_cmp_S=False, lm_wets=None):
        
        m = self.get_model(model_type, vids)
        
        dim_s = m[1].shape[-1]
        dim_e = m[2].shape[-1]

        nV, dim_data = dst.shape

        use_e = wets[2] > 0
        if T_trans is not None:
            #assert len(T_trans)==2 and T_trans.shape[0] == T_trans.shape[1]
            Rt = T_trans[:dim_data].T
            m1 = m
            m = [None, None, None]
            #nV = len(m1[0])

            m[0] = m1[0].dot(Rt)

            m[1] = np.zeros((dim_data * nV, dim_s), m[0].dtype)
            for i in range(dim_s):
                m[1][:, i] = m1[1][:, i].reshape(-1, 3).dot(Rt).reshape(-1)

            if use_e:
                m[2] = np.zeros((dim_data * nV, dim_e), m[0].dtype)
                for i in range(dim_e):
                    m[2][:, i] = m1[2][:, i].reshape(-1, 3).dot(Rt).reshape(-1)


        if use_e:
            W = np.concatenate((m[1], m[2]), 1)
            if self.has_reg_val:
                reg = np.hstack((self.arr_regs[0], self.arr_regs[1]))
            else: reg = np.ones(dim_s + dim_e)

            reg[dim_s:] *= wets[2] / wets[0]
        else:
            W = m[1]  # np.concatenate( (m[1]), 1 )
            if self.has_reg_val:
                reg = self.arr_regs[0].copy()
            else: reg = np.ones(dim_s)
            xe = None

        reg[:dim_s] *= wets[1] / wets[0]

        #reg_b = np.zeros(reg.shape[0])
        #reg = np.diag(reg)

        #if lm_wets is None:
        #    A = np.concatenate((W, reg), 0)
        #    b = np.concatenate(((dst - m[0]).reshape(-1), reg_b))
        #else:
        #    _, n2 = W.shape
        #    A1=W.reshape((nV, -1)) * lm_wets[:, None]
        #    b1 = (dst - m[0]) * lm_wets[:, None]
        #    A = np.concatenate((A1.reshape(-1, n2), reg), 0)
        #    b = np.concatenate((b1.reshape(-1), reg_b))
        
       
        # the first is too slow
        #import time
        #bg=time.time()
        #x = np.linalg.lstsq(A, b, rcond=None)[0]
        #np.save("/tmp/chj/a.npy", x)
        #exit()
        
        #ed = time.time()
        #print("lstsq time1", ed-bg)

        #bg=time.time()
        #x2 = np.linalg.solve(A.T.dot(A), A.T.dot(b) ) 
        #ed = time.time()
        #print("lstsq time2", ed-bg)
        #print(np.allclose(x, x2))


        if lm_wets is None:
            lm_wets = np.zeros( m[0].shape[0], m[0].dtype ) + 1/m[0].shape[0]

        n1, n2 = W.shape
        b = ( (dst - m[0]) * lm_wets[:, None] ).reshape(-1)
        A = (W.reshape((nV, -1)) * lm_wets[:, None]).reshape(-1, n2)

        Atb = A.T.dot(b)
        AtA = A.T.dot(A)
        AtA[ np.diag_indices_from(AtA) ] += reg**2

        x = scipy.linalg.solve( AtA, Atb )
        #np.save("/tmp/chj/b.npy", x)
        #exit() 
        xs = x[:dim_s].astype(m[0].dtype)

        if use_e:
            xe = x[dim_s:].astype(m[0].dtype)

        V = None
        if isres_cmp_S: V = self.get_S(xs, xe, model_type, vids)

        return xs, xe, V


def get_repeat_vids(sel_ids):
    sel_ids = (sel_ids * 3).repeat(3).reshape(-1, 3)
    sel_ids[:, 1] += 1
    sel_ids[:, 2] += 2
    sel_ids = sel_ids.reshape(-1)
    return sel_ids


def auto_S(S, wh, rate=0.8, M=None):
    '''
    transform=[s, 0, tx; 0, s, ty]
    '''

    a = S[:, :2].min(axis=0)
    b = S[:, :2].max(axis=0)
    c = (a+b)/2
    d = (b-a).max()

    if type(wh) == int:
        # get s
        s = wh * rate / d
        # get tx, ty according to the center point
        c_t = np.array([wh, wh]) / 2
    else:
        s = min(wh) * rate / d
        c_t = np.array(wh) / 2
    txy = c_t - s*c
    TS = S * s
    TS[:, :2] += txy #[np.newaxis]
    if M is not None:
        M[0] = s 
        M[1] = txy 
    return TS
