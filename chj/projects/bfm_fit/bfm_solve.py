from . import transform
import numpy as np
import time

def get_TS(S, s, R, t2d):
    TS = S.dot( s*R.T )
    TS[:, :2] += t2d
    return TS

def fit_lm_v3(cls_bfm, t_lm, init_vids, pose_fit_lmids, n_it=5, wets=[1, 0.2 , -1], n_it_pose=3):

    init_vids = pose_fit_lmids 
    # step1: init
    vids = cls_bfm.ldmk_table.get_only_vids_by_y_p(0, 0)
    solve_sRt = transform.solve_Orthogonal3D_2D_sR_t2d
    solve_sRt_v2 = transform.solve_Orthogonal3D_2D_sR_t2d_v2
    y_lm = cls_bfm.get_S(None, None, vids=vids[init_vids])
    #y_lm = cls_bfm.get_S(None, None, vids=None)

    #save_obj("/tmp/chj/ck.obj", y_lm*1e5, cls_bfm.F)
    #exit()
    s, R, t2d = solve_sRt(y_lm, t_lm[init_vids])

    _dst = (t_lm - t2d) / s
    if hasattr(cls_bfm,'init_xse'):
        cls_bfm.pre_xs = cls_bfm.init_xse[0]
    vids, lm_wet = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, use_wet=True)  # 每次都要算个新的 R
    lm_wet = np.sqrt(lm_wet)
    #print(info)
    xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, vids=vids, isres_cmp_S=True, lm_wets=lm_wet)
    
    #print(s, t2d)
    _ids=pose_fit_lmids
    for i in range(n_it):
        for _ in range(n_it_pose):
            s, R, t2d = solve_sRt_v2(V[_ids], t_lm[_ids], lm_wet[_ids])
            #s, R, t2d = solve_sRt(V[_ids], t_lm[_ids])
            vids, lm_wet = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, use_wet=True)  # 每次都要算个新的 R
            V =  cls_bfm.get_S(xs, xe, vids=vids)
            lm_wet = np.sqrt(lm_wet)

        #print(s, t2d)
        _dst = (t_lm - t2d) / s
        cls_bfm.pre_xs = xs
        #vids, lm_wet = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, use_wet=True)  # 每次都要算个新的 R
        #xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, vids=vids, isres_cmp_S=True, lm_wets=lm_wet)

    
    return s, R, t2d, xs, xe, V, vids
   


def fit_lm_v2(cls_bfm, t_lm, init_vids, n_it=5, wets=[1, 0.2 , -1], is_slide_key=True, isres_cmp_S=False):
    '''
    先用部分点进行初始化, 然后再迭代求解
    :param cls_bfm: 需要提前初始化完毕
    :return:
    '''

    # step1: init
    vids = cls_bfm.ldmk_table.get_only_vids_by_y_p(0, 0)
    solve_sRt = transform.solve_Orthogonal3D_2D_sR_t2d
    solve_sRt_v2 = transform.solve_Orthogonal3D_2D_sR_t2d_v2
    y_lm = cls_bfm.get_S(None, None, vids=vids[init_vids])
    #y_lm = cls_bfm.get_S(None, None, vids=None)

    #save_obj("/tmp/chj/ck.obj", y_lm*1e5, cls_bfm.F)
    #exit()

    s, R, t2d = solve_sRt(y_lm, t_lm[init_vids])
    _dst = (t_lm - t2d) / s
    if hasattr(cls_bfm,'init_xse'):
        cls_bfm.pre_xs = cls_bfm.init_xse[0]
    if is_slide_key:
        vids, lm_wet = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, use_wet=True)  # 每次都要算个新的 R
        lm_wet = np.sqrt(lm_wet)
        #print(info)
        #bg = time.time()
        xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, vids=vids, isres_cmp_S=True, lm_wets=lm_wet)
        #ed = time.time()
        #print("solve one", ed - bg)
    else:
        xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, model_type=1, isres_cmp_S=True)

    #print(s, t2d)
    for i in range(n_it):
        s, R, t2d = solve_sRt_v2(V, t_lm, lm_wet)
        #print(s, t2d)
        _dst = (t_lm - t2d) / s
        cls_bfm.pre_xs = xs
        if is_slide_key:
            vids, lm_wet = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, use_wet=True)  # 每次都要算个新的 R
            lm_wet = np.sqrt(lm_wet)
            xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, vids=vids, isres_cmp_S=True, lm_wets=lm_wet)
        else:
            xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, model_type=1, isres_cmp_S=True)

    
    return s, R, t2d, xs, xe, V
   

# 获得系数
def fit_lm(cls_bfm, t_lm, init_vids, n_it=5, wets=[1, 0.2 , -1], is_slide_key=True, isres_cmp_S=False):
    '''
    先用部分点进行初始化, 然后再迭代求解
    :param cls_bfm: 需要提前初始化完毕
    :return:
    '''

    # step1: init
    vids = cls_bfm.ldmk_table.get_only_vids_by_y_p(0, 0)
    solve_sRt = transform.solve_Orthogonal3D_2D_sR_t2d
    solve_sRtv2 = transform.solve_Orthogonal3D_2D_sR_t2d_v2
    y_lm = cls_bfm.get_S(None, None, vids=vids[init_vids])
    #y_lm = cls_bfm.get_S(None, None, vids=None)

    #save_obj("/tmp/chj/ck.obj", y_lm*1e5, cls_bfm.F)
    #exit()

    s, R, t2d = solve_sRt(y_lm, t_lm[init_vids])
    _dst = (t_lm - t2d) / s
    if hasattr(cls_bfm,'init_xse'):
        cls_bfm.pre_xs = cls_bfm.init_xse[0]
    if is_slide_key:
        vids, lm_wet = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, use_wet=True)  # 每次都要算个新的 R
        #vids,_, info = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, isdbg=True)  # 每次都要算个新的 R
        #print(info)
        xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, vids=vids, isres_cmp_S=True, lm_wets=lm_wet)
    else:
        xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, model_type=1, isres_cmp_S=True)

    for i in range(n_it):
        s, R, t2d = solve_sRt(V, t_lm)
        _dst = (t_lm - t2d) / s
        cls_bfm.pre_xs = xs
        if is_slide_key:
            vids, lm_wet = cls_bfm.ldmk_table.get_only_vids_by_R(R, isocv=True, use_wet=True)  # 每次都要算个新的 R
            xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, vids=vids, isres_cmp_S=True, lm_wets=lm_wet)
        else:
            xs, xe, V = cls_bfm.solve_param(_dst, wets=wets, T_trans=R, model_type=1, isres_cmp_S=True)

    
    return s, R, t2d, xs, xe, V
   

