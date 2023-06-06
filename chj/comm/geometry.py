# -*- coding:utf-8 -* 

import re
import numpy 
import numpy as np
from chj.comm.pic import readlines, showimg, cv
from scipy.spatial.transform import Rotation as sciR

def save_obj_ids(fname, vtx, idx, face=None):
    vtx = vtx.reshape(-1, 3)

    num = len(vtx)
    with open(fname, "w") as fp:
        for i in range(num):
            vec = list(vtx[i])
            vec = ["%.3f" % x for x in vec]
            s = " ".join(vec)
            if i in idx: s += " 1 0 0"
            s = "v " + s + "\n"
            fp.write(s)

        if face is None: return
        for f in face:
            vec = list(f + 1)
            vec = ["%d" % x for x in vec]
            s = " ".join(vec)
            s = "f " + s + "\n"
            fp.write(s)

#@19-12-26 优化            
def save_obj(fname, vtx, face=None, tex=None, decimals=3, v_fmt=None):
    if v_fmt is not None: print("don't use v_fmt")
    vtx = vtx.reshape(-1, 3)
    if tex is not None:
        tex[tex < 0] = 0
        tex[tex > 1] = 1
        tex = tex.reshape(-1, 3)

    #num = vtx.shape[0]
    if tex is not None:
        vtx = np.hstack( (vtx, tex) )
    vtx=np.around(vtx, decimals=decimals).astype(str).tolist()
    with open(fname, "w") as fp:
        for vec in vtx:
            s = " ".join(vec)
            s = f"v {s}\n"
            fp.write(s)

        if face is None: return
        face=face+1
        face = face.astype(str).tolist()
        for f in face:
            s = " ".join(f)
            s = f"f {s}\n"
            fp.write(s)

def save_objs(fname, Vs, Fs=None, Ts=None, decimals=3):
    Vs = [ e.reshape(-1, 3) for e in Vs ]
    ns = [ e.shape[0] for e in Vs ]
    V = np.vstack(Vs)
    if Fs is not None:
        #print( len(Fs), len(ns) )
        for i in range(1, len(ns)): Fs[i]+=ns[i-1]
        F = np.vstack(Fs)
    else: F=None
    if Ts is not None:
        T = np.vstack(Ts)
    else: T=None
    save_obj(fname, V, F, T, decimals)

def get_nums_str(t):
    return " ".join( [ str(x) for x in t ] )

def show3d_points_diff_obj(fobj, T, S):
    T=T.reshape(-1,3)
    S = S.reshape(-1, 3)
    with open(fobj, "w") as fp:
        for t in T:
            ss="v "+get_nums_str(t)+" 0 1 0\n"
            fp.write(ss)
        for t in S:
            ss="v "+get_nums_str(t)+" 1 0 0\n"
            fp.write(ss)
    
def show3d_lm_points(fname, vtx, face, lms, color=[1, 0, 0]):
    vtx = vtx.reshape(-1, 3)
    num = len(vtx)
    with open(fname, "w") as fp:
        for i in range(num):
            vec = list(vtx[i])
            if i in lms: vec += color
            vec = ["%.3f" % x for x in vec]
            s = " ".join(vec)
            s = "v " + s + "\n"
            fp.write(s)

        for f in face:
            vec = list(f + 1)
            vec = ["%d" % x for x in vec]
            s = " ".join(vec)
            s = "f " + s + "\n"
            fp.write(s)

# advage by f
#vt will change
def get_vc_by_vt( F, ft, vt, fuv_img, vt_ogl=True):
    nV = F.max()+1
    if type(fuv_img) == str:
        uvimg = cv.imread(fuv_img)
    else: uvimg = fuv_img

    if vt_ogl: vt[:, 1] = 1 - vt[:, 1]
    vt[:, 0] *= (uvimg.shape[1]-1)
    vt[:, 1] *= (uvimg.shape[0]-1)

    vt = np.around(vt, 0).astype(np.int32)

    f_vt = vt[ ft.reshape(-1) ]
    _t = uvimg[ f_vt[:, 1], f_vt[:, 0], ::-1 ].astype(np.float32) / 255.0

    vc = np.zeros( (nV, 3), dtype=_t.dtype )
    np.add.at(vc, F.reshape(-1), _t)
    vc_cnt = np.zeros( (nV), dtype=np.int32 )
    np.add.at(vc_cnt, F.reshape(-1), 1)
    
    ids = vc_cnt>0
    vc[ids] /= vc_cnt[ids, None]
    return vc

def get_obj_raw_infos(fnm, hd_nms_str=""):
    hd_nms = hd_nms_str.split()
    lines=readlines(fnm)
    v=[]
    vc=[]
    vt=[]
    vn=[]
    fv=[]
    ft=[]
    fn=[]
    f=[]
    for line in lines:
        sz=line.split()
        n = len(sz)
        if n<3: continue
        if sz[0] == "v":
            v.append( sz[1:4] )
            if n>4: vc.append( sz[4:] )
        elif sz[0] == "vt":
            vt.append( sz[1:3] )
        elif sz[0] == "vn":
            vn.append( sz[1:4] )
        elif sz[0] == "f":
            f.append( sz[1:] )

    if len(vt)>0 or len(vn)>0:
        for vs in f:
            a= [ x.split("/") for x in vs ]
            fv.append( [ x[0] for x in a ] )
            if len(vt) > 0: ft.append( [ x[1] for x in a ] ) 
            if len(vn) > 0: fn.append( [ x[2] for x in a ] ) 
    else:
        fv = f

    if "v" in hd_nms:
        v = np.array(v, dtype=np.float32)
    if "vc" in hd_nms:
        vc = np.array(vc, dtype=np.float32)
    if "vt" in hd_nms:
        vt = np.array(vt, dtype=np.float32)
        if "ocv" in hd_nms:
            vt[:, 1] = 1 - vt[:, 1]
    if "vn" in hd_nms:
        vn = np.array(vn, dtype=np.float32)
        if "norm" in hd_nms:
            lens=np.linalg.norm(vn, axis=1)
            ids=lens>0
            vn[ids] /= lens[ids, None]

    if "fv" in hd_nms:
        fv = np.array(fv, dtype=np.int32) - 1
    if "ft" in hd_nms:
        ft = np.array(ft, dtype=np.int32) - 1
    if "fn" in hd_nms:
        fn = np.array(fn, dtype=np.int32) - 1

    class info: pass
    info.v=v
    info.vc=vc
    info.vt=vt
    info.vn=vn
    info.fv=fv
    info.ft=ft
    info.fn=fn
    info.nV = len(v)
    info.nF = len(fv)
    info.F = fv
    return info

#@2018-12-8            
def get_obj_v(fnm):
    lines=readlines(fnm)
    v=[]
    for line in lines:
        sz=line.split()
        
        if len(sz)>=4 and sz[0]=="v":
            v.append( [ float(x) for x in sz[1:4] ] )
        if len(sz)==4 and sz[0]=="f":
            break
    v = np.array( v ).astype(np.float32)
    
    #print(v.shape)
    return v    

def get_obj_v_f(fnm, only_f=False):
    lines=readlines(fnm)
    v=[]
    f=[]
    for line in lines:
        sz=line.split()
        
        if not only_f:
            if len(sz)>=4 and sz[0]=="v":
                v.append( [ float(x) for x in sz[1:4] ] )

        if len(sz)>=4 and sz[0]=="f":
            #f.append( [ int(x) for x in sz[1:4] ] )
            f.append( [ int(x) for x in sz[1:] ] )
    
    if not only_f:
        v = np.array( v ).astype(np.float32)
    f = np.array( f ).astype(np.int32) - 1 # 这个要注意
    if only_f: return f
    return v, f 

def get_obj_v_t_f(fnm):
    lines=readlines(fnm)
    v=[]
    t=[]
    f=[]
    for line in lines:
        sz=line.split()
        
        if len(sz)>=4 and sz[0]=="v":
            v.append( [ float(x) for x in sz[1:4] ] )
            t.append( [ float(x) for x in sz[4:7] ] )

        if len(sz)==4 and sz[0]=="f":
            f.append( [ int(x) for x in sz[1:4] ] )
            
    v = np.array( v ).astype(np.float32)
    t = np.array( t ).astype(np.float32)
    f = np.array( f ).astype(np.int32) - 1 # 这个要注意
    
    return v, t, f     


def get_obj_v_t(fnm):
    lines=readlines(fnm)
    v=[]
    t=[]
    for line in lines:
        sz=line.split()
        
        if len(sz)>=4 and sz[0]=="v":
            v.append( [ float(x) for x in sz[1:4] ] )
            t.append( [ float(x) for x in sz[4:7] ] )

        if len(sz)==4 and sz[0]=="f":
            break
            
    v = np.array( v ).astype(np.float32)
    t = np.array( t ).astype(np.float32)
    
    return v, t

# get vids by sub face
def get_subF_vids(fobj):
    F=get_obj_v_f("canonical_face_model.obj", only_f=True)
    return np.unique( F.reshape(-1) )

# @19-12-27 new faces by selected vids
def get_new_F_by_vids(F, vids):
    n1 = F.max() + 1
    n2 = len(vids)
    orgs = np.zeros(n1 , dtype=np.int32) - 3*n1
    orgs[vids] = np.arange(n2)
    f = orgs[F.reshape(-1)].reshape(F.shape[0], -1)
    f = f[ f.sum(axis=1) >= 0 ]
    return f
    
# @19-12-30
re_connection = get_new_F_by_vids
# the vertices are not remove, so the faces are from the org faces
def get_sub_F(F, vids, inv=False):
    n1 = F.max() + 1
    orgs = np.zeros(n1 , dtype=np.int32)
    if inv: orgs[vids] = -1
    else: 
        orgs -= 1
        orgs[vids] = 0
    f = F[ orgs[F.reshape(-1)].reshape(F.shape[0], -1).sum(axis=-1) == 0 ]
    return f

# @19-12-29
def get_img_F(h, w):
    tabley = np.arange(h).repeat(w).reshape(h, w).reshape(-1)
    tablex = np.arange(w).repeat(h).reshape(w, h).T.reshape(-1)
    x, y = [ e.reshape(h, w) for e in [tablex, tabley] ]
    vids = y * w + x
    F = np.hstack( ( vids[:-1, :-1].reshape(-1, 1),
                     vids[:-1, 1:].reshape(-1, 1),
                     vids[1:, 1:].reshape(-1, 1),
                     vids[1:, :-1].reshape(-1, 1),
                     ) )
    return F
    

# ids 是 numpy 会很快的
def check_sel_v(foutobj, finobj, ids, c_org=False, V=None):
    cnt=-1
    line_face=""
    with open(foutobj, "w") as fpo:
        with open(finobj) as fp:
            lines=fp.readlines()
            for i, line in enumerate(lines):
                #if (i+1)%1000==0: print(i, len(lines))
                if line[0]=='v':
                    cnt+=1
                    sz=line.split()
                    if not c_org: sz=sz[:4]
                    
                    line=" ".join(sz)
                    if cnt in ids:
                        line+=" 1 0 0"
                        #line=line.strip()+" 1 0 0\n"
                    #fpo.write(line+"\n")
                    line_face += line+"\n"
                elif line[0]=='f':
                    line_face += line
                    #fpo.write(line)
                if (i+1)%100 == 0: 
                    fpo.write(line_face)
                    line_face = ""

        fpo.write(line_face)

def check_sel_v_diff(foutobj, finobj, ids1, ids2, c_org=False, V=None):
    cnt=-1
    line_face=""
    with open(foutobj, "w") as fpo:
        with open(finobj) as fp:
            lines=fp.readlines()
            for i, line in enumerate(lines):
                if line[0]=='v':
                    cnt+=1
                    sz=line.split()
                    if not c_org: sz=sz[:4]
                    
                    line=" ".join(sz)
                    if cnt in ids1:
                        line+=" 0 1 0"
                    if cnt in ids2:
                        line+=" 1 0 0"
                    line_face += line+"\n"
                elif line[0]=='f':
                    line_face += line
                    #fpo.write(line)
        fpo.write(line_face)


def check_v_diff(foutobj, v1, v2):

    line_face = ""
    with open(foutobj, "w") as fpo:
        for v in v1:
            ss = [ str(e) for e in v ]
            line = "v "+" ".join(ss)
            line += " 0 1 0"
            line_face += line + "\n"

        for v in v2:
            ss = [str(e) for e in v]
            line = "v "+" ".join(ss)
            line += " 1 0 0"
            line_face += line + "\n"

        fpo.write(line_face)


def rough_check(V, imgw, imgh, img_org=None, is_return_img=False):
    V = V[V[:, 0] > 0]
    V = V[V[:, 1] > 0]
    V = V[V[:, 0] < imgw]
    V = V[V[:, 1] < imgh]
    V[:, 1] = imgh - V[:, 1]
    V = V.astype(np.int32)

    img = np.zeros((imgh, imgw, 3), np.uint8)
    if img_org is not None:
        img[V[:, 1], V[:, 0]] = img_org[V[:, 1], V[:, 0]]
    else:
        img[V[:, 1], V[:, 0], 1] = 255

    if is_return_img: return img
    else:
        key = showimg(img)
        # if key==27: break
        return key

#@18-12-27
# A中的索引，找到B中的索引, B 是 A 的子集
def find_idx_refA(B, A, Aidx=None):
    import scipy.spatial
    kdtree = scipy.spatial.cKDTree( B )
    if Aidx is None:
        dis, Bidx=kdtree.query( A )
    else:
        dis, Bidx=kdtree.query( A[Aidx] )
    return dis, Bidx
    
def find_sub_obj_index(fobjA, fobjB):
    vtgt = get_obj_v(fobjA)
    vsrc = get_obj_v(fobjB)
    import scipy.spatial
    kdtree = scipy.spatial.cKDTree( vtgt )
    dis, idx=kdtree.query( vsrc )
    return dis, idx


