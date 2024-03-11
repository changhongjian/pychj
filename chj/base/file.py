# -*- coding:utf-8 -* 

'''
first 2018
'''

import codecs
import os
import numpy as np
import struct
import subprocess
import pickle, yaml, json
try:
    import zipfile
    from easydict import EasyDict as e_dict
except ImportError:
    pass

def chdir():
    #os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 
    os.chdir( os.path.dirname(os.path.realname( sys.argv[0] )) ) 

def readlines(fname, strip=True, encoding='utf-8'):
    with codecs.open(fname, encoding=encoding) as f:
        lists=f.readlines()
    if strip:
        lists = [x.strip() for x in lists]
    return lists

def writelines(fname, lines, encoding='utf-8'):
    with codecs.open(fname, "w", encoding=encoding) as f:
        if type(lines) is str:
            f.write(lines)
        else:
            n = len(lines)
            for i, line in enumerate(lines):
                if line is None: continue
                if type(line)!=str and len(line)>1:
                    arr = [ str(e) for e in line ]
                    arr=" ".join(arr)
                    f.write(arr)
                else: f.write(str(line))
                if i!=(n-1): f.write('\n')

def mkdir(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def mkdir_by_fnm(fnm):
    dnm, nm = os.path.split(fnm)
    mkdir(dnm)

def getdir_all_files(file_dir): 
    arr_files = []
    for root, dirs, files in os.walk(file_dir):  
        #print(root) #当前目录路径  
        #print(dirs) #当前路径下所有子目录  
        #print(files) #当前路径下所有非目录子文件

        ''' 只需要一个 '''
        for f in files:
            fnm = os.path.join(root,f)
            arr_files.append(fnm)
    return arr_files 

'''
id=read_bytes(fp,'L', 4)
pos = read_bytes(fp, '3f', 4*3)
'''
        
def read_bytes(fp,fmt,nbyte):
    bytes = fp.read(nbyte)
    res=struct.unpack(fmt, bytes)
    return res

def edict2dict(cfg): return json.loads(json.dumps(cfg))
def edict_update_adv(a, a_add):
    for k,v in a_add.items():
        #print(k, type(v))
        if type(v)==easydict.EasyDict and k in a:
            if '__overlap__' in v:
                del v['__overlap__']
                a[k] = v
            else:
                edict_update_adv(a[k], v)
        else:
            a[k] = v

#3-4

def make_temp_dir(fdir):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    else:
        files = os.listdir( fdir )
        if len(files) != 0:
            #subprocess.call( [ 'rm '+ fdir ] , shell=True)
            #subprocess.call( [ 'del '+ fdir ] , shell=True)
            os.system('rm -rf '+ fdir+"*")

'''
整理下，从 @3-8 开始不断改进并增加内容，感觉已经很完善了
'''

__mp_dtype={
    'int32': 0, 'float32':1, 'int64': 2, 'float64':3, 'bool':4, 'uint8':5
}
__mp_dtype_r={
    0:np.int32, 1:np.float32, 2:np.int64, 3:np.float64, 4:bool, 5:np.uint8
}

# 3-8 多次试验总结出来的    
def save_np_mat(fp, mt):
    dim=len(mt.shape)
    dims_and_type=np.array( [dim]+list(mt.shape)+[ __mp_dtype[str(mt.dtype)] ], np.int32)
    dims_and_type.tofile(fp)
    mt.tofile(fp)
    
    #print("save info: ", [dim]+list(mt.shape))
    
def save_np_mats(fp, mts):
    if type(fp) == str: fp = open(fp, "wb")
    mats_num=np.array( [len(mts)],  np.int32)
    mats_num.tofile(fp)
    for mt in mts: save_np_mat(fp, mt)

def load_np_mat(fp):
    #if type(fp)==str: fp=open(fp,"rb")
    dim = np.fromfile(fp, np.int32, 1)[0]
    dims = np.fromfile(fp, np.int32, dim)
    type_id=np.fromfile(fp, np.int32, 1)[0]
    dtype=__mp_dtype_r[type_id]
    mt = np.fromfile(fp, dtype, dims.prod())
    mt=mt.reshape(dims)
    #print("load info: ", dim, dims, " | ", mt.shape," | ", mt.dtype)
    return mt
    
def load_np_mats(fp):
    if type(fp)==str: fp=open(fp,"rb")
    mats_num = np.fromfile(fp, np.int32, 1)[0]
    mts=[]
    for i in range(mats_num): mts.append( load_np_mat(fp) )
    return mts
        
# 3-9
def save_np_as_txt(fp, fmt, mat):
    # fmt="%.5f"
    np.savetxt(fp, mat, fmt, delimiter=' ')

def _load_py(fp, func):
    if type(fp) == str:
        with open(fp, "rb") as _fp: return func(_fp)
    return func(fp)

#3-15
def save_pickle(fp,obj):
    if type(fp) == str: fp = open(fp, "wb")
    pickle.dump(obj, fp)

def load_pickle(fp):
    return _load_py(fp, pickle.load)

save_pkl=save_pickle
load_pkl=load_pickle

def load_yaml(fyaml, loader=yaml.FullLoader):
    if os.path.isfile(fyaml):
        with open(fyaml) as fp: return yaml.load(fp.read(), Loader=loader)
    else:
        return yaml.load(fyaml, Loader=loader)

def _save_yaml(fp, obj, tp=1):
    if tp ==1:
        from ruamel import yaml
        return yaml.dump(obj, fp, default_flow_style=False, allow_unicode = True, encoding = None)
    else:
        return yaml.dump(obj, fp)

def save_yaml(fp, obj, tp=1):
    if type(fp) == str: 
        with open(fp, "w") as _fp: return _save_yaml(fp,obj,tp) 
    else: return _save_yaml(fp,boj,tp) 

def load_json(fjson):
    if os.path.isfile(fjson):
        with open(fjson) as fp: return json.loads(fp.read())
    else:
        return json.loads(fjson)

def save_json(fp, obj, **kargs):
    if type(fp) == str: 
        with open(fp, "w") as _fp: return json.dump(obj, _fp, **kargs)
    return json.dump(obj, fp, **kargs)


#4-14
from scipy.sparse import csr_matrix

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def load_cfg_from_json(fjson):
    return e_dict( load_json(fjson) )

def load_cfg_from_yaml(fyaml, nofile_return='ERROR'):
    if nofile_return != 'ERROR' and not os.path.isfile(fyaml): return nofile_return 
    return e_dict( load_yaml(fyaml) )

def get_nm(fnm, has_path=0): 
    if has_path:
        return fnm.rsplit(".", 1)[0]
    else:
        return fnm.rsplit("/",1)[-1].rsplit(".",1)[0]

def save_zip(fzip, arr_nm_file, mode="w"):
    zip_file = zipfile.ZipFile(fzip,mode)
    for e in arr_nm_file:
        zip_file.write(e[0], arcname=e[1],compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()

