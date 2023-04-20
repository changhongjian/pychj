import multiprocessing
from chj.base.file import readlines
from tqdm import tqdm
def split_run(func, flist,  bg, ed=None ):
    if isinstance(bg, list):
        assert ed is None
        bg, ed = [ int(x) for x in bg ]

    if isinstance(flist, list):
        lines = flist
    else:
        lines = readlines(flist)
    if ed<0:
        nms = lines[bg:]
    else:
        nms = lines[bg:ed]

    idnm=f"{bg}-{ed}"

    return func(nms,  bg, idnm )

def get_n_cores(n_cores):
    if n_cores is None:
        n_cores = int( multiprocessing.cpu_count() )
    _tp = type(n_cores)
    if _tp == int:
        pass
    elif _tp == float:
        n_cores = int( multiprocessing.cpu_count() * n_cores )
    else:
        print("n_cores error:", n_cores)
        exit()
    
    return n_cores

def multi_run_map( func, arr_argvs, n_cores):
    n_cores = get_n_cores(n_cores)
    # start a pool
    pool = multiprocessing.Pool(processes=n_cores)
    #tasks = [ (lines[i], cls_face, i) for i in range(len(lines)) ]
    return pool.map(func, arr_argvs)

def multi_run_apply_async( func, arr_argvs, n_cores ):
    n_cores = get_n_cores(n_cores)
    pool = multiprocessing.Pool(processes=n_cores)
    res=[]
    for e in arr_argvs:
        res.append(pool.apply_async(func, args=tuple(e)))
    pool.close()
    pool.join()
    for i, e in enumerate(res): res[i] = e.get()
    return res

def multi_run_apply( func, arr_argvs, n_cores ):
    n_cores = get_n_cores(n_cores)
    pool = multiprocessing.Pool(processes=n_cores)
    res=[]
    for e in tqdm(arr_argvs):
        pool.apply(func, args=tuple(e))


