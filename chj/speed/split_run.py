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


def torch_spawn_run( func, arr_argvs, n_cores ):
    import torch.multiprocessing as mp
    
    arr=split_list( arr_argvs, n_cores )
    mp.spawn(func, args=(arr, n_cores), nprocs=njobs, join=True)

def split_list(input_list, n):
    avg = len(input_list) // n
    remainder = len(input_list) % n
    result = []
    start = 0

    for i in range(n):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg

        result.append(input_list[start:end])
        start = end

    return result

def multi_run_onbatch( process_chunk, data, chunk_size, n_cores ):
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with Pool(processes=n_cores) as pool:  # 这里使用4个进程，您可以根据需要调整
        pool.map(process_chunk, chunks)

    
