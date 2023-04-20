import numpy as np

# create nxn mat idx
def get_cross_idx(n):
    x=np.arange(n)
    y=np.arange(n)
    a, b = [ e.reshape(-1, 1).astype(np.int32) for e in  np.meshgrid(x,y) ]
    ids=np.hstack( (b,a) )
    return ids

def split_n(n, neach):
    c = (n-1) // neach + 1
    arr = np.zeros( ( c, 2 ), np.int32 )
    arr[:, 0] = np.arange(0, n, neach)
    arr[:-1, 1] = arr[1:, 0]
    arr[-1, 1] = n
    return arr


def distance_mat_toarray_n(n): return n*(n-1)//2
def distance_mat_toarray_id(n,i,j):
    if i==j: return -1
    if i>j: i,j=j,i
    return ((n-1)+(n-i))*i//2+(j-i) - 1 # 1-base 2 0-base


