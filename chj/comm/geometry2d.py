import numpy as np

def get_rectxy_by_lm(lm):
    return np.hstack( ( lm.min(axis=0), lm.max(axis=0) ) )


def fix_lm_(lm, w, h):
    lm[ lm<=0 ] = 0
    lm[ lm[:, 0]>=w , 0] = w-1
    lm[ lm[:, 1]>=h , 1] = h-1
    return lm

def get_table_xy(w, h):
    tabley = np.arange(h).repeat(w).reshape(h, w).reshape(-1)
    tablex = np.arange(w).repeat(h).reshape(w, h).T.reshape(-1)
    return tablex, tabley
    #return np.vstack( (tablex, tabley) ).astype(np.int32)


    