# -*- coding:utf-8 -*

import numpy as np
import torch

def npcoo_to_torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col)).astype(np.int64)
    
    #ps(indices)
    #p(indices.dtype)
    #p(type(indices))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    
    #v = torch.from_numpy(values)
    #i = torch.from_numpy(indices)
    
    shape = coo.shape

    th_coo=torch.sparse.FloatTensor(i, v, torch.Size(shape))    
    return th_coo
