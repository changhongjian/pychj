import numpy as np
import torch

#   tha=np2th(exp)
#    tha = tha.t().unsqueeze(0)
#    sz = tha.shape
#    x = F.interpolate(tha, int( sz[2]/25*30 ) , mode='nearest')
#    x = x.squeeze(0).t()
#    exp30fps = th2np(x)

# B x C x D
def resize_seq_torch(a, tgt_size, isrvt=False, isres_np=False):
    if type(a)==np.ndarray:
        a = torch.from_numpy( a ).float()
    if len(a.shape)==2:
        a=a.unsqueeze(0)
        isuns=True
    else: isuns=False
    if isrvt: a = a.transpose(1,2)
    b=torch.nn.functional.interpolate(a, size=(tgt_size), mode="linear", align_corners=True)
    if isrvt: b = b.transpose(1,2)
    if isuns: b = b.squeeze(0)
    if isres_np: b = b.numpy()
    return b

def remake_item_num(arr, rate):
    src = dst = 0
    res=[]
    for i,e in enumerate(arr):
        src += e[1]
        n = int(src * rate - dst)
        dst+=n
        res.append( [ e[0], n ] )
    return res 

