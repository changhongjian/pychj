from chj.base.file import readlines, writelines 
import random
def split_train_val_test(dnm, fnm, n_test, n_val, 
        isshf=True, issort=False, isorder=False,
        nms=["train", "val", "test"]):
    if isinstance(fnm, str):
        lines=readlines(fnm) 
    else: lines=fnm
    if isshf: random.shuffle(lines) 
    n_tv = n_test+n_val
    n = len(lines)
    if isorder:
        dts = [lines[:-n_tv], lines[n-n_tv:n-n_test], lines[n-n_test:], ]
    else:
        dts = [lines[n_tv:], lines[n_test:n_tv], lines[:n_test], ]
    
    for nm, dt in zip(nms, dts):
        if issort: dt=sorted(dt)
        fnm=f"{dnm}/{nm}.list"
        writelines(fnm, dt)

