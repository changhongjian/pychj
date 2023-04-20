# -*- coding:utf-8 -* 

import re
import numpy 
import numpy as np
from .pic import readlines, showimg, cv
from scipy.spatial.transform import Rotation as sciR

from .geometry import *

def check_sel_v2(foutobj, finobj, ids, c_org=False, V=None):
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
                    else:
                        line+=" 1 1 1"
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

def check_sel_v_diff2(foutobj, finobj, ids1, ids2, c_org=False, V=None):
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
                    else:
                        line+=" 1 1 1"
                    line_face += line+"\n"
                elif line[0]=='f':
                    line_face += line
                    #fpo.write(line)
        fpo.write(line_face)


def check_v_diff2(foutobj, v1, v2):

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

