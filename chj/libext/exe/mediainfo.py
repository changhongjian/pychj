#import subprocess
import os, sys
from chj.comm.pic import exitinfo
def get_media_info_ss(fnm):
    vs=get_media_info(fnm)
    ss=" | ".join(vs)
    return f"{ss}"

def _get_v(ss): return ss.strip().split(': ')[1]
def get_media_info(fnm):
    nms=["General", "Video", "Audio", "CHJ_INVALID"]
    r = os.popen(f"mediainfo {fnm}")
    flag=0
    vs=[]
    for line in r.readlines():
        #print(flag, line, len(vs))
        line=line.strip()
        if line == nms[flag]:
            flag+=1
            continue
        if flag==2:
            if 'Duration' in line:
                vs.append( _get_v(line) )
            elif 'Width' in line:
                vs.append( _get_v(line) )
            elif 'Height' in line:
                vs.append( _get_v(line) )
            elif 'Frame rate     ' in line:
                vs.append( _get_v(line) )
        elif flag==3:
            if 'Duration     ' in line:
                vs.append( _get_v(line) )
            elif 'Sampling rate' in line:
                vs.append( _get_v(line) )
    return vs


#print( sys.path, sys.argv )
# python -m chj.libext.exe.mediainfo fnm
if __name__=="__main__":
    if len(sys.argv)==0: 
        print("must giv a file")
        exit()
    fnm=sys.argv[1]
    if fnm[0]!='/': fnm=f"{sys.path[0]}/{fnm}"
    if not os.path.isfile(fnm): exitinfo(f"no such file: {fnm}") 
    print( get_media_info_ss(sys.argv[1]) )

