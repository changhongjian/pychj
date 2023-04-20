import subprocess

def nolog_precmd(ss=""):
    return f"ffmpeg -y -loglevel error {ss} "

def nolog_cmd(aug):
    cmd = f"{nolog_precmd()} {aug} "
    return subprocess.call([cmd], shell=True)
    
def cvt_wav(fout, fin, ar=16000):
    command = f"{nolog_precmd()} -i {fin} -vn -acodec pcm_s16le -ac 1 -ar {ar} {fout}"  
    return subprocess.call([command], shell=True)

def cvt_v2h264(fout, fin, brate, isbg=False):
    command = f"{nolog_precmd()} -i {fin} -c:v h264 -strict -2 -b:v {brate} {fout}"
    if isbg: command += " &"
    return subprocess.call([command], shell=True)

# fout, fin, times
def change_time(*arr):
    assert len(arr) in [3, 4]
    #assert 0.5=<arr[2]<=2
    vtm, atm = 1/arr[2], arr[2]
    aug=arr[3] if len(arr)==4 else ""
    cmd=f"-filter_complex '[0:v]setpts={vtm}*PTS[v];[0:a]atempo={atm}[a]' -map '[v]' -map '[a]' "
    cmd=f"-i {arr[1]} {cmd} {aug}  {arr[0]}"
    return nolog_cmd( cmd )

# https://blog.csdn.net/Gary__123456/article/details/88742705
def merge_video(fout, arr_f, tp="h", aug="-strict -2 -shortest"):
    assert isinstance(arr_f, list)
    file_cmd , n_f="", len(arr_f)
    for fnm in arr_f: file_cmd+=f"-i {fnm} "
    if tp=="h":
        assert n_f==2
        cmdtype="[0:v]pad=iw*2:ih*1[a];[a][1:v]overlay=w"
    elif tp=="v":
        assert n_f==2
        cmdtype="[0:v]pad=iw:ih*2[a];[a][1:v]overlay=0:h"
    elif tp=="3h":
        assert n_f==3
        cmdtype="[0:v]pad=iw*3:ih*1[a];[a][1:v]overlay=w[b];[b][2:v]overlay=2.0*w"
    elif tp=="3v":
        assert n_f==3
        cmdtype="[0:v]pad=iw:ih*3[a];[a][1:v]overlay=0:h[b];[b][2:v]overlay=0:2*h"

    else: exit("Error cmdtype")
    return nolog_cmd(f"{file_cmd} -filter_complex '{cmdtype}' {aug} {fout}")

def gen_by_video_wav(fav, fv, fa, aug="-c:v h264 -strict -2 -shortest"):
    cmd=f"-i {fv} -i {fa} -map 0:v -map 1:a -c:v h264 -strict -2 -shortest {fav}"
    nolog_cmd(cmd)

