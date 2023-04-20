import  cv2 
import  cv2 as cv 
import os
import numpy as np
import imageio

try:
    from moviepy.editor import VideoFileClip
except:
    pass

def get_video_times(video_path):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps == 0:
        print( video_path )
    return frame_count / fps
    # too slow
    #video_clip = VideoFileClip(video_path)
    #durantion = video_clip.duration
    #return float( durantion )

def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

def get_cap_info(cap):
    # w, h, 
    arr=[ 3, 4, 5, 7  ]
    arr= [ int( cap.get(x) ) for x in arr ]
    #for i in [0, 1, 2, 4] : arr[i]=int(arr[i]) 
    return arr + [ decode_fourcc( cap.get(6) ) ]

def get_video_infos(video_path):
    vid_cap = cv2.VideoCapture(video_path)
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {'height': height, 'width': width, 'fps': fps, 'nframes':total_frames}

class Cls_put_image:
    def __init__(self, xywh):
        self.xywh = xywh
    def putimg(self, bkimg, img, imga):
        # 会修改bkimg
        x,y,w,h=self.xywh
        imgs=[ cv.resize(e, (w,h)) for e in [img, imga] ]
        crop=bkimg[y:y+h,x:x+w]
        img, mask = imgs
        if len(img.shape) != len(mask.shape):
            mask = mask[...,np.newaxis]
        mask = mask/255.0
        crop[:]=((1-mask)*crop+mask*img).astype(np.uint8)
        return bkimg

class Cls_video_read:
    def init_files(self, fnms):
        self.caps=[ cv.VideoCapture(e) for e in fnms ]
        return self
    
    def pre_read(self, idx, n):
        return [ self.caps[idx].read()[1] for i in range(n) ]
    def read_imgs(self):
        res=[ cap.read() for cap in self.caps ]
        imgs=[ e[1] for e in res ]
        isend=False
        for e in imgs:
            if e is None:
                isend=True
                break
        return isend, imgs

class Cls_video:
    def __init__(self, dbg=False, aug="-shortest"):
        self.dbg=dbg
        #self.ffmpeg_aug="-shortest -b:v 1000000k"
        self.ffmpeg_aug=aug
        self.get_video_infos = get_video_infos
    def task_comm(self, stage, arr=None):
        if stage==1:
            fv_pre, fps, self.fwav = arr[:3]
            self.audio_offset=None
            if len(arr)==4:
               self.audio_offset = arr[3] 
            self.fv_pre = fv_pre
            self._delay_info = [fv_pre+".avi", fps, 'XVID', 1]
            self.isfirst = True
            return self
        elif stage==2:
            self.write_delay( arr )
        elif stage==21:
            assert len(arr) >0
            for img in arr: self.write_delay( img )
        elif stage==3:
            self.close()
            mp={"fwav": self.fwav, "aug": self.ffmpeg_aug}
            if self.audio_offset:
                mp["offset"] = self.audio_offset
                self.convert_by_ffmpeg( self.fv_pre+".mp4", **mp )
            else:
                self.convert_by_ffmpeg( self.fv_pre+".mp4", **mp )
        elif stage==31:
            self.close()
        else:
            print("Cls_video: stage not existed")

    def init_delay(self, fvideo, fps, fourcc='XVID', iscolor=1):
        self._delay_info = [fvideo, fps, fourcc, iscolor]
        self.isfirst = True
        return self
    def write_delay(self, img):
        if self.isfirst:
            self.isfirst=False
            wh=img.shape[:2]
            wh = ( wh[1], wh[0] )
            fv, fps, fourcc, iscolor = self._delay_info
            self.init(fv, fps=fps, wh=wh, fourcc=fourcc, iscolor=iscolor)
        self.write(img)

    def init_by_cap(self, fvideo, cap_or_f, fourcc=None, verbos=False):
        if isinstance(cap_or_f, str):
            cap_or_f = cv.VideoCapture(cap_or_f)
        infos = get_cap_info( cap_or_f )
        if verbos: print("video info:", infos)
        w, h, fps = infos[:3]
        fcc = infos[-1]
        self.infos = infos
        if fourcc is not None: fcc = fourcc
        #print(fcc)
        return self.init(fvideo, fps=fps, wh=(w, h), fourcc=fcc)

    def init(self, fvideo, fps=20, wh=(512, 512), fourcc='MJPG', iscolor=1):
        fourcc  = cv2.VideoWriter_fourcc(*fourcc) #(*'XVID')
        cap_out = cv2.VideoWriter(fvideo, fourcc, fps, wh, iscolor)
        self.cap_out = cap_out
        self.fvideo = fvideo
        return self

    def write(self, img): self.cap_out.write(img)

    def close(self): self.cap_out.release()
    
    def convert_by_ffmpeg(self, fout, vcodec='h264', fin=None, fwav=None, aug="-shortest", offset=None):
        if fin is None: fin=self.fvideo
        #os.system(f"ffmpeg -y -loglevel error -i {fin} -vcodec {vcodec} -c:a copy {fout}")
        cmdbg=f"ffmpeg -y -loglevel error -i {fin}"
        if fwav:
            if offset:
                cmd=f"{cmdbg} -itsoffset {offset} -i {fwav} -vcodec {vcodec} -strict -2 {aug} {fout}"
            else:
                cmd=f"{cmdbg} -i {fwav} -vcodec {vcodec} -strict -2 {aug} {fout}"
        else:
            cmd=f"{cmdbg} -vcodec {vcodec} {aug} {fout}"
        if self.dbg:
            print(cmd)
        os.system(cmd)

    # 自动支持h264，但会改变图片大小
    def init_easy(self, fv, fps):
        self.cap_out = imageio.get_writer(fv, fps=fps)
    def write_easy(self, img, isbgr):
        if isbgr: img=np.flip(img, axis=-1)
        self.cap_out.append_data(img)

def imgs2video(fv, dimgs, fps=1):
    import moviepy
    import moviepy.video.io.ImageSequenceClip
    
    if type(dimgs) is list:
        frames_path = dimgs
    else:
        frames_name = sorted(os.listdir(dimgs))
        frames_path = [dimgs+frame_name for frame_name in frames_name]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(fv, codec='libx264')

def video2imgs(fv, isbgr=True):
    imgs=[]
    cap = cv.VideoCapture(fv)
    while True:
        ret, img = cap.read()
        if img is None: break
        imgs.append(img if isbgr else np.flip(img, axis=-1))
    return imgs

if __name__=="__main__":
    import argparse
    import numpy as np
    from tqdm import tqdm
    from .pic import drawCirclev2
    from chj.libext.exe import ffmpeg
    parser = argparse.ArgumentParser(description='python -m chj.comm.video --mergev "merge.mp4;$fnm1:$fnm2;2h"')
    parser.add_argument('--showlm', type=str, help='fout;fv;flm;fvtmp')
    parser.add_argument('--merge', type=int, default=1, help='0,1,2')
    parser.add_argument('--mergev', type=str, help='fout;v1:v2:..;[2h|2v|3h]')
    parser.add_argument('--color', type=str, default="0,255,0")
    
    args = parser.parse_args()
    if args.showlm:
        sz=args.showlm.split(";")
        assert len(sz)==4
        color=[ int(e) for e in args.color.split(',') ]
        lms=np.load(sz[2])
        cap = cv.VideoCapture(sz[1])
        cls_video = Cls_video().init_by_cap(sz[3], cap, fourcc="XVID") 
        n_f=int(cap.get(7))
        for i in tqdm(range(n_f)):
            ret, img= cap.read()
            if img is None: break
            if i>=lms.shape[0]: break
            drawCirclev2(img, lms[i],color=color)
            cls_video.write(img)
        cls_video.close()
        if args.merge==1:
            ffmpeg.merge_video(sz[0], [sz[1], sz[3]], tp="h", aug="-strict -2")
        elif args.merge==2:
            cls_video.convert_by_ffmpeg(sz[0])
    elif args.mergev:
        sz=args.mergev.split(";")
        assert len(sz)==3
        fnms=sz[1].split(":")
        ffmpeg.merge_video(sz[0], fnms, tp=sz[2], aug="-c:v h264 -strict -2") # -b:v 100k

