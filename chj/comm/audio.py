from pydub import AudioSegment

# https://zhuanlan.zhihu.com/p/196387205 很多基础技术
# https://www.codenong.com/51434897/ 
# https://www.jianshu.com/p/65b32512f768

class Cls_audio_handle:
    pass

def change_voice(fdst, fsrc, adddb ):
    sound = AudioSegment.from_file(fsrc, "wav")
    dBFS = sound.dBFS + adddb
    normalized_sound = sound.apply_gain(dBFS)
    normalized_sound.export(fdst, format="wav")

