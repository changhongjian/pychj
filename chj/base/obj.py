# -*- coding:utf-8 -* 
import re
def get_object_attrs(obj):
    p1 = r"^__\S+__$"
    pattern1 = re.compile(p1)
    lines=dir(obj )
    res=[]
    for line in lines:
        if pattern1.match(line)==None: res.append(line)
        else: pass
    return res

class jsobj():pass

def set_arr_mp(mp,  k, v):
    arr=mp.get(k, [])
    arr.append(v)
    mp[k]=arr

def set_obj_attr(obj, k, v):
    _v = getattr(obj, k, "")
    setattr(obj, k, type(_v)(v))

def set_obj_by_mp(obj, mp):
    for k, v in mp.items():
        set_obj_attr(obj, k, v)

def set_obj_by_str(hparams, hparams_string):
    if not hparams_string: return
    if len(hparams_string.strip())=="": return
    arr=hparams_string.split(";")
    for e in arr:
        k, v = [ x.strip() for x in e.split("=") ]
        set_obj_attr(hparams, k, v)

class HParams:
    def __init__(self, **kwargs):
        self.data = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


if __name__=="__main__":
    class obj:
        a=jsobj()
        b=12
    obj.a.c = 99
    set_obj_by_str( obj, "c=111;a.c=33" )
    print( obj.b, obj.a.c )

