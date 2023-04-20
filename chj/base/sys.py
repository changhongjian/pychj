# -*- coding:utf-8 -* 

import re
import os
import platform
import subprocess

#2018-3-8
def exec_ruby(ruby_code, show_runcode=1):
    # 主要是多行的问题，在windows ubuntu不太一样，就统一这样吧
    ruby_code=re.sub(r'\s*^#', "", ruby_code)
    #p(ruby_code)
    ruby_code=re.sub(r'\n', ";", ruby_code)
    ruby_code=re.sub(r'"', "\\\"", ruby_code)
    cmd="ruby -e \""+ruby_code+"\""
    if show_runcode==1: print( cmd )
    return exec_cmd(cmd) 

def exec_cmd(cmd, tp=0):
    if tp==1:
        r = os.popen(cmd)  
        text = r.read()  
        r.close()  
        return text 
    elif tp == 0:
        return subprocess.call(cmd, shell=True)

def sys_type():
    sysstr = platform.system()
    if(sysstr =="Windows"):
        return 0
    elif(sysstr == "Linux"):
        return 1
    elif(sysstr == 'Darwin'):
        return 2
    else:
        return -1
        

# CHJ_HOMEDIR(1, '/wpr/chj/', 0) )
# CHJ_HOMEDIR(0, '/bin/anaconda',1) 
# real 表示磁盘真实位置，不考虑相对的
def CHJ_HOMEDIR(isreal, refdir, refdir_type):
    func = os.path.realpath if isreal else os.path.abspath
    path = func(__file__)
    arr=path.split(refdir)
    if refdir_type == 0: return arr[0]+refdir
    else: return arr[0]

