# -*- coding:utf-8 -* 

import re
import os
import platform

# pip intall pyerclip
import pyperclip

def clip(txt=None):
    if txt is None: return pyperclip.paste() 
    else: pyperclip.copy(txt)  

