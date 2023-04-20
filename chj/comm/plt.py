# -*- coding:utf-8 -* 

import re
import numpy 
import numpy as np
import scipy.io as scio
import cv2 as cv
import cv2
from PIL import Image
from numpy import random
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fig2img(fig=None, mode="BGR"):
    if fig is None: fig = plt.figure()
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    if mode=="RGBA":
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
    elif mode in ["BGR", "RGB"]:
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (w, h, 3)
        image = Image.frombytes("RGB", (w, h), buf.tostring())
        image = np.asarray(image)
        if mode == "BGR":
            return image[..., ::-1].copy()
    return image

def draw_signal(y, fimg=None, size=None, peak=None):
    if size:
        plt.figure(figsize=size) # (w,h)
    x = np.arange(len(y))
    plt.plot(x,y,'g-')
    if peak is not None:
        y2 = y.copy()
        y2[:] = -1
        y2[peak] = y.mean()
        plt.plot(x,y2,'rx')
    if fimg: plt.savefig(fimg)
    
# 2018-3-4

# o p 可以快捷操作图片
# 终于找到退出程序的方法了
def on_release(event):
    if event.button == 3:  plt.close()
def on_press(event):
    if event.dblclick == True:  exit(0)

def create_fig():
    fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    return fig

def set_fig(fig):
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    return fig

def draw_mesh(pts,tris=None):
    if pts is not None: plt.plot(pts[:, 0], pts[:, 1], '.')
    if tris is not None: plt.triplot(pts[:, 0], pts[:, 1], tris)


def plt_show(img=None, pts=None, tris=None, fig=None):
    if fig is None: fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)

    #plt.clf()
    #plt.cla()
    #fig.clear()

    if img is not None:
        if len(img.shape)==2: plt.imshow(img, cmap="gray")
        else:  plt.imshow(img) # 这里修改了
    if pts is not None: plt.plot(pts[:, 0], pts[:, 1], '.')
    if tris is not None: plt.triplot(pts[:, 0], pts[:, 1], tris)

    plt.show()

    '''
    plt.imshow(img)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # key=showimg(data)
    # if key==27: exit(0)

    fig = plt.figure()
    plt.imshow(data[ 250:350,250:350 ])
    fig.canvas.mpl_connect("button_release_event", on_release)
    plt.show(fig)
    '''
    
def plt_save(fname, img=None, pts=None, tris=None, fig=None):
    if fig is None: fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)

    if pts is not None: plt.plot(pts[:, 0], pts[:, 1], '.')
    if tris is not None: plt.triplot(pts[:, 0], pts[:, 1], tris)
    if img is not None:
        if len(img.shape)==2: plt.imshow(img)
        else:  plt.imshow(img)
    #plt.show()
    fig.savefig(fname)
    plt.close()

def plt_cmp_3D(point_a, point_b, ax=None, isshow=True):
    if ax is None: 
        create_fig()
        ax = plt.subplot(111, projection='3d') # 创建一个三维的绘图工程 
    #  将数据点分成三部分画，在颜色上有区分度 
    ax.scatter(point_a[:, 0], point_a[:, 1], point_a[:, 2], c='g') 
    ax.scatter(point_b[:, 0], point_b[:, 1], point_b[:, 2], c='r') 
    ax.set_zlabel('Z') # 坐标轴 
    ax.set_ylabel('Y') 
    ax.set_xlabel('X') 
    if isshow: plt.show()
    
def plt_3D(point_a, has_fig=False, isshow=True, type="scatter"):
    if not has_fig: create_fig()
    ax = plt.subplot(111, projection='3d') # 创建一个三维的绘图工程 
    if type == "surf":
        ax.plot_trisurf(point_a[:, 0], point_a[:, 1], point_a[:, 2], c='g')
    else:
        ax.scatter(point_a[:, 0], point_a[:, 1], point_a[:, 2], c='g') 
    ax.set_zlabel('Z') # 坐标轴 
    ax.set_ylabel('Y') 
    ax.set_xlabel('X') 
    if isshow: plt.show()

def plt_cmp_2D(point_a, point_b, has_fig=False, isshow=True):
    if not has_fig: create_fig()
    plt.plot(point_a[:, 0], point_a[:, 1], ".g")
    plt.plot(point_b[:, 0], point_b[:, 1], ".r")
    if isshow: plt.show()

def plt_2D(point_a, has_fig=False, isshow=True, tp=".g"):
    if not has_fig: create_fig()
    plt.plot(point_a[:, 0], point_a[:, 1], tp)
    if isshow: plt.show()

def draw3Dplane(x,y,z,d=0,range=100,ax=None):
    if ax==None:
        fig = plt.figure()
        ax = Axes3D(fig)
    X = np.arange(-range, range, 20)
    Y = np.arange(-range, range, 20)
    X, Y = np.meshgrid(X, Y)
    Z = (-d-X*x-Y*y)/z
   
    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')


'''

# 终于改成自己需要的样子了
def save_mesh_and_img(fname, img, pts, tris):
    fig = plt.figure(figsize=(8, 12))
    #fig = plt.figure()
    
    h = img.shape[0]
    plt.triplot(pts[:, 0], pts[:, 1], tris)
    plt.plot(pts[:, 0], pts[:, 1], '.')
    plt.imshow(img[:,:,::-1])
    
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
    #mng.window.showMaximized()
    
    plt.savefig(fname)
    plt.close()
'''    
    
        
