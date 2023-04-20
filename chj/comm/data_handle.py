from .include_full import *
'''
author: wpr @2017-11-15
email:  wozaibeihua@163.com
'''

'''
  @2017-12-21 再次修改
'''

def confine_num(num, dn, up):
    '''
    dn <= num <= up
    '''
    if num < dn : num = dn
    if num > up: num = up
    return num

def confine_rectXY(wh, rectXY):
    '''
    :param wh: 最大的大小 0<= <=wh
    :param rect: 未修改之前的
    :return: rect 修改之后的
    '''

    res = rectXY
    for i,num in enumerate(res): res[i] = confine_num(num, 0, wh[i%2])

    return res


def large_rectXY(rXY, rates):
    '''
    左上角减少，右下角增加 根据长宽
    :param rXY:
    :param rates: size=4
    :return:
    '''
    wh = [ rXY[2]-rXY[0], rXY[3]-rXY[1] ]
    wh = wh * 2
    operate = [ lambda x,y: x-y ] * 2 + [ lambda x,y: x+y] * 2
    for i in range(4):
        add = wh[i]*rates[i]
        rXY[i] = operate[i](rXY[i],add)

    return rXY

def crop_with_rectXY(img, lm, rectXY, large_rates=[1]*4):
    if type(large_rates) == float: large_rates=[large_rates]*4
    assert( 4 == len(large_rates) )
    newrect = large_rectXY(rectXY, large_rates )
    #r = confine_rectXY( img.shape - 1, newrect )
    r = confine_rectXY( [img.shape[1] - 1, img.shape[0] - 1], newrect)

    r = [ int(x) for x in r ]
    img = img[ r[1]:r[3]+1, r[0]:r[2]+1 ].copy()
    lm = lm - np.array([r[0], r[1]])

    return img,lm


# lm 是原来的 这个就根据lm 和 img的大小进行裁剪 随机裁剪
def shrine_img(img,lm):

    # 额外多出一个像素
    rectXY = np.array([np.floor(lm.min(axis=0))-1,
                   np.ceil(lm.max(axis=0))+1]).reshape(-1)

    rates = [ random.uniform(0,1)  for i in range(4) ]

    img, lm = crop_with_rectXY(img, lm, rectXY, rates)

    return img, lm

''' 并未在 12-21 修改 '''
def shift_img(img,lm, rectXY=NULL, bias_rate=0.1):
    '''
    就是按照rect裁剪图片，同时对裁剪框做一定的随机扰动
    :param img: 原始图片大的
    :param lm: 原始图片中的lm点的位置
    :param rectXY: 输出图片的大小就按照这个来裁剪。如果为NULL 表明是原图大小
        注意 lm 必须要在 rectXY 中才行
    :param bias_rate: 允许偏移的最大比例
    :return: 返回裁剪后的图片，集对应的lm
    '''
    imgh, imgw, _ = img.shape

    # rect
    if rectXY == NULL:
        rectXY = np.array([0,0,imgw-1,imgh-1],np.int32)
    else: rectXY = np.array(rectXY,np.int32)

    # 额外多出一个像素
    bb = np.array([np.floor(lm.min(axis=0)),
                   np.ceil(lm.max(axis=0))]).reshape(-1).astype(int)


    org_w = rectXY[2] - rectXY[0]
    org_h = rectXY[3] - rectXY[1]

    addwh=[None]*4
    addwh[0] = bb[0] - rectXY[0]
    addwh[1] = bb[1] - rectXY[1]
    addwh[2] = rectXY[2] - bb[2]
    addwh[3] = rectXY[3] - bb[3]

    addwh = [ x * bias_rate for x in addwh]

    shift_x = random.uniform( -addwh[0], addwh[2] )
    shift_y = random.uniform( -addwh[1], addwh[3] )

    x1 = int(shift_x)
    x2 = int(x1+org_w)
    y1 = int(shift_y)
    y2 = int(y1+org_h)

    # 大的地方也需要有不同的颜色进行填充
    #newImg = np.random.randint(256, size=img.shape)
    #newImg = newImg.astype(np.uint8)

    newshape = np.array([0]*len(img.shape))
    #ps(newshape)
    # 这里自己都忘记加1了
    newshape[-1]=img.shape[-1]
    newshape[0] = org_h+1
    newshape[1] = org_w+1
    newImg = np.zeros(newshape,np.uint8)
    #p(newImg.shape)
    #p(img.shape)

    #我试图改成0-based
    ul = [x1, y1]
    br = [x2, y2]
    newX = np.array(
        [max(0, -ul[0]), min(br[0], imgw-1) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(0, -ul[1]), min(br[1], imgh-1) - ul[1]], dtype=np.int32)
    oldX = np.array([max(0, ul[0] ), min(br[0], imgw-1)], dtype=np.int32)
    oldY = np.array([max(0, ul[1] ), min(br[1], imgh-1)], dtype=np.int32)
    # 经过上面的处理其实二者的长宽都是一样的，很巧妙的做法
    # p(oldX[1] - oldX[0], newX[1] - newX[0])
    newImg[newY[0]:newY[1]+1, newX[0]:newX[1]+1] =  img[oldY[0]:oldY[1]+1, oldX[0]:oldX[1]+1, :]

    lm -= ul
    return [newImg, lm]

def getRotateMatrix2D(theta, opencv=True):
    if opencv == True:
        theta = -theta

    theta = theta * np.pi / 180
    alpha = np.cos(theta)
    beta = np.sin(theta)
    # 逆时针， 且对于一般的坐标系
    M_shear = np.array([ [alpha, -beta], [beta, alpha]], dtype=np.float32)
    return M_shear
def getRotateMatrix(theta, center, opencv=True):
    if opencv == True:
        theta = -theta

    theta = theta * np.pi / 180
    alpha = np.cos(theta)
    beta = np.sin(theta)
    # 逆时针， 且对于一般的坐标系
    M_shear = np.array([
        [alpha, -beta, (1 - alpha) * center[0] + beta * center[1]],
        [beta, alpha, (1 - alpha) * center[1] - beta * center[0]]],
        dtype=np.float32)
    return M_shear

def image_rotate(img, lm, range=[-15, +15]):
    theta = random.uniform(*range)

    iHeight, iWidth, _ = img.shape

    center = [0, 0]
    M_shear = getRotateMatrix2D(theta, center)

    # 最终还是由对角线的点来决定
    corner_points = np.ones((2, 4))
    corner_points[0, 0] = 0
    corner_points[1, 0] = 0
    corner_points[0, 1] = iWidth - 1
    corner_points[1, 1] = 0
    corner_points[0, 2] = 0
    corner_points[1, 2] = iHeight - 1
    corner_points[0, 3] = iWidth - 1
    corner_points[1, 3] = iHeight - 1
    corner_points = M_shear.dot(corner_points)

    minx, maxx = corner_points[0, :].min(), corner_points[0, :].max()
    miny, maxy = corner_points[1, :].min(), corner_points[1, :].max()

    # 之前这里少了1
    iNewWidth = np.max([int(maxx - minx)+1, iWidth])
    iNewHeight = np.max([int(maxy - miny)+1, iHeight])

    newimg = np.zeros((iNewHeight, iNewWidth, 3), dtype=np.uint8)

    offset_x = (iNewWidth - iWidth) // 2
    offset_y = (iNewHeight - iHeight) // 2

    newimg[offset_y: offset_y + iHeight,
    offset_x: offset_x + iWidth, :] = img[:, :, :]

    center = [(iNewWidth - 1) / 2, (iNewHeight - 1) / 2]
    M_shear = getRotateMatrix(theta, center)

    img_sheared = cv.warpAffine(newimg, M_shear, (iNewWidth, iNewHeight))

    xy=None
    if lm is not None:
        xy = lm + np.array([offset_x, offset_y])

        M_shear = M_shear[0:2, :]
        M_shear = M_shear.transpose()

        xy = xy.dot(M_shear[0:2,:])
        xy += M_shear[2, :]

    return img_sheared, xy

# 传入的xy 是 n*2
def occluding_boxes_v1(img, xy, range=[0.05, 0.15]):

    scale = random.uniform(*range)

    minx, maxx = xy[:, 0].min(), xy[:, 0].max()
    miny, maxy = xy[:, 1].min(), xy[:, 1].max()
    dx = maxx - minx
    dy = maxy - miny
    len = np.sqrt(dx * dx + dy * dy)
    blen = int(len * scale)
    h, w, _ = img.shape
    h-=1
    w-=1
    px = random.randint(int(minx), int(maxx-blen))
    py = random.randint(int(miny), int(maxy-blen))
    img[py:py + blen, px:px + blen, :] = 0

def occluding_boxes(img, xy, range=[0.05, 0.15]):

    scale = random.uniform(*range)

    minx, maxx = xy[:, 0].min(), xy[:, 0].max()
    miny, maxy = xy[:, 1].min(), xy[:, 1].max()
    dx = maxx - minx
    dy = maxy - miny
    blenw = int( dx * random.uniform(*range) )
    blenh = int( dy * random.uniform(*range) )

    px = random.randint(int(minx), int(maxx-blenw))
    py = random.randint(int(miny), int(maxy-blenh))
    img[py:py + blenh, px:px + blenw, :] = 0

def image_random_brightness(img,range=[0.7,1.4]):
    gamma=random.uniform(*range)
    img = exposure.adjust_gamma(img, gamma)
    return img

def image_random_saturability_opencv(img,range=[0.7,1.3]):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s=random.uniform(*range)
    img_hsv[:, :, 1] = s * img_hsv[:, :, 1]
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_hsv


''' 先从大图中抠出小图  '''
def rect_to_rectXY(r):
    r[2] += r[0]
    r[3] += r[1]
    return r

def rectXY_to_center(r):
    center = np.array(
        [r[2] - (r[2] - r[0]) / 2.0, r[3] - (r[3] - r[1]) / 2.0], np.float32)
    return center

def rect_to_center(r):
    center = np.array(
        [r[0] + r[2] / 2.0, r[1] + r[3] / 2.0], np.float32)
    return center

def c_s_to_rect(c,h):
    r = np.array(
        [c[0]-h,c[1]-h,c[0] + h, c[1] + h], np.float32)
    return r

# 平移和缩放
# 它构造的是 pos_input = T * pos_orgImage
# 里面其实用的是比例，或者似乎可以理解为先平移后缩放
def transform(point, center, scale, resolution, invert=False):
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    #h = 200.0 * scale
    h =  scale
    t = np.eye(3)

    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.linalg.inv(t)

    new_point = (np.matmul(t, _pt))[0:2]

    return new_point.astype(np.int32)

def crop(image, center, scale, resolution=256.0):
    # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    # 这个就是找到up left和bottom right的点到底能跑在那里
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)

    # 好像没用
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)

    # RGB or gray
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    # 确实巧妙， ul 和 br 是针对原图片的，这个newXY 应该牢记本应该是从原图上扣下来的，至于代码是那样其实完全可以自己想到。
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    # 经过上面的处理其实二者的长宽都是一样的，很巧妙的做法
    #p(oldX[1] - oldX[0], newX[1] - newX[0])
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
    ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg

def crop_img(image, ul, br):
    # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    # RGB or gray
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    # 确实巧妙， ul 和 br 是针对原图片的，这个newXY 应该牢记本应该是从原图上扣下来的，至于代码是那样其实完全可以自己想到。
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    # 经过上面的处理其实二者的长宽都是一样的，很巧妙的做法
    #p(oldX[1] - oldX[0], newX[1] - newX[0])
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]  ] = \
        image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

    return newImg

def crop_image_label(img,rect,lb, resolution):
    center=rect_to_center(rect)
    #center[1] = center[1] - rect[3] * 0.1
    #scale = (rect[2] + rect[3]) #/ 200.0
    maxlen = max(rect[2] , rect[3])
    scale = 1.8 * maxlen

    #cv2.circle(img, (center[0],center[1]), 3, (0, 0, 255),-1)

    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)

    img=crop_img(img,ul,br)

    ul-=1
    br-=1

    #rect=[ul[0],ul[1],br[0]-ul[0],br[1]-ul[1]] error

    #cv2.rectangle(img, (ul[0], ul[1]), (br[0], br[1]), (0, 255, 0), 3)

    lb-=ul
    #img=img[ul[1]:br[1]+1,ul[0]:br[0]+1].copy()

    return img,lb


def SWAP(shape, i, j):
    tmp = shape[i - 1, 0]
    shape[i - 1, 0] = shape[j - 1, 0]
    shape[j - 1, 0] = tmp

    tmp = shape[i - 1, 1]
    shape[i - 1, 1] = shape[j - 1, 1]
    shape[j - 1, 1] = tmp


def swap_lm68(gt_shape):
    for k in range(1, 8 + 1):
        SWAP(gt_shape, k, 18 - k)
    for k in range(18, 22 + 1):
        SWAP(gt_shape, k, 45 - k)
    for k in range(37, 40 + 1):
        SWAP(gt_shape, k, 83 - k)

    SWAP(gt_shape, 42, 47)
    SWAP(gt_shape, 41, 48)
    SWAP(gt_shape, 32, 36)
    SWAP(gt_shape, 33, 35)

    for k in range(49, 51 + 1):
        SWAP(gt_shape, k, 104 - k)

    SWAP(gt_shape, 60, 56)
    SWAP(gt_shape, 59, 57)
    SWAP(gt_shape, 61, 65)
    SWAP(gt_shape, 62, 64)
    SWAP(gt_shape, 68, 66)

#@5-10 偷个懒添加了 rect
def flip_image_label(img,lb):
    img=cv2.flip(img, 1)

    if lb is not None:
        w = img.shape[1] - 1
        lb[:,0]=w-lb[:,0]
        swap_lm68(lb)

    return img,lb


#@3-27
def compute_T(cx, cy, length, resolution):
    T = np.eye(3)

    t = np.eye(3)
    t[0, 2] = -cx
    t[1, 2] = -cy

    T = t.dot(T)

    t = np.eye(3)
    t[0, 0] = 1 / length
    t[1, 1] = 1 / length

    T = t.dot(T)

    t = np.eye(3)
    t[0, 2] = 0.5
    t[1, 2] = 0.5

    T = t.dot(T)

    t = np.eye(3)
    t[0, 0] = resolution
    t[1, 1] = resolution

    T = t.dot(T)

    return T, np.linalg.inv(T)


def pre_compute_T(cx, cy, length, resolution, inv=False):
    t = np.eye(3)  # t=np.eye((3,3), np.float64)

    t[0, 0] = resolution / length
    t[1, 1] = resolution / length
    t[0, 2] = resolution * (-cx / length + 0.5)
    t[1, 2] = resolution * (-cy / length + 0.5)

    if inv == True: return np.linalg.inv(t)
    return t


def compute_newxy_and_oldxy(T, r, wd, ht):
    wd -= 1
    ht -= 1

    pts = np.array([[0, 0, 1], [r, r, 1]]).transpose()
    pts = T.dot(pts).astype(np.int32)  # 根本不是四舍五入
    #p(pts)
    ul = pts[0:2, 0]
    br = pts[0:2, 1]
    #p(br - ul)

    newDim = np.array([br[1] - ul[1] + 1, br[0] - ul[0] + 1], dtype=np.int32)

    # 确实巧妙， ul 和 br 是针对原图片的，这个newXY 应该牢记本应该是从原图上扣下来的，至于代码是那样其实完全可以自己想到。
    # 第一个就是如果new是负数就折回来，第二个就是谁小要谁
    #p("wh ", wd, ht)
    newX = np.array(
        [max(0, -ul[0]), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(0, -ul[1]), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(0, ul[0]), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(0, ul[1]), min(br[1], ht)], dtype=np.int32)
    # 经过上面的处理其实二者的长宽都是一样的，很巧妙的做法
    #p(max(0, -ul[0]), max(0, ul[0]), min(br[0], wd) - ul[0], min(br[0], wd))
    #p(max(0, -ul[1]), max(0, ul[1]), min(br[1], wd) - ul[1], min(br[1], wd))
    #p(oldX, newX, oldY, newY)
    #p(newDim)
    return newDim, [newY[0], newY[1] + 1, newX[0], newX[1] + 1], [oldY[0], oldY[1] + 1, oldX[0], oldX[1] + 1], ul


def get_crop_img(image, new_dims, newxy, oldxy):
    newDim = np.array(new_dims, dtype=np.int32)
    newImg = np.zeros(newDim, dtype=np.uint8)

    newImg[newxy[0]:newxy[1], newxy[2]:newxy[3]] = \
        image[oldxy[0]:oldxy[1], oldxy[2]:oldxy[3]]

    return newImg


def my_crop_image_from_rectxy(img, rxy, resolution, maxlen_rate=1.6, lm=None):
    #p("rxy", rxy)
    #p("rect", rxy[2] - rxy[0], rxy[3] - rxy[1])
    center = rectXY_to_center(rxy)
    #p("center", center)
    maxlen = max(rxy[2] - rxy[0], rxy[3] - rxy[1]) * maxlen_rate

    r = resolution - 1

    T = pre_compute_T(center[0], center[1], maxlen, r, inv=True)
    #p("T", T)
    ht, wd, c = img.shape  # ht, wd, _ = np.array(img.shape)-1
    new_dims, newxy, oldxy, ul = compute_newxy_and_oldxy(T, r, wd, ht)

    new_dims = list(new_dims) + [c]
    newimg = get_crop_img(img, new_dims, newxy, oldxy)
    if lm is not None:
        lm-=ul
    return newimg, lm

#@18-10-30 将中心上移
def my_crop_image_from_rectxy_fix_center(img, rxy, resolution, maxlen_rate=1.3, c_rate=0.05 ,lm=None):

    center = rectXY_to_center(rxy)
    #p("center", center)
    maxlen = max(rxy[2] - rxy[0], rxy[3] - rxy[1])

    # 只有y比较大的情况下才做
    if rxy[3] - rxy[1] > rxy[2] - rxy[0]:
        center[1] -= maxlen*c_rate # add!!
    
    maxlen *=  maxlen_rate
    
    r = resolution - 1
    T = pre_compute_T(center[0], center[1], maxlen, r, inv=True)
    #p("T", T)
    ht, wd, c = img.shape  # ht, wd, _ = np.array(img.shape)-1
    new_dims, newxy, oldxy, ul = compute_newxy_and_oldxy(T, r, wd, ht)

    new_dims = list(new_dims) + [c]
    newimg = get_crop_img(img, new_dims, newxy, oldxy)
    if lm is not None:
        lm-=ul
    return newimg, lm

#@19-1-20 
def chj_crop_image(img, rxy, resolution, maxlen_rate, c_rate):

    center = rectXY_to_center(rxy)
    maxlen = max(rxy[2] - rxy[0], rxy[3] - rxy[1])

    # 只有y比较大的情况下才做
    if rxy[3] - rxy[1] > rxy[2] - rxy[0]:
        center[1] -= maxlen*c_rate # add!!
    
    maxlen *=  maxlen_rate
    
    r = resolution - 1

    T = pre_compute_T(center[0], center[1], maxlen, r, inv=True)
    #p("T", T)
    ht, wd, c = img.shape  # ht, wd, _ = np.array(img.shape)-1
    new_dims, newxy, oldxy, ul = compute_newxy_and_oldxy(T, r, wd, ht)

    new_dims = list(new_dims) + [c]
    newimg = get_crop_img(img, new_dims, newxy, oldxy)
    return newimg, ul

#@2018-3-28 增加rect 扰动
def rect_add(x, y, id):
    if id < 2: return x - y
    if id >= 2: return x + y

def rect_fix(x, y):
    if x<0: return 0
    elif y<=0: return x
    else:
        if x>=y: x=y-1
        return x

def random_rectxy(rxy, img, rate=[-0.2, 0.3]):
    h, w = img.shape[:2]
    length =  min(rxy[2]-rxy[0],rxy[3]-rxy[1])

    bd =[0,0,w,h]
    for i in range(4):
        lth = length *  random.uniform(*rate)
        rxy[i] = rect_fix(rect_add(rxy[i], lth, i), bd[i])


# 又一次补充的
def change_img_01_no_trans(img, w, h):
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = img.astype(np.float32).copy()
    return img

def recover_img_01_no_trans(img):
    im=(img*255).astype(np.uint8)
    return im

def change_img_01(img, w, h):
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32).copy()

    return img

def change_img_128(img, w, h):
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    img = img / 128.0 - 1
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32).copy()
    return img

def recover_img_01(img):
    im=(img*255).astype(np.uint8)
    im = im.transpose((1, 2, 0))
    return im

def recover_img_128(img):
    img_float = img*128+128
    im = img_float.astype(np.uint8)
    im = im.transpose((1, 2, 0))
    return im



def get_lable_rxy_from_line(dbdir, line):
    sz = line.split()
    nm = sz[0]

    img_path = dbdir + "img/" + nm  + ".jpg"

    img = cv.imread(img_path)
    rect = [int(x) for x in sz[1:5]]

    return img, rect

def get_label_from_file(nm, enclose=False):
    sz = nm.split()
    nm=sz[0]

    img_path = param.dbdir + "img/" + nm + ".jpg"
    lm_path = param.dbdir + "label/" + nm + ".lm"

    #print(img_path)
    #exit()
    img = cv.imread(img_path)
    lm = numpy.loadtxt(lm_path, delimiter=" ", dtype=np.float32)
    if( len(lm[0])!=2): lm=lm[:,0:2].copy()

    rect= sz[1:]
    rect = [int(x) for x in rect]

    return img, lm, rect

#@18-3-27 一行中已经包含了所有的数据了
def get_label_from_line(dbdir, line):
    sz = line.split()
    nm=sz[0]

    img_path = dbdir + "img/" + nm  + ".jpg"

    img = cv.imread(img_path)
    rect = [ int(x) for x in sz[1:5] ]

    lm = np.array(sz[5:]).astype(np.float32)
    lm=lm.reshape(-1,2)

    return img, lm, rect

# @18-7-24
import abc
class BaseDataLoadMange:
    def __init__(self, db_dir, flist, need_wh):
        self.db_dir = db_dir
        self.list = readlines( flist )
        self.n_len = len(self.list)
        self.need_wh = need_wh

    def get_data(self, id=-1):
        if id<0: id = np.random.randint(self.n_len)
        return self.handle_id(id)

    @abc.abstractclassmethod
    def handle_id(self, id):
        return None



