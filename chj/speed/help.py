
def get_chj_speed(fcdll):
    from chj.speed.cdll import cls_speed
    speed = cls_speed()
    speed.load_cdll(fcdll)
    return speed

