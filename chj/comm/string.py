
def time_fmt(e):
    mil = e%1000
    e = e//1000
    m = e//60
    s = e%60
    return f"{m:02d}m{s:02d}s.{mil:03d}"
