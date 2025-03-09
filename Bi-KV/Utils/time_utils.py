from datetime import datetime

def now_time():
    now = datetime.now()
    nowtime = now.strftime("%Y-%m-%d %H:%M:%S") + f",{now.microsecond // 1000:03d}"
    return nowtime