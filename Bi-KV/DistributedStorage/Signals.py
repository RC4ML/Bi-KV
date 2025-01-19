# Signals.py

# 定义信号常量
SIGNAL_SEND = 1      # 发送数据信号
SIGNAL_RECV = 2      # 接收数据信号
SIGNAL_ACK = 3       # 确认信号（未使用，可根据需要扩展）
SIGNAL_CHECK = 4     # 查询Cache信号
SIGNAL_SKIP = 5      # 跳过信号
SIGNAL_TERMINATE = -1 # 终止信号

CACHE_MISS = 0
CACHE_HIT = 1
