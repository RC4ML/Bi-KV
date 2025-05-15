import grpc
import atexit
from threading import Lock

class ChannelPool:
    def __init__(self):
        self.channels = {}
        self.lock = Lock()
        # 注册退出时的回调
        atexit.register(self.shutdown)

    def get_channel(self, address):
        """获取一个已有的或新建的 gRPC Channel"""
        with self.lock:
            if address not in self.channels:
                self.channels[address] = grpc.insecure_channel(
                    address,
                    options=[
                        ('grpc.keepalive_time_ms', 30000),
                        ('grpc.keepalive_timeout_ms', 10000),
                    ]
                )
            return self.channels[address]

    def shutdown(self):
        """关闭所有 Channel，释放资源"""
        with self.lock:
            print("ChannelPool 正在关闭所有 Channel...")
            for addr, channel in self.channels.items():
                print(f"正在关闭 {addr}")
                channel.close()
            self.channels.clear()