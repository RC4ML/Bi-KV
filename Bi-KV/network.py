import os
import subprocess
import re
import socket

def get_local_ip():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        print(f"获取本地 IP 地址时出错: {e}")
        return "未知"

def get_network_interface(ip_prefix="10.0.0."):
    try:
        # 执行 ip 命令，列出所有网络接口及其 IP 地址
        result = subprocess.run(["ip", "-4", "addr"], capture_output=True, text=True)
        output = result.stdout
        local_ip = get_local_ip()
        # 查找与 IP 前缀匹配的网络接口名称
        match = re.search(rf"(\d+): (\S+):.*\n\s+inet {ip_prefix}\d+", output)
        if match:
            interface_name = match.group(2)
            # print(f"{local_ip} 找到网络接口: {interface_name}")
            return interface_name
        else:
            print(f"未找到与前缀 {ip_prefix} 匹配的网络接口。")
            return None
    except Exception as e:
        print(f"获取网络接口时出错: {e}")
        return None

def set_gloo_socket(ip_prefix="10.0.0."):
    interface = get_network_interface(ip_prefix)
    if interface:
        os.environ["GLOO_SOCKET_IFNAME"] = interface
        os.environ["TP_SOCKET_IFNAME"] = interface
        # print(f"GLOO_SOCKET_IFNAME 已设置为 {interface}")
    else:
        print("未找到匹配的网络接口，无法设置 GLOO_SOCKET_IFNAME。")

def init_network():
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', '10.0.0.1')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29051')
    set_gloo_socket()
    # print(f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}")
    # print(f"Using TP_SOCKET_IFNAME={os.environ['TP_SOCKET_IFNAME']}")
    # print(f"MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")