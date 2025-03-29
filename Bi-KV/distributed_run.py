import subprocess
import time
import socket
import signal
import os

import yaml

# 获取本机 IP 地址
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip

# 清理所有机器上的相关进程
def cleanup_processes(processes, hosts):
    print("Cleaning up all processes across all machines...")
    
    # 首先尝试优雅终止本地已启动的进程
    for p in processes:
        if p.poll() is None:  # 如果进程仍在运行
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)  # 向进程组发送终止信号
                p.wait(timeout=5)  # 等待 5 秒
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)  # 强制杀死
            except Exception as e:
                print(f"Error terminating process {p.pid}: {e}")

    # 在所有机器上查找并杀死 distributed_grpc_init.py 进程
    script_name = "distributed_grpc_init.py"
    for ip, _ in hosts:
        if ip == get_local_ip():
            # 本地机器
            cmd = f"ps -ef | grep '{script_name}' | grep -v grep | awk '{{print $2}}' | xargs -r kill -9"
            print(f"Cleaning up on local machine {ip}: {cmd}")
            subprocess.run(cmd, shell=True)
        else:
            # 远程机器
            cmd = f"""ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {ip} "ps -ef | grep '{script_name}' | grep -v grep | awk '{{print \$2}}' | xargs -r kill -9" """
            print(f"Cleaning up on remote machine {ip}: {cmd}")
            subprocess.run(cmd, shell=True)
    
    print("All processes have been terminated across all machines.")

config_path = "../config.yml"
with open(config_path, 'r') as file:
    yaml_config = yaml.safe_load(file)

# 定义主节点 IP（相当于 MASTER_ADDR）
MASTER_ADDR = yaml_config['grpc']['master_addr']
MASTER_PORT = yaml_config['grpc']['master_port']

# 读取 hostfile（机器列表）
hostfile = yaml_config['grpc']['slots']

# 解析 hostfile
hosts = []
for line in hostfile:
    parts = line.split()
    ip = parts[0]
    slots = int(parts[1].split("=")[1])  # 获取 slots 数量
    hosts.append((ip, slots))

WORLD_SIZE = sum(slots for _, slots in hosts)  # 计算总进程数
script_path = "distributed_grpc_init.py"  # 替换为你的 Python 脚本路径

# Conda 环境设置
CONDA_ENV = "llamarec"  # 替换为你的 Conda 环境名称
CONDA_INIT_SCRIPT = "/home/wsh/miniconda3/etc/profile.d/conda.sh"  # 替换为你的 Conda 初始化脚本路径

# 获取本机 IP
LOCAL_IP = get_local_ip()

# 启动所有进程
processes = []
rank = 0  # 进程 rank 计数

try:
    for ip, slots in hosts:
        for local_rank in range(slots):
            # 构造基本命令
            base_cmd = f"cd /share/nfs/wsh/Bi-KV/Bi-KV/ && source {CONDA_INIT_SCRIPT} && conda activate {CONDA_ENV} && env RANK={rank} WORLD_SIZE={WORLD_SIZE} MASTER_ADDR={MASTER_ADDR} MASTER_PORT={MASTER_PORT} KVCACHE_NUM={int(WORLD_SIZE/2 - 1)} WORKER_NUM={int(WORLD_SIZE/2 - 1)} python3 {script_path}"
            
            if ip == LOCAL_IP:
                # 如果是本机，直接运行命令
                cmd = f"bash -c '{base_cmd}'"
                # print(f"Launching locally: {cmd}")
                p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)  # 使用进程组
            else:
                # 如果是远程机器，使用 SSH
                cmd = f"""ssh -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {ip} "bash -c '{base_cmd}'" """
                # print(f"Launching remotely: {cmd}")
                p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)  # 使用进程组
            
            processes.append(p)
            rank += 1  # 递增 rank

    # 监控进程
    while processes:
        for i, p in enumerate(processes):
            if p.poll() is not None:  # 如果进程已结束
                return_code = p.returncode
                if return_code != 0:  # 如果返回码非 0，表示异常退出
                    print(f"Process {i} (PID: {p.pid}) failed with return code {return_code}. Terminating all processes...")
                    cleanup_processes(processes, hosts)
                    exit(1)  # 退出脚本
                else:
                    print(f"Process {i} (PID: {p.pid}) completed successfully.")
                processes.pop(i)  # 移除已完成的进程
                break
        time.sleep(1)  # 短暂休眠，避免高 CPU 占用

except KeyboardInterrupt:
    print("Received KeyboardInterrupt. Terminating all processes...")
    cleanup_processes(processes, hosts)
except Exception as e:
    print(f"An error occurred: {e}. Terminating all processes...")
    cleanup_processes(processes, hosts)
    exit(1)

print("All processes have finished successfully.")