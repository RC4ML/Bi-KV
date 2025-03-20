#!/bin/bash

# 获取当前用户
USER_NAME=$(whoami)

echo "Checking for NVIDIA processes owned by user: $USER_NAME"

# 使用 nvidia-smi 查看运行中的 GPU 进程并过滤属于当前用户的
NVIDIA_PROCESSES=$(nvidia-smi | grep $USER_NAME | awk '{print $5}')

if [ -z "$NVIDIA_PROCESSES" ]; then
    echo "No NVIDIA processes found for user: $USER_NAME"
    exit 0
fi

# 列出进程 ID
echo "Found NVIDIA processes:"
echo "$NVIDIA_PROCESSES"

# 遍历进程 ID，尝试逐一结束进程
for PID in $NVIDIA_PROCESSES; do
    echo "Attempting to kill process: $PID"
    kill $PID 2>/dev/null

    # 检查是否成功结束
    if ps -p $PID > /dev/null; then
        echo "Process $PID still running, sending SIGKILL..."
        kill -9 $PID 2>/dev/null

        if ps -p $PID > /dev/null; then
            echo "Failed to kill process $PID. You may need to check manually."
        else
            echo "Process $PID killed successfully."
        fi
    else
        echo "Process $PID terminated successfully."
    fi
done

echo "All NVIDIA processes for user $USER_NAME have been handled."
