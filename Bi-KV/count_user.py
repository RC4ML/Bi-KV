import re
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_token_distribution(log_file):
    # 初始化字典来存储token长度分布
    token_distribution = defaultdict(int)
    
    # 正则表达式匹配用户历史token长度
    pattern = re.compile(r'user_history_tokens=(\d+)')
    
    with open(log_file, 'r') as file:
        for line in file:
            # 检查是否是包含用户token信息的行
            if '[LLMInput] Generate prompt' in line:
                match = pattern.search(line)
                if match:
                    tokens = int(match.group(1))
                    token_distribution[tokens] += 1
    
    return token_distribution

def print_distribution(distribution):
    print("用户历史token长度分布统计:")
    print("Token长度\t用户数量")
    for tokens in sorted(distribution.keys()):
        print(f"{tokens}\t\t{distribution[tokens]}")

def plot_distribution(distribution):
    # 准备数据
    tokens = sorted(distribution.keys())
    counts = [distribution[t] for t in tokens]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.bar(tokens, counts)
    plt.xlabel('Token长度')
    plt.ylabel('用户数量')
    plt.title('用户历史Token长度分布')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    log_file = "distributed_system.log"  # 替换为你的日志文件路径
    
    # 分析日志文件
    distribution = analyze_token_distribution(log_file)
    
    # 打印统计结果
    print_distribution(distribution)
    
    # 绘制分布图
    plot_distribution(distribution)