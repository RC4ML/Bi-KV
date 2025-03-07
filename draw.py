import matplotlib.pyplot as plt

# 定义文件路径
file_paths = [
    '/data/zzm/Bi-KV/Bi-KV/kvcache_log_rank_3.txt',
    '/data/zzm/Bi-KV/Bi-KV/kvcache_log_rank_5.txt',
    '/data/zzm/Bi-KV/Bi-KV/kvcache_log_rank_7.txt',
    '/data/zzm/Bi-KV/Bi-KV/kvcache_log_rank_9.txt',
    '/data/zzm/Bi-KV/Bi-KV/kvcache_log_rank_11.txt'
]

# 初始化存储数据的字典
data = {
    'remote_send_times': [],
    'remote_send_tokens': [],
    'local_move_times': [],
    'local_move_tokens': []
}

# 遍历每个文件并提取数据
for file_path in file_paths:
    with open(file_path, 'r') as file:
        for line in file:
            if "[KVCache:finish remote send]" in line:
                # 提取 remote send 数据
                parts = line.split(',')
                time_part = parts[0].split('=')[1]
                token_part = parts[1].split('=')[1]
                time = float(time_part)
                tokens = int(token_part)
                if tokens > 0:  # 确保 token 数量大于 0
                    data['remote_send_times'].append(time)
                    data['remote_send_tokens'].append(tokens)
            elif "[KVCache:finish local move]" in line:
                # 提取 local move 数据
                parts = line.split(',')
                time_part = parts[0].split('=')[1]
                token_part = parts[1].split('=')[1]
                time = float(time_part)
                tokens = int(token_part)
                if tokens > 0:  # 确保 token 数量大于 0
                    data['local_move_times'].append(time)
                    data['local_move_tokens'].append(tokens)

# 计算总 token 数
total_remote_send_tokens = sum(data['remote_send_tokens'])
total_local_move_tokens = sum(data['local_move_tokens'])

# 计算平均传输速率
remote_send_speed = [tokens / time for tokens,time in zip(data['remote_send_tokens'], data['remote_send_times'])]
local_move_speed = [tokens / time for tokens,time  in zip(data['local_move_tokens'], data['local_move_times'])]

avg_remote_send_rate = sum(remote_send_speed) / len(remote_send_speed) if remote_send_speed else 0
avg_local_move_rate = sum(local_move_speed) / len(local_move_speed) if local_move_speed else 0
print(sum(remote_send_speed),len(remote_send_speed))
print(f'Average remote send speed: {avg_remote_send_rate}')
print(f'Average local move speed: {avg_local_move_rate}')
print(f'Total remote send tokens: {total_remote_send_tokens}')
print(f'Total local move tokens: {total_local_move_tokens}')

# 绘制散点图
plt.figure(figsize=(10, 6))  # 调整图表大小
plt.scatter(data['remote_send_tokens'], data['remote_send_times'], label='Remote Send', alpha=0.7)
plt.scatter(data['local_move_tokens'], data['local_move_times'], label='Local Move', alpha=0.7)

# 设置纵坐标范围
plt.ylim(0, 0.01)  # 纵坐标范围设置为 0 到 0.01

# 添加标题和标签
plt.title('Comparison of Remote Send and Local Move Times', fontsize=14)
plt.xlabel('Token Number', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.legend(fontsize=12)

# 动态计算文本位置
max_tokens = max(data['remote_send_tokens'] + data['local_move_tokens'])
max_time = max(data['remote_send_times'] + data['local_move_times'])

# 文本的 x 和 y 位置
text_x = max_tokens * 0.6  # 文本的 x 位置
text_y = max_time * 0.16    # 文本的 y 位置

# # 确保文本位置不会超出图表范围
# if text_x > max_tokens * 0.9:  # 如果 x 位置太靠右，调整到 90% 的位置
#     text_x = max_tokens * 0.9
# if text_y > max_time * 0.9:    # 如果 y 位置太靠上，调整到 90% 的位置
#     text_y = max_time * 0.9

# 在图表上添加平均值和总 token 数
plt.text(text_x, text_y, 
         f'Remote Send:\nAvg speed: {avg_remote_send_rate:.3f}Tokens/s\nTotal Tokens: {total_remote_send_tokens}',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
plt.text(text_x, text_y * 0.7, 
         f'Local Move:\nAvg speed: {avg_local_move_rate:.3f}Tokens/s\nTotal Tokens: {total_local_move_tokens}',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# 保存图表为文件
plt.savefig('/data/zzm/Bi-KV/remote_vs_local_move_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved to /data/zzm/Bi-KV/remote_vs_local_move_comparison.png")

# 显示图表（可选）
plt.show()