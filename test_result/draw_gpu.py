import matplotlib.pyplot as plt

# 测试数据
sizes = [200, 500, 1000, 1200, 1300]  # MB
average_rates = [455.50, 720.61, 1175.98, 1116.19, 1257.87]  # MB/s

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(sizes, average_rates, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2c7fb8')

# 添加标签和标题
plt.title('GPU Average Shared MEM Transfer Rate vs Test Size', fontsize=14, pad=20)
plt.xlabel('Test Size (MB)', fontsize=12)
plt.ylabel('Average Rate (MB/s)', fontsize=12)

# 设置刻度
plt.xticks(sizes)
plt.yticks(range(400, 800, 100))

# 添加数据标签
for x, y in zip(sizes, average_rates):
    plt.text(x, y+10, f'{y:.1f}', ha='center', va='bottom', fontsize=10)

# 网格设置
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.tight_layout()
# 保存图片（必须先于plt.show()调用）
plt.savefig('gpu_shared_mem_transfer_rate_analysis.png',  # 保存路径
            dpi=300,                      # 分辨率（默认100）
            bbox_inches='tight',          # 去除多余白边
            transparent=False,            # 透明背景
            facecolor='white')            # 背景颜色
plt.show()