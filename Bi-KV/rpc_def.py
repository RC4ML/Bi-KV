import os

# 可通过环境变量指定，若未设置则使用默认值
KVCACHE_NUM = int(os.environ.get("KVCACHE_NUM", "2"))
WORKER_NUM = int(os.environ.get("WORKER_NUM", "1"))

# 定义进程类型及其数量
PROCESS_TYPES = [
    ('scheduler', 1),
    ('coordinator', 1),
    ('inferworker', WORKER_NUM),
    ('kvcache', KVCACHE_NUM),
]

# 根据PROCESS_TYPES计算偏移量
# scheduler rank = 0
# coordinator rank = 1
# worker从 rank=2 开始
WORKER_offset = 1 + 1  # scheduler(0) + coordinator(1) = 2
KVCACHE_offset = WORKER_offset + WORKER_NUM  # worker结束后是kvcache的起始

# 模拟宏定义，创建一个类型到函数的映射(如果有需要在其他地方动态创建实例时使用)
typefunc_map = {
    'scheduler': 'LLMScheduler',
    'coordinator': 'CacheCoordinator',
    'inferworker': 'Worker',
    'kvcache': 'KVCache'
}

def get_process_info(rank, process_types=PROCESS_TYPES):
    """
    根据全局 rank 返回进程类型和该类型下的索引。

    Args:
        rank (int): 全局 rank。
        process_types (list): 进程类型及其数量的有序列表。

    Returns:
        tuple: (process_type, type_index)
    """
    current_rank = 0
    for process_type, count in process_types:
        if current_rank + count > rank:
            type_index = rank - current_rank
            return process_type, type_index
        current_rank += count
    raise ValueError(f"Rank {rank} 超出定义的进程类型范围。")

