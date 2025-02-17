import os

# 可通过环境变量指定，若未设置则使用默认值
KVCACHE_NUM = int(os.environ.get("KVCACHE_NUM", "4"))
WORKER_NUM = int(os.environ.get("WORKER_NUM", "4"))

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
WORKER_offset = 2  # LLMScheduler和CacheCoordinator结束后是worker的起始
KVCACHE_offset = 3  # 一个worker对应一个kvcache，所以kvcache的rank从worker的下一个开始

# 模拟宏定义，创建一个类型到函数的映射(如果有需要在其他地方动态创建实例时使用)
typefunc_map = {
    'scheduler': 'LLMScheduler',
    'coordinator': 'CacheCoordinator',
    'inferworker': 'Worker',
    'kvcache': 'KVCache'
}
def glo_CacheRank(cacherank):
    a=1
def get_process_info(rank):
    """
    根据全局 rank 返回进程类型和该类型下的索引。

    Args:
        rank (int): 全局 rank。
        process_types (list): 进程类型及其数量的有序列表。

    Returns:
        tuple: (process_type, type_index)
    """
    # current_rank = 0
    # for process_type, count in process_types:
    #     if current_rank + count > rank:
    #         type_index = rank - current_rank
    #         return process_type, type_index
    #     current_rank += count
    # raise ValueError(f"Rank {rank} 超出定义的进程类型范围。")
    if rank == 0 :   # rank 0 is LLMScheduler
        return 'LLMScheduler',0
    if rank ==1 :    # rank 1 is CacheCoordinator
        return 'CacheCoordinator',0
    if rank % 2 ==0:  # rank 2,4,6,8.... is Worker
        return 'Worker',int(rank/2)-1
    else:             # rank 3,5,7,9.... is kvcache
        return 'KVCache' , int(rank/2)-1


