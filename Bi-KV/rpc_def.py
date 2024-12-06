# 模拟宏定义，创建一个类型到函数的映射
typefunc_map = {
    'scheduler': 'LLMScheduler',
    'coordinator': 'CacheScheduler',
    'inferworker': 'Worker',
    'kvcache': 'KVCache'
}
KVCACHE_NUM =4
WORKER_NUM=4
WORKER_offset = 2
KVCACHE_offset =6