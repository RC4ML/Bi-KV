from typing import Dict, Tuple
from collections import OrderedDict, defaultdict
from typing import Optional
import numpy as np

class LRUCache:
    def __init__(self, capacity: int, kvcache_num: int):
        self.cache = OrderedDict() # OrderedDict会根据放入元素的先后顺序进行排序
        self.capacity = capacity
        self.kvcache_num = kvcache_num
        # 容量按照token_num管理
        self.current_size = 0
    
    def get(self, task_info: Dict) -> Tuple[int,int]:
        cache_id = task_info['id']
        if cache_id not in self.cache:
            return None
        self.cache.move_to_end(cache_id) #get:移动到队尾
        return self.cache[cache_id]

    def put(self, task_info: Dict):
        '''put目前只看task_info的id和token_num'''
        cache_id = task_info['id']
        size = task_info['token_num']
        if cache_id in self.cache:
            self.current_size -= self.cache[cache_id][1]
            self.cache.move_to_end(cache_id) #put:OrderedDict保证在队尾
        kvcache_id = self.strategy(cache_id)
        self.cache[cache_id] = (kvcache_id,size)
        self.current_size += size
        
        while self.current_size > self.capacity:
            # print(f"[LRU] eviction: current_size {self.current_size} > capacity {self.capacity}")
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.current_size -= oldest_value[1]
            # print(f"[LRU] eviction: removed key {oldest_key}")

    def strategy(self, req_id: int) -> int:
        return req_id % self.kvcache_num

