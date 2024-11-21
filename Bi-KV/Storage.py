from collections import OrderedDict, defaultdict
from typing import Optional
import numpy as np

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict() # OrderedDict会根据放入元素的先后顺序进行排序
        self.capacity = capacity
        self.current_size = 0
    
    def get(self, key: int) -> Optional[np.ndarray]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key) #get:移动到队尾
        return self.cache[key]

    def put(self, key: int, value: np.ndarray):
        size = value.nbytes
        if key in self.cache:
            self.current_size -= self.cache[key].nbytes
            self.cache.move_to_end(key) #put:OrderedDict保证在队尾
        self.cache[key] = value
        self.current_size += size
        
        while self.current_size > self.capacity:
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.current_size -= oldest_value.nbytes
            print(f"LRU eviction: removed key {oldest_key}")

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.current_size = 0
        self.cache = {}  # key -> value
        self.freq_map = defaultdict(OrderedDict)  # freq -> OrderedDict(key)
        self.key_to_freq = {}  # key -> freq
        self.min_freq = 0  # Track the minimum frequency

    def get(self, key: int) -> Optional[np.ndarray]:
        if key not in self.cache:
            return None
        
        # Retrieve the value and increase its frequency
        value = self.cache[key]
        old_freq = self.key_to_freq[key]
        new_freq = old_freq + 1
        self.key_to_freq[key] = new_freq
        
        # Move the key from old frequency group to new frequency group
        del self.freq_map[old_freq][key]
        if not self.freq_map[old_freq]:  # Remove empty frequency list
            del self.freq_map[old_freq]
            if self.min_freq == old_freq:
                self.min_freq += 1
        self.freq_map[new_freq][key] = None  # Only storing the key, no value in freq_map
        
        # print(f"After get({key}): key_to_freq = {self.key_to_freq}")
        return value

    def put(self, key: int, value: np.ndarray):
        size = value.nbytes
        if self.capacity == 0 or size > self.capacity:
            return  # Cannot insert if item size exceeds total capacity or capacity is 0
        
        if key in self.cache:
            # Update the existing value in cache and increase frequency
            self.current_size -= self.cache[key].nbytes
            self.cache[key] = value  # Update value in cache
            self.current_size += size
            self.get(key)  # Increase frequency, only if key already exists
            return

        # Remove items until there is enough capacity
        while self.current_size + size > self.capacity:
            # Remove the least frequently used key
            evict_key, _ = self.freq_map[self.min_freq].popitem(last=False)
            self.current_size -= self.cache[evict_key].nbytes
            del self.cache[evict_key]
            del self.key_to_freq[evict_key]
            print(f"LFU eviction: removed key {evict_key} with freq {self.min_freq}")
            if not self.freq_map[self.min_freq]:
                del self.freq_map[self.min_freq]

        # Insert the new key and set its frequency to 1 without calling get
        self.cache[key] = value
        self.key_to_freq[key] = 1
        self.freq_map[1][key] = None  # Only storing the key, no value in freq_map
        self.min_freq = 1  # Reset min frequency to 1 for new keys
        self.current_size += size
        
        # print(f"After put({key}): key_to_freq = {self.key_to_freq}")



class KVCache:
    def __init__(self, total_capacity: int, user_cache_ratio: float, model_layers: int, vector_dim: int):
        self.total_capacity = total_capacity
        self.user_capacity = int(total_capacity * user_cache_ratio)
        self.item_capacity = total_capacity - self.user_capacity
        self.model_layers = model_layers
        self.vector_dim = vector_dim
        self.user_cache = LRUCache(self.user_capacity)
        self.item_cache = LFUCache(self.item_capacity)

    def compute_memory_size(self, sequence_length: int) -> int:
        return 4 * self.model_layers * self.vector_dim * sequence_length

    def get(self, cache_type: str, key: int) -> Optional[np.ndarray]:
        if cache_type == 'user':
            return self.user_cache.get(key)
        elif cache_type == 'item':
            return self.item_cache.get(key)
        else:
            raise ValueError("cache_type must be 'user' or 'item'")

    def put(self, cache_type: str, key: int, sequence_length: int):
        # 创建一个填充值为 key 的 [2, l, sequence_length, dim] 矩阵
        simulated_data = np.full((2, self.model_layers, sequence_length, self.vector_dim), fill_value=key, dtype=np.float16)
        
        if cache_type == 'user':
            self.user_cache.put(key, simulated_data)
        elif cache_type == 'item':
            self.item_cache.put(key, simulated_data)
        else:
            raise ValueError("cache_type must be 'user' or 'item'")

# 测试代码
def test_cache_eviction():
    # 增大缓存容量来测试换出
    total_capacity = 200000  # 较大的容量来容纳更多条目，但也可以触发换出
    user_cache_ratio = 0.5
    model_layers = 2
    vector_dim = 32
    cache = KVCache(total_capacity=total_capacity, user_cache_ratio=user_cache_ratio, model_layers=model_layers, vector_dim=vector_dim)

    sequence_length = 64  # 测试的序列长度

    # 插入用户数据到缓存，超过容量以触发换出
    print("Inserting user cache data to trigger LRU eviction:")
    for i in range(1, 12):
        cache.put(cache_type='user', key=i, sequence_length=sequence_length)
        print(f"Inserted user key={i}, sequence_length={sequence_length}")
        user_data = cache.get(cache_type='user', key=i)
        print(f"Retrieved user data for key={i}, shape={user_data.shape if user_data is not None else 'Cache Miss'}")
        print("Current user cache size:", cache.user_cache.current_size)
        print("Current user cache keys:", list(cache.user_cache.cache.keys()))
        print("-" * 50)
    user_data = cache.get(cache_type='user', key=14)
    print(f"Retrieved user data for key={14}, shape={user_data.shape if user_data is not None else 'Cache Miss'}")
    # 重复访问用户缓存中的特定键，以提升其在 LRU 中的优先级
    print("\nAccessing user cache key=6 and key=7 and key=8 multiple times to update LRU order:")
    for i in [6, 7,8]:  # 访问键 6 和 7,8
        user_data = cache.get(cache_type='user', key=i)
        print(f"Accessed user data for key={i}, shape={user_data.shape if user_data is not None else 'Cache Miss'}")
    print("Current user cache keys after accessing key=6 and key=7 and key=8:", list(cache.user_cache.cache.keys()))
    print("-" * 50)

    # 插入新的数据项以触发 LRU 换出
    cache.put(cache_type='user', key=12, sequence_length=sequence_length)
    print(f"Inserted user key=12, sequence_length={sequence_length}")
    print("Current user cache size:", cache.user_cache.current_size)
    print("Current user cache keys after inserting key=12:", list(cache.user_cache.cache.keys()))
    print("-" * 50)

    # 插入项目数据到缓存，超过容量以触发 LFU 换出
    print("\nInserting item cache data to trigger LFU eviction:")
    for i in range(1, 12):
        cache.put(cache_type='item', key=i + 100, sequence_length=sequence_length)
        print(f"Inserted item key={i + 100}, sequence_length={sequence_length}")
        item_data = cache.get(cache_type='item', key=i + 100)
        print(f"got item key={i + 100}, freq ++")
        print(f"Retrieved item data for key={i + 100}, shape={item_data.shape if item_data is not None else 'Cache Miss'}")
        print("Current item cache size:", cache.item_cache.current_size)
        print("Current item cache keys:", list(cache.item_cache.cache.keys()))
        print("-" * 50)

    # 重复访问项目缓存中的某个键以提升其频率（LFU）
    print("\nAccessing item cache key=106,106,107 multiple times to increase frequency:")
    for i in [106, 106, 107]:
        result = cache.get(cache_type='item',key=i)
        if result is not None:
            print(f"Accessed item data for key={i}, shape={result.shape}")
    print("-" * 50)

    # 插入更多数据以触发 LFU 换出
    print("\nInserting additional item data to trigger LFU eviction:")
    for i in range(12, 15):
        cache.put(cache_type='item', key=i + 100, sequence_length=sequence_length)
        print(f"Inserted item key={i + 100}, sequence_length={sequence_length}")
        print("Current item cache size:", cache.item_cache.current_size)
        print(f"Current item cache keys after inserting key={i + 100}:", list(cache.item_cache.cache.keys()))
        print("-" * 50)

# 运行测试
# test_cache_eviction()

