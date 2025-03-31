from config import *
import time
import threading

eviction_lock = threading.Lock()
# allocate_lock = threading.Lock()

class PageManager:
    def __init__(self, cache_size, page_size, pm_id = 0):
        self.pm_id = pm_id
        # self.buffer = cache_size*['']
        self.page_size = page_size
        self.num_pages = cache_size // page_size
        # set是不是不合适，换成队列？
        self.free_pages = set(range(self.num_pages))
        self.page_table = {}  # 键为item ID，值为{'pages': set, 'last_accessed': int}
        self.current_time = 0  # 模拟时间戳，用于LRU
        # print(f"初始空闲页数:{len(self.free_pages)}")

    def load_item(self, item_id, list_length):
        # 如果id已经存在则直接返回
        if item_id in self.page_table:
            return self.access_item(item_id), []
        """加载列表，必要时换出旧列表"""
        required_pages = (list_length + self.page_size - 1) // self.page_size
        if required_pages > self.num_pages:
            raise ValueError("列表过大，无法存入缓冲区")

        freed_ids = []

        # eviction_lock.acquire()
        # 检查是否需要换出
        if len(self.free_pages) < required_pages:
            freed_ids = self._perform_eviction(required_pages)

        # 分配新页
        allocated_pages = self._allocate_pages(required_pages)
        # 目前的情况看不加锁也没有问题
        # eviction_lock.release()
        self.page_table[item_id] = {
            'pages': allocated_pages,
            'last_accessed': self.current_time,
            'protected': 0
        }
        # for i in allocated_pages:
        #     for j in range(self._page_to_location(i), self._page_to_location(i)+self.page_size):
        #         self.buffer[j] = item_id
        self.current_time += 1
        return allocated_pages,freed_ids

    def access_item(self, item_id):
        """访问列表，更新最近使用时间，返回页号"""
        if item_id not in self.page_table:
            raise KeyError(f"列表 {item_id} 未加载")
        self.page_table[item_id]['last_accessed'] = self.current_time
        self.current_time += 1
        return self.page_table[item_id]['pages']

    def _perform_eviction(self, required_pages):
        """执行换出操作直到有足够空间"""
        # protected_items = sum(len(info['pages']) for info in self.page_table.values() if info['protected'] > 0)
        # paged_items = sum(len(info['pages']) for info in self.page_table.values())
        # print(f"触发换出操作，当前空闲页数:{len(self.free_pages)} 保护数量:{protected_items} 占用数量：{paged_items} 要求页数:{required_pages}")
        # 按LRU排序现有列表
        lru_entries = sorted(
            self.page_table.items(),
            key=lambda item: item[1]['last_accessed'],
        )

        freed_ids = []
        for item_id, info in lru_entries:
            # 如果保护则继续
            if self.page_table[item_id]['protected']!=0:
                continue
            # 换出该列表
            del self.page_table[item_id]
            self.free_pages.update(info['pages'])
            freed_ids.append(item_id)
            if DEBUG:
                print(f"[[{time.time()}]] 换出列表 {item_id}，释放页数 {len(info['pages'])} 空闲页数:{len(self.free_pages)}")
            if len(self.free_pages) >= required_pages:
                break

        if len(self.free_pages) < required_pages:
            # protected_items = sum(len(info['pages']) for info in self.page_table.values() if info['protected'] > 0)
            # print(f"[[{time.time()}]] 无法换出足够页面，当前页面:{list(self.page_table.keys())} 占用数量:{protected_items}")
            raise RuntimeError(f"Unable to free enough pages, even if all items are swapped out") 
        # 返回换出的页号
        return freed_ids

    def set_protected(self, item_id):
        """保护列表不被换出"""
        if item_id not in self.page_table:
            raise KeyError(f"列表 {item_id} 未加载")
        self.page_table[item_id]['protected'] += 1
        # print(f"[[{time.time()}]] 保护item {item_id} {self.page_table[item_id]['protected']}次")

    def remove_protected(self, item_id):
        """取消保护"""
        if item_id not in self.page_table:
            raise KeyError(f"列表 {item_id} 未加载")
        self.page_table[item_id]['protected'] -= 1
        # print(f"[[{time.time()}]] 取消保护item {item_id} {self.page_table[item_id]['protected']}次")

    def _allocate_pages(self, n):
        """分配指定数量的页（无需连续）"""
        if len(self.free_pages) < n:
            raise RuntimeError("内部错误：分配时页面不足")
        allocated = set()
        for _ in range(n):
            # 取第一个可用页（实现可优化）
            page = next(iter(self.free_pages))
            allocated.add(page)
            self.free_pages.remove(page)
        return list(allocated)
    
    def _page_to_location(self, page):
        """将页号转换为实际位置"""
        return page * self.page_size

    def get_loaded_lists(self):
        """返回当前加载的列表ID"""
        return list(self.page_table.keys())

class MultiPageManager:
    '''多个PageManager组成的KV缓存'''
    def __init__(self, cache_size, page_size, kvcahe_num):
        self.kvcahe_num = kvcahe_num
        self.page_managers = [PageManager(cache_size, page_size, pm_id=i) for i in range(kvcahe_num)]
        self.buffer_size = cache_size
        self.page_size = page_size
        self.num_pages = cache_size // page_size
        # 记录每个pm的缓存列表
        self.cached_ids = [set() for _ in range(kvcahe_num)]

    def load_item(self, item_id, list_length):
        """加载列表，必要时换出旧列表"""
        # NOTE 需要注意的是，应该先查询在不在缓存中，如果在缓存中则直接返回页号
        # 计算最大剩余页数
        max_page_num = max(len(pm.free_pages) for pm in self.page_managers)
        if max_page_num>(self.num_pages/10):
            # 选择剩余页数最多的pm
            target_pm = max(self.page_managers, key=lambda x: len(x.free_pages))
        else:
            # 随机选择一个pm 防止负载不均衡
            target_pm = random.choice(self.page_managers)
        target_pm_id = target_pm.pm_id
        allocated_pages, freed_ids = target_pm.load_item(item_id, list_length)
        # 记录已缓存的列表
        self.cached_ids[target_pm_id].add(item_id)
        # 删除已换出的列表
        for freed_id in freed_ids:
            self.cached_ids[target_pm_id].remove(freed_id)
        return target_pm_id, allocated_pages

    def access_item(self, item_id):
        """访问列表，更新最近使用时间，返回页号，否则返回None"""
        for idx, cached_id in enumerate(self.cached_ids):
            if item_id in cached_id:
                return idx,self.page_managers[idx].access_item(item_id)
        return None, None

# 示例用法
if __name__ == "__main__":
    # 初始化：32元素缓冲区，每页4元素 → 8页
    pm = PageManager(cache_size=32, page_size=4)

    # 加载列表A（5元素 → 2页）
    print(pm.load_item("A", 5))  # 例如输出 {0, 1}
    print("当前加载的列表:", pm.get_loaded_lists())
    # print("当前缓冲区状态:", pm.buffer)


    # 加载列表B（7元素 → 2页）
    print(pm.load_item("B", 7))  # 例如输出 {2, 3}
    print("当前加载的列表:", pm.get_loaded_lists())
    # print("当前缓冲区状态:", pm.buffer)


    # 访问列表A更新LRU时间
    pm.access_item("A")

    # 加载列表C（10元素 → 3页，需换出B）
    print(pm.load_item("C", 20))  # 例如输出 {4, 5, 6}
    print("当前加载的列表:", pm.get_loaded_lists())  # 应包含A和C
    # print("当前缓冲区状态:", pm.buffer)


    # 加载列表C（10元素 → 3页，需换出B）
    print(pm.load_item("D", 30))  # 例如输出 {4, 5, 6}
    print("当前加载的列表:", pm.get_loaded_lists())  # 应包含A和C
    # print("当前缓冲区状态:", pm.buffer)

    #测试 MultiPageManager
    mpm = MultiPageManager(32, 4, 2)
    mpm.load_item("A", 5)
    mpm.load_item("B", 7)
    mpm.load_item("C", 20)
    mpm.load_item("D", 30)
    print(mpm.access_item("A"))
    print(mpm.access_item("B"))
    print(mpm.access_item("C"))
    print(mpm.access_item("D"))
