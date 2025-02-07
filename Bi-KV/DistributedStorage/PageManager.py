class PageManager:
    def __init__(self, buffer_size, page_size, pm_id = 0):
        self.pm_id = pm_id
        self.buffer = buffer_size*['']
        self.page_size = page_size
        self.num_pages = buffer_size // page_size
        self.free_pages = set(range(self.num_pages))
        self.page_table = {}  # 键为列表ID，值为{'pages': set, 'last_accessed': int}
        self.current_time = 0  # 模拟时间戳，用于LRU

    def load_list(self, list_id, list_length):
        """加载列表，必要时换出旧列表"""
        required_pages = (list_length + self.page_size - 1) // self.page_size
        if required_pages > self.num_pages:
            raise ValueError("列表过大，无法存入缓冲区")

        freed_ids = []
        # 检查是否需要换出
        if len(self.free_pages) < required_pages:
            freed_ids = self._perform_eviction(required_pages)

        # 分配新页
        allocated_pages = self._allocate_pages(required_pages)
        self.page_table[list_id] = {
            'pages': allocated_pages,
            'last_accessed': self.current_time
        }
        for i in allocated_pages:
            for j in range(self._page_to_location(i), self._page_to_location(i)+self.page_size):
                self.buffer[j] = list_id
        self.current_time += 1
        return allocated_pages,freed_ids

    def access_list(self, list_id) -> set:
        """访问列表，更新最近使用时间，返回页号"""
        if list_id not in self.page_table:
            raise KeyError(f"列表 {list_id} 未加载")
        self.page_table[list_id]['last_accessed'] = self.current_time
        self.current_time += 1
        return self.page_table[list_id]['pages']

    def _perform_eviction(self, required_pages):
        """执行换出操作直到有足够空间"""
        # 按LRU排序现有列表
        lru_entries = sorted(
            self.page_table.items(),
            key=lambda item: item[1]['last_accessed']
        )

        freed_ids = []
        for list_id, info in lru_entries:
            # 换出该列表
            self.free_pages.update(info['pages'])
            print(f"换出列表 {list_id}，页号 {info['pages']}")
            del self.page_table[list_id]
            freed_ids.append(list_id)
            if len(self.free_pages) >= required_pages:
                break

        if len(self.free_pages) < required_pages:
            raise RuntimeError("无法释放足够页面，即使换出所有列表")
        # 返回换出的页号
        return freed_ids

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
        return allocated
    
    def _page_to_location(self, page):
        """将页号转换为实际位置"""
        return page * self.page_size

    def get_loaded_lists(self):
        """返回当前加载的列表ID"""
        return list(self.page_table.keys())

class MultiPageManager:
    '''多个PageManager组成的KV缓存'''
    def __init__(self, buffer_size, page_size, kvcahe_num):
        self.kvcahe_num = kvcahe_num
        self.page_managers = [PageManager(buffer_size, page_size, i) for i in range(kvcahe_num)]
        self.buffer_size = buffer_size
        self.page_size = page_size
        # 记录每个pm的缓存列表
        self.cached_ids = [set() for _ in range(kvcahe_num)]

    def load_list(self, list_id, list_length):
        """加载列表，必要时换出旧列表"""
        # 选择剩余页数最多的pm
        target_pm = max(self.page_managers, key=lambda x: len(x.free_pages))
        target_pm_id = target_pm.pm_id
        _, freed_ids = target_pm.load_list(list_id, list_length)
        # 记录已缓存的列表
        self.cached_ids[target_pm_id].add(list_id)
        # 删除已换出的列表
        for id in freed_ids:
            self.cached_ids[target_pm_id].remove(id)

    def access_list(self, list_id):
        """访问列表，更新最近使用时间，返回页号，否则返回None"""
        for idx, cached_id in enumerate(self.cached_ids):
            if list_id in cached_id:
                return idx,self.page_managers[idx].access_list(list_id)
        return None, None

# 示例用法
if __name__ == "__main__":
    # 初始化：32元素缓冲区，每页4元素 → 8页
    pm = PageManager(buffer_size=32, page_size=4)

    # 加载列表A（5元素 → 2页）
    print(pm.load_list("A", 5))  # 例如输出 {0, 1}
    print("当前加载的列表:", pm.get_loaded_lists())
    print("当前缓冲区状态:", pm.buffer)


    # 加载列表B（7元素 → 2页）
    print(pm.load_list("B", 7))  # 例如输出 {2, 3}
    print("当前加载的列表:", pm.get_loaded_lists())
    print("当前缓冲区状态:", pm.buffer)


    # 访问列表A更新LRU时间
    pm.access_list("A")

    # 加载列表C（10元素 → 3页，需换出B）
    print(pm.load_list("C", 20))  # 例如输出 {4, 5, 6}
    print("当前加载的列表:", pm.get_loaded_lists())  # 应包含A和C
    print("当前缓冲区状态:", pm.buffer)


    # 加载列表C（10元素 → 3页，需换出B）
    print(pm.load_list("D", 30))  # 例如输出 {4, 5, 6}
    print("当前加载的列表:", pm.get_loaded_lists())  # 应包含A和C
    print("当前缓冲区状态:", pm.buffer)

    #测试 MultiPageManager
    mpm = MultiPageManager(32, 4, 2)
    mpm.load_list("A", 5)
    mpm.load_list("B", 7)
    mpm.load_list("C", 20)
    mpm.load_list("D", 30)
    print(mpm.access_list("A"))
    print(mpm.access_list("B"))
    print(mpm.access_list("C"))
    print(mpm.access_list("D"))
