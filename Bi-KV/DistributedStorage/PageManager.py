class PageManager:
    def __init__(self, buffer_size, page_size):
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

        # 检查是否需要换出
        if len(self.free_pages) < required_pages:
            self._perform_eviction(required_pages)

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
        return allocated_pages

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

        freed_pages = 0
        for list_id, info in lru_entries:
            # 换出该列表
            self.free_pages.update(info['pages'])
            print(f"换出列表 {list_id}，页号 {info['pages']}")
            del self.page_table[list_id]
            freed_pages += len(info['pages'])
            if len(self.free_pages) >= required_pages:
                break

        if len(self.free_pages) < required_pages:
            raise RuntimeError("无法释放足够页面，即使换出所有列表")

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
