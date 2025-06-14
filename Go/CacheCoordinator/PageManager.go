package coordinator

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	evictionLock sync.Mutex // 全局锁，模拟 Python 的 eviction_lock
)

// PageEntry 定义缓存项
type PageEntry struct {
	Pages        []int32 // 分配的页面集合
	LastAccessed int64   // 最近访问时间戳（Unix 纳秒）
	Protected    int     // 保护计数
	Priority     int     // 优先级（越小越优先）
}

// Item 优先队列中的元素
type Item struct {
	id           int32   // 缓存项ID
	priority     int     // 优先级
	lastAccessed int64   // 最近访问时间
	pages        []int32 // 页面列表
	index        int     // 堆中的索引
}

// PriorityQueue 优先队列
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// 最小堆：优先级小的优先；优先级相同时，时间戳小的优先
	if pq[i].priority == pq[j].priority {
		return pq[i].lastAccessed < pq[j].lastAccessed
	}
	return pq[i].priority < pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x any) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

// PageManager 单个页面管理器
type PageManager struct {
	pmID      int32
	pageSize  int
	numPages  int
	freePages map[int32]struct{}  // 使用 map 模拟 set
	pageTable map[int32]PageEntry // 缓存项
	mu        sync.Mutex
}

// NewPageManager 初始化 PageManager
func NewPageManager(cacheSize, pageSize int, pmID int32) *PageManager {
	numPages := cacheSize / pageSize
	pm := &PageManager{
		pmID:      pmID,
		pageSize:  pageSize,
		numPages:  numPages,
		freePages: make(map[int32]struct{}),
		pageTable: make(map[int32]PageEntry),
	}
	for i := range numPages {
		pm.freePages[int32(i)] = struct{}{}
	}
	log.Printf("初始空闲页数: %d\n", len(pm.freePages))
	return pm
}

// LoadItem 加载到缓存
func (pm *PageManager) LoadItem(itemID int32, itemLength int, weight int) ([]int32, []int32, time.Duration, time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 如果已存在，直接返回
	if _, ok := pm.pageTable[itemID]; ok {
		pages := pm.AccessItem(itemID)
		return pages, []int32{}, 0, 0
	}

	requiredPages := (itemLength + pm.pageSize - 1) / pm.pageSize
	if requiredPages > pm.numPages {
		panic(fmt.Sprintf("列表过大，无法存入缓冲区: %d > %d", requiredPages, pm.numPages))
	}
	var evtCost time.Duration
	var freedIDs []int32
	evictionLock.Lock()
	if len(pm.freePages) < requiredPages {
		start := time.Now()
		freedIDs = pm.performEviction(requiredPages)
		evtCost = time.Since(start)
	}
	start := time.Now()
	allocatedPages := pm.allocatePages(requiredPages)
	allCost := time.Since(start)
	evictionLock.Unlock()

	priority := 0
	if weight != 0 {
		// 用户初始优先级为2
		priority = 2
	}
	// 插入新项
	pm.pageTable[itemID] = PageEntry{
		Pages:        allocatedPages,
		LastAccessed: time.Now().UnixNano(),
		Protected:    0,
		Priority:     priority,
	}
	return allocatedPages, freedIDs, evtCost, allCost
}

// AccessItem 访问列表并更新时间戳
func (pm *PageManager) AccessItem(itemID int32) []int32 {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if entry, ok := pm.pageTable[itemID]; ok {
		entry.LastAccessed = time.Now().UnixNano()
		pm.pageTable[itemID] = entry
		return entry.Pages
	}
	panic(fmt.Sprintf("列表 %d 未加载", itemID))
}

// performEviction 执行优先队列换出
func (pm *PageManager) performEviction(requiredPages int) []int32 {
	pq := make(PriorityQueue, 0)
	heap.Init(&pq)

	// 将所有未保护的项加入优先队列
	for id, entry := range pm.pageTable {
		if entry.Protected != 0 {
			continue
		}
		if entry.Priority == 2 {
			continue
		}
		heap.Push(&pq, &Item{
			id:           id,
			priority:     entry.Priority,
			lastAccessed: entry.LastAccessed,
			pages:        entry.Pages,
		})
	}

	var freedIDs []int32
	for pq.Len() > 0 && len(pm.freePages) < requiredPages {
		item := heap.Pop(&pq).(*Item)
		delete(pm.pageTable, item.id)
		for _, page := range item.pages {
			pm.freePages[page] = struct{}{}
		}
		freedIDs = append(freedIDs, item.id)
		
		// log.Printf("[[%d]] 换出列表 %d，优先级 %d，释放页数 %d 空闲页数: %d\n",
		// 	time.Now().Unix(), item.id, item.priority, len(item.pages), len(pm.freePages))
		
	}

	if len(pm.freePages) < requiredPages {
		panic(fmt.Sprintf("无法换出足够页面，当前空闲页数: %d，要求页数: %d", len(pm.freePages), requiredPages))
	}
	return freedIDs
}

// SetProtected 设置保护状态
func (pm *PageManager) SetProtected(itemID int32) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if entry, ok := pm.pageTable[itemID]; ok {
		entry.Protected++
		pm.pageTable[itemID] = entry
		if DEBUG {
			log.Printf("[[%d]] 保护item %d %d次\n", time.Now().Unix(), itemID, entry.Protected)
		}
	} else {
		panic(fmt.Sprintf("列表 %d 未加载", itemID))
	}
}

// RemoveProtected 取消保护
func (pm *PageManager) RemoveProtected(itemID int32) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if entry, ok := pm.pageTable[itemID]; ok {
		entry.Protected--
		pm.pageTable[itemID] = entry
		if DEBUG {
			log.Printf("[[%d]] 取消保护item %d %d次\n", time.Now().Unix(), itemID, entry.Protected)
		}
	} else {
		panic(fmt.Sprintf("列表 %d 未加载", itemID))
	}
}

// allocatePages 分配页面
func (pm *PageManager) allocatePages(n int) []int32 {
	if len(pm.freePages) < n {
		panic(fmt.Sprintf("内部错误：分配时页面不足，剩余: %d，要求: %d", len(pm.freePages), n))
	}
	allocated := make(map[int32]struct{})
	for range n {
		for page := range pm.freePages {
			allocated[page] = struct{}{}
			delete(pm.freePages, page)
			break
		}
	}
	return setToList(allocated)
}

// GetLoadedLists 获取当前加载的列表 ID
func (pm *PageManager) GetLoadedLists() []int32 {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	var ids []int32
	for id := range pm.pageTable {
		ids = append(ids, id)
	}
	return ids
}

func (pm *PageManager) ShowFreePages() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	log.Printf("[PM %d] 空闲页数: %d\n", pm.pmID, len(pm.freePages))
}

// MultiPageManager 多个 PageManager 的集合
type MultiPageManager struct {
	kvcacheNum   int
	pageManagers []*PageManager
	bufferSize   int
	pageSize     int
	numPages     int
	p1MaxPages   int
	cachedIDs    []map[int32]struct{} // 每个 PM 的缓存 ID 集合
	p2Items      map[int32]struct{}   // 优先级为2的缓存项
	loadDuration time.Duration
	evtDuration  time.Duration
	allDuration  time.Duration
	gcInterval   time.Duration
	ttlInterval  time.Duration
	mu           sync.Mutex
	userCacheCount int 
}

// NewMultiPageManager 初始化 MultiPageManager
func NewMultiPageManager(cacheSize, pageSize, kvcacheNum, gcInterval, ttlInterval int) *MultiPageManager {
	mpm := &MultiPageManager{
		kvcacheNum:   kvcacheNum,
		pageManagers: make([]*PageManager, kvcacheNum),
		bufferSize:   cacheSize,
		pageSize:     pageSize,
		numPages:     cacheSize / pageSize,
		p1MaxPages:   cacheSize / pageSize / 4,
		cachedIDs:    make([]map[int32]struct{}, kvcacheNum),
		p2Items:      make(map[int32]struct{}),
		gcInterval:   time.Duration(gcInterval) * time.Second,
		ttlInterval:  time.Duration(ttlInterval) * time.Second,
		loadDuration: 0,
		evtDuration:  0,
		allDuration:  0,
		userCacheCount: 0, // 【新增】用户类型缓存计数
	}
	for i := range kvcacheNum {
		mpm.pageManagers[i] = NewPageManager(cacheSize, pageSize, int32(i))
		mpm.cachedIDs[i] = make(map[int32]struct{})
	}
	go mpm.ttlGc()
	return mpm
}

// LoadItem 加载到缓存
func (mpm *MultiPageManager) LoadItem(itemID int32, itemLength int, weight int32, itemType string) (int32, []int32) {
	mpm.mu.Lock()
	defer mpm.mu.Unlock()
	// 检查是否已在缓存中
	for idx, cached := range mpm.cachedIDs {
		if _, ok := cached[itemID]; ok {
			pages := mpm.pageManagers[idx].AccessItem(itemID)
			return int32(idx), pages
		}
	}

	// 选择目标 PM
	maxPageNum := 0
	for _, pm := range mpm.pageManagers {
		pm.mu.Lock()
		if len(pm.freePages) > maxPageNum {
			maxPageNum = len(pm.freePages)
		}
		pm.mu.Unlock()
	}

	var targetPM *PageManager
	if maxPageNum > mpm.numPages/10 {
		targetPM = mpm.pageManagers[0] // 默认第一个，后面更新
		for _, pm := range mpm.pageManagers {
			pm.mu.Lock()
			if len(pm.freePages) == maxPageNum {
				targetPM = pm
			}
			pm.mu.Unlock()
		}
	} else {
		targetPM = mpm.pageManagers[rand.Intn(mpm.kvcacheNum)] // 随机选择
	}

	targetPMID := targetPM.pmID
	start := time.Now()
	allocatedPages, freedIDs, evtCost, allCost := targetPM.LoadItem(itemID, itemLength, int(weight))
	mpm.loadDuration += time.Since(start)
	mpm.evtDuration += evtCost
	mpm.allDuration += allCost
	if itemType == "user cache" {
		mpm.p2Items[itemID] = struct{}{}
	}
	mpm.cachedIDs[targetPMID][itemID] = struct{}{}
	for _, freedID := range freedIDs {
		delete(mpm.cachedIDs[targetPMID], freedID)
	}
	mpm.checkAndAdjustP1Pages()
	return targetPMID, allocatedPages
}

func (mpm *MultiPageManager) LoadPreparedItem(itemID int32, itemLength int, weight int32, itemType string) {
	mpm.mu.Lock()
	defer mpm.mu.Unlock()
	// 检查是否已在缓存中
	for idx, cached := range mpm.cachedIDs {
		if _, ok := cached[itemID]; ok {
			mpm.pageManagers[idx].AccessItem(itemID)
			return
		}
	}

	// 选择目标 PM
	maxPageNum := 0
	for _, pm := range mpm.pageManagers {
		pm.mu.Lock()
		if len(pm.freePages) > maxPageNum {
			maxPageNum = len(pm.freePages)
		}
		pm.mu.Unlock()
	}

	var targetPM *PageManager
	if maxPageNum > mpm.numPages/10 {
		targetPM = mpm.pageManagers[0] // 默认第一个，后面更新
		for _, pm := range mpm.pageManagers {
			pm.mu.Lock()
			if len(pm.freePages) == maxPageNum {
				targetPM = pm
			}
			pm.mu.Unlock()
		}
	} else {
		targetPM = mpm.pageManagers[rand.Intn(mpm.kvcacheNum)] // 随机选择
	}

	targetPMID := targetPM.pmID
	start := time.Now()
	_, freedIDs, evtCost, allCost := targetPM.LoadItem(itemID, itemLength, int(weight))
	mpm.loadDuration += time.Since(start)
	mpm.evtDuration += evtCost
	mpm.allDuration += allCost
	mpm.cachedIDs[targetPMID][itemID] = struct{}{}
	for _, freedID := range freedIDs {
		delete(mpm.cachedIDs[targetPMID], freedID)
	}
}

// AccessItem 访问缓存中的列表
func (mpm *MultiPageManager) AccessItem(itemID int32) (int32, []int32) {
	mpm.mu.Lock()
	defer mpm.mu.Unlock()

	for idx, cached := range mpm.cachedIDs {
		if _, ok := cached[itemID]; ok {
			pages := mpm.pageManagers[idx].AccessItem(itemID)
			return int32(idx), pages
		}
	}
	return -1, nil // 表示未找到
}

// checkAndAdjustP1Pages 检查并调整优先级1的页数
func (mpm *MultiPageManager) checkAndAdjustP1Pages() {
	totalP1Pages := 0
	for _, pm := range mpm.pageManagers {
		pm.mu.Lock()
		for _, entry := range pm.pageTable {
			if entry.Priority == 1 {
				totalP1Pages += len(entry.Pages)
			}
		}
		pm.mu.Unlock()
	}

	if totalP1Pages > mpm.p1MaxPages {
		mpm.downgradeP1Items(totalP1Pages - mpm.p1MaxPages)
	}
}

// downgradeP1Items 将优先级1的项降级为优先级0
func (mpm *MultiPageManager) downgradeP1Items(excessPages int) {
	type p1Item struct {
		pmID         int32
		itemID       int32
		lastAccessed int64
		pages        int
	}
	var p1Items []p1Item

	// 收集所有优先级为1的项
	for idx, pm := range mpm.pageManagers {
		pm.mu.Lock()
		for id, entry := range pm.pageTable {
			if entry.Priority == 1 {
				p1Items = append(p1Items, p1Item{
					pmID:         int32(idx),
					itemID:       id,
					lastAccessed: entry.LastAccessed,
					pages:        len(entry.Pages),
				})
			}
		}
		pm.mu.Unlock()
	}

	// 按 lastAccessed 升序排序（LRU）
	sort.Slice(p1Items, func(i, j int) bool {
		return p1Items[i].lastAccessed < p1Items[j].lastAccessed
	})

	// 降级项直到释放足够的页数
	releasedPages := 0
	for _, item := range p1Items {
		if releasedPages >= excessPages {
			break
		}
		pm := mpm.pageManagers[item.pmID]
		pm.mu.Lock()
		if entry, ok := pm.pageTable[item.itemID]; ok && entry.Priority == 1 {
			entry.Priority = 0
			pm.pageTable[item.itemID] = entry
			releasedPages += item.pages
			if DEBUG {
				log.Printf("[[%d]] 降级item %d from priority 1 to 0\n", time.Now().Unix(), item.itemID)
			}
		}
		pm.mu.Unlock()
	}
}

func (mpm *MultiPageManager) ttlGc() {
	ticker := time.NewTicker(mpm.gcInterval)
	defer ticker.Stop()
	for range ticker.C {
		if DEBUG {
			log.Printf("[[%d]] 定期清理过期缓存\n", time.Now().Unix())
		}
		mpm.downgradeP2Items()
		// 每次GC后检查优先级1的容量
		mpm.mu.Lock()
		mpm.checkAndAdjustP1Pages()
		mpm.mu.Unlock()
	}
}

func (mpm *MultiPageManager) downgradeP2Items() {
	mpm.mu.Lock()
	defer mpm.mu.Unlock()
	currentTime := time.Now().UnixNano()
	ttl := mpm.ttlInterval.Nanoseconds()

	for itemID := range mpm.p2Items {
		for _, pm := range mpm.pageManagers {
			pm.mu.Lock()
			if entry, ok := pm.pageTable[itemID]; ok && entry.Priority == 2 {
				elapsed := currentTime - entry.LastAccessed
				if elapsed >= ttl {
					entry.Priority = 1
					pm.pageTable[itemID] = entry
					if DEBUG {
						log.Printf("[[%d]] 降级item %d from priority 2 to 1 due to TTL\n", time.Now().Unix(), itemID)
					}
					delete(mpm.p2Items, itemID)
				}
			}
			pm.mu.Unlock()
		}
	}
}

func (pm *MultiPageManager) ShowFreePages() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	for _, pm := range pm.pageManagers {
		pm.ShowFreePages()
	}
}

func (pm *MultiPageManager) ReadPreparedData(dataPath string, indexPath string) {
	file, _ := os.ReadFile(dataPath)
	var data map[string]int
	json.Unmarshal(file, &data)
	log.Printf("读取预加载数据: %d 条\n", len(data))
	indexFile, _ := os.ReadFile(indexPath)
	var userIDs []int
	json.Unmarshal(indexFile, &userIDs)
	log.Printf("读取预加载数据热度索引: %d 条\n", len(userIDs))
	maxNumPages := pm.kvcacheNum * pm.numPages
	totalPages := 0
	for _, k := range userIDs {
		var v int
		if strings.Contains(indexPath, "user") {
			v = data[fmt.Sprintf("%d", k+2000000)]
		} else {
			v = data[fmt.Sprintf("%d", k)]
		}
		pageNum := (v + pm.pageSize - 1) / pm.pageSize
		pm.LoadPreparedItem(int32(k), v, 0, "prepared")
		totalPages += pageNum
		if totalPages > maxNumPages {
			log.Printf("预加载数据超过最大页数限制: %d > %d\n", totalPages, maxNumPages)
			break
		}
	}
	pm.ShowFreePages()
}

func (mpm *MultiPageManager) ShowUserCacheCount() {
	log.Printf("当前用户类型缓存数量（ID > 2000000）: %d\n", len(mpm.p2Items))
}

// 调试开关
const DEBUG = false // 可根据需要设置为 true
