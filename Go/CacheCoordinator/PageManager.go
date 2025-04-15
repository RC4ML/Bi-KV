package coordinator

import (
	"container/heap"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"sync"
	"time"
)

var (
	evictionLock sync.Mutex // 全局锁，模拟 Python 的 eviction_lock
)

// PageEntry 定义缓存项
type PageEntry struct {
	Pages        []int32 // 分配的页面集合
	LastAccessed int64   // 最近访问时间戳
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
	pmID        int32
	pageSize    int
	numPages    int
	freePages   map[int32]struct{}  // 使用 map 模拟 set
	pageTable   map[int32]PageEntry // 缓存项
	weights     map[int32]int       // 缓存项的权重
	currentTime int64               // 用于时间戳
	p0Scale     float64
	p1Scale     float64
	mu          sync.Mutex
}

// NewPageManager 初始化 PageManager
func NewPageManager(cacheSize, pageSize int, pmID int32) *PageManager {
	numPages := cacheSize / pageSize
	pm := &PageManager{
		pmID:        pmID,
		pageSize:    pageSize,
		numPages:    numPages,
		freePages:   make(map[int32]struct{}),
		pageTable:   make(map[int32]PageEntry),
		weights:     make(map[int32]int),
		currentTime: 0,
		p0Scale:     0.7,
		p1Scale:     0.8,
	}
	for i := range numPages {
		pm.freePages[int32(i)] = struct{}{}
	}
	log.Printf("初始空闲页数: %d\n", len(pm.freePages))
	return pm
}

func (pm *PageManager) SetScale(p0scale, p1scale float64) {
	pm.p0Scale = p0scale
	pm.p1Scale = p1scale
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

	// 记录权重
	pm.weights[itemID] = weight
	priority := 0
	if weight != 0 {
		priority = 1
	}
	// 插入新项
	pm.pageTable[itemID] = PageEntry{
		Pages:        allocatedPages,
		LastAccessed: pm.currentTime,
		Protected:    0,
		Priority:     priority, // 优先级在mpm.computePriorities中设置
	}
	pm.currentTime++
	return allocatedPages, freedIDs, evtCost, allCost
}

// AccessItem 访问列表并更新时间戳
func (pm *PageManager) AccessItem(itemID int32) []int32 {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if entry, ok := pm.pageTable[itemID]; ok {
		entry.LastAccessed = pm.currentTime
		pm.currentTime++
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
		if entry.Priority != 0 {
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
		delete(pm.weights, item.id) // 删除权重记录
		for _, page := range item.pages {
			pm.freePages[page] = struct{}{}
		}
		freedIDs = append(freedIDs, item.id)
		if DEBUG {
			log.Printf("[[%d]] 换出列表 %d，优先级 %d，释放页数 %d 空闲页数: %d\n",
				time.Now().Unix(), item.id, item.priority, len(item.pages), len(pm.freePages))
		}
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

// MultiPageManager 多个 PageManager 的集合
type MultiPageManager struct {
	kvcacheNum   int
	pageManagers []*PageManager
	bufferSize   int
	pageSize     int
	numPages     int
	cachedIDs    []map[int32]struct{} // 每个 PM 的缓存 ID 集合
	loadDuration time.Duration
	evtDuration  time.Duration
	allDuration  time.Duration
	p0Scale      float64
	p1Scale      float64
	mu           sync.Mutex
}

// NewMultiPageManager 初始化 MultiPageManager
func NewMultiPageManager(cacheSize, pageSize, kvcacheNum int, p0Scale, p1Scale float64) *MultiPageManager {
	mpm := &MultiPageManager{
		kvcacheNum:   kvcacheNum,
		pageManagers: make([]*PageManager, kvcacheNum),
		bufferSize:   cacheSize,
		pageSize:     pageSize,
		numPages:     cacheSize / pageSize,
		cachedIDs:    make([]map[int32]struct{}, kvcacheNum),
		loadDuration: 0,
		evtDuration:  0,
		allDuration:  0,
		p0Scale:      p0Scale,
		p1Scale:      p1Scale,
	}
	for i := range kvcacheNum {
		mpm.pageManagers[i] = NewPageManager(cacheSize, pageSize, int32(i))
		mpm.pageManagers[i].SetScale(p0Scale, p1Scale)
		mpm.cachedIDs[i] = make(map[int32]struct{})
	}
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
	if itemType == "user cache" {
		mpm.ComputePriorities()
	}
	mpm.loadDuration += time.Since(start)
	mpm.evtDuration += evtCost
	mpm.allDuration += allCost
	mpm.cachedIDs[targetPMID][itemID] = struct{}{}
	for _, freedID := range freedIDs {
		delete(mpm.cachedIDs[targetPMID], freedID)
	}
	return targetPMID, allocatedPages
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

func (mpm *MultiPageManager) ComputePriorities() {
	type weightEntry struct {
		id     int32
		weight int
		pmID   int
	}
	var weightList []weightEntry
	for pmID, pm := range mpm.pageManagers {
		for id, weight := range pm.weights {
			// 如果是商品不用排
			// TODO 看一下初始化priority
			if weight == 0 {
				continue
			}
			weightList = append(weightList, weightEntry{id, weight, pmID})
		}
	}
	if len(weightList) == 0 {
		return
	}

	// 按优先级排序
	sort.Slice(weightList, func(i, j int) bool {
		return weightList[i].weight < weightList[j].weight
	})

	// 计算百分位阈值
	n := len(weightList)
	threshold70 := int(float64(n) * mpm.p0Scale)
	threshold80 := int(float64(n) * mpm.p1Scale)

	// 分配优先级
	for i, entry := range weightList {
		priority := 0
		if i >= threshold80 {
			priority = 2
		} else if i >= threshold70 {
			priority = 1
		}
		// 更新 pageTable 中的优先级
		if pe, ok := mpm.pageManagers[entry.pmID].pageTable[entry.id]; ok {
			pe.Priority = priority
			mpm.pageManagers[entry.pmID].pageTable[entry.id] = pe
		}
	}
}

// 调试开关
const DEBUG = false // 可根据需要设置为 true
