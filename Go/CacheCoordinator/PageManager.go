package coordinator

import (
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

// PageManager 单个页面管理器
// TODO 验证用map模拟set是否顺序
type PageManager struct {
	pmID        int32
	pageSize    int
	numPages    int
	freePages   map[int32]struct{} // 使用 map 模拟 set
	pageTable   map[int32]PageEntry
	currentTime int64 // 用于 LRU 的时间戳
	mu          sync.Mutex
}

type PageEntry struct {
	Pages        map[int32]struct{} // 分配的页面集合
	LastAccessed int64              // 最近访问时间戳
	Protected    int                // 保护计数
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
		currentTime: 0,
	}
	for i := range numPages {
		pm.freePages[int32(i)] = struct{}{}
	}
	log.Printf("初始空闲页数: %d\n", len(pm.freePages))
	return pm
}

// LoadItem 加载列表到缓存
func (pm *PageManager) LoadItem(itemID int32, listLength int) (map[int32]struct{}, []int32, time.Duration, time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 如果已存在，直接返回
	if _, ok := pm.pageTable[itemID]; ok {
		pages := pm.AccessItem(itemID)
		return pages, []int32{}, 0, 0
	}

	requiredPages := (listLength + pm.pageSize - 1) / pm.pageSize
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

	pm.pageTable[itemID] = PageEntry{
		Pages:        allocatedPages,
		LastAccessed: pm.currentTime,
		Protected:    0,
	}
	pm.currentTime++
	return allocatedPages, freedIDs, evtCost, allCost
}

// AccessItem 访问列表并更新时间戳
func (pm *PageManager) AccessItem(itemID int32) map[int32]struct{} {
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

// performEviction 执行 LRU 换出
func (pm *PageManager) performEviction(requiredPages int) []int32 {
	type entry struct {
		id           int32
		lastAccessed int64
		pages        map[int32]struct{}
	}
	var lruEntries []entry
	for id, info := range pm.pageTable {
		lruEntries = append(lruEntries, entry{id, info.LastAccessed, info.Pages})
	}
	sort.Slice(lruEntries, func(i, j int) bool {
		return lruEntries[i].lastAccessed < lruEntries[j].lastAccessed
	})

	var freedIDs []int32
	for _, entry := range lruEntries {
		if pm.pageTable[entry.id].Protected != 0 {
			continue
		}
		delete(pm.pageTable, entry.id)
		for page := range entry.pages {
			pm.freePages[page] = struct{}{}
		}
		freedIDs = append(freedIDs, entry.id)
		if DEBUG {
			log.Printf("[[%d]] 换出列表 %d，释放页数 %d 空闲页数: %d\n",
				time.Now().Unix(), entry.id, len(entry.pages), len(pm.freePages))
		}
		if len(pm.freePages) >= requiredPages {
			break
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
func (pm *PageManager) allocatePages(n int) map[int32]struct{} {
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
	return allocated
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
	mu           sync.Mutex
}

// NewMultiPageManager 初始化 MultiPageManager
func NewMultiPageManager(cacheSize, pageSize, kvcacheNum int) *MultiPageManager {
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
	}
	for i := range kvcacheNum {
		mpm.pageManagers[i] = NewPageManager(cacheSize, pageSize, int32(i))
		mpm.cachedIDs[i] = make(map[int32]struct{})
	}
	return mpm
}

// LoadItem 加载列表到缓存
func (mpm *MultiPageManager) LoadItem(itemID int32, listLength int) (int32, map[int32]struct{}) {
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
	allocatedPages, freedIDs, evtCost, allCost := targetPM.LoadItem(itemID, listLength)
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
func (mpm *MultiPageManager) AccessItem(itemID int32) (int32, map[int32]struct{}) {
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

// 调试开关
const DEBUG = false // 可根据需要设置为 true
