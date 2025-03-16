package coordinator

import (
	"fmt"
	"log"
	"math/rand"
	"slices"
	"sort"
	"sync"
	"time"
)

var (
	evictionLock sync.Mutex // 全局锁，模拟 Python 的 eviction_lock
)

// PageManager 单个页面管理器
type PageManager struct {
	pmID        int32
	pageSize    int
	numPages    int
	freePages   []int32 // 有序空闲页列表
	pageTable   map[int32]PageEntry
	currentTime int64
	mu          sync.Mutex
}

type PageEntry struct {
	Pages        []int32 // 改为有序slice
	LastAccessed int64
	Protected    int
}

// NewPageManager 初始化 PageManager
func NewPageManager(cacheSize, pageSize int, pmID int32) *PageManager {
	numPages := cacheSize / pageSize
	pm := &PageManager{
		pmID:        pmID,
		pageSize:    pageSize,
		numPages:    numPages,
		freePages:   make([]int32, numPages),
		pageTable:   make(map[int32]PageEntry),
		currentTime: 0,
	}
	for i := range numPages {
		pm.freePages[i] = int32(i)
	}
	log.Printf("初始空闲页数: %d\n", len(pm.freePages))
	return pm
}

// LoadItem 加载列表到缓存
func (pm *PageManager) LoadItem(itemID int32, listLength int) ([]int32, []int32) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if entry, ok := pm.pageTable[itemID]; ok {
		entry.LastAccessed = pm.currentTime
		pm.pageTable[itemID] = entry
		pm.currentTime++
		return entry.Pages, nil
	}

	requiredPages := (listLength + pm.pageSize - 1) / pm.pageSize
	if requiredPages > pm.numPages {
		panic(fmt.Sprintf("列表过大，无法存入缓冲区: %d > %d", requiredPages, pm.numPages))
	}

	var freedIDs []int32
	evictionLock.Lock()
	if len(pm.freePages) < requiredPages {
		freedIDs = pm.performEviction(requiredPages)
	}
	allocatedPages := pm.allocatePages(requiredPages)
	evictionLock.Unlock()

	pm.pageTable[itemID] = PageEntry{
		Pages:        allocatedPages,
		LastAccessed: pm.currentTime,
		Protected:    0,
	}
	pm.currentTime++
	return allocatedPages, freedIDs
}

// AccessItem 访问列表并更新时间戳
func (pm *PageManager) AccessItem(itemID int32) []int32 {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	entry, ok := pm.pageTable[itemID]
	if !ok {
		panic(fmt.Sprintf("列表 %d 未加载", itemID))
	}
	entry.LastAccessed = pm.currentTime
	pm.pageTable[itemID] = entry
	pm.currentTime++
	return entry.Pages
}

// performEviction 执行 LRU 换出
func (pm *PageManager) performEviction(requiredPages int) []int32 {
	type entry struct {
		id           int32
		lastAccessed int64
		pages        []int32
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
		pm.returnPages(entry.pages)
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

// returnPages 将页面按顺序插入到freePages中
func (pm *PageManager) returnPages(pages []int32) {
	// 将释放的页面排序
	slices.Sort(pages)
	// 合并到freePages中保持有序
	pm.freePages = mergeSortedSlices(pm.freePages, pages)
}

// 合并两个有序slice
func mergeSortedSlices(a, b []int32) []int32 {
	result := make([]int32, 0, len(a)+len(b))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		if a[i] < b[j] {
			result = append(result, a[i])
			i++
		} else {
			result = append(result, b[j])
			j++
		}
	}
	result = append(result, a[i:]...)
	result = append(result, b[j:]...)
	return result
}

// allocatePages 分配连续的页面
func (pm *PageManager) allocatePages(n int) []int32 {
	if len(pm.freePages) < n {
		panic(fmt.Sprintf("内部错误：分配时页面不足，剩余: %d，要求: %d", len(pm.freePages), n))
	}
	allocated := make([]int32, n)
	copy(allocated, pm.freePages[:n])
	pm.freePages = pm.freePages[n:]
	return allocated
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
	}
	for i := range kvcacheNum {
		mpm.pageManagers[i] = NewPageManager(cacheSize, pageSize, int32(i))
		mpm.cachedIDs[i] = make(map[int32]struct{})
	}
	return mpm
}

// LoadItem 加载列表到缓存
func (mpm *MultiPageManager) LoadItem(itemID int32, listLength int) (int32, []int32) {
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
	allocatedPages, freedIDs := targetPM.LoadItem(itemID, listLength)
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

// 调试开关
const DEBUG = false // 可根据需要设置为 true
