package coordinator

import (
	"math/rand"
	"sync"
	"testing"
	"time"
)

func TestPageManager_LoadAndAccessItem(t *testing.T) {
	cacheSize := 1000
	pageSize := 100
	pmID := int32(0)
	pm := NewPageManager(cacheSize, pageSize, pmID)

	// Test loading an item
	itemID := int32(1)
	listLength := 250 // Requires 3 pages
	priority := 10
	pages, freedIDs, _, _ := pm.LoadItem(itemID, listLength, priority)

	if len(pages) != 3 {
		t.Errorf("Expected 3 pages allocated, got %d", len(pages))
	}
	if len(freedIDs) != 0 {
		t.Errorf("Expected no freed IDs, got %d", len(freedIDs))
	}

	// Test accessing the loaded item
	accessedPages := pm.AccessItem(itemID)
	if len(accessedPages) != len(pages) {
		t.Errorf("AccessItem returned %d pages, expected %d", len(accessedPages), len(pages))
	}
	for i, page := range accessedPages {
		if page != pages[i] {
			t.Errorf("AccessItem pages mismatch at index %d: got %d, expected %d", i, page, pages[i])
		}
	}

	// Test accessing non-existent item
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic when accessing non-existent item")
		}
	}()
	pm.AccessItem(int32(999))
}

func TestPageManager_Eviction(t *testing.T) {
	cacheSize := 500
	pageSize := 100
	pmID := int32(0)
	pm := NewPageManager(cacheSize, pageSize, pmID)

	// Load items to fill cache
	for i := range 5 {
		_, _, _, _ = pm.LoadItem(int32(i), 100, 10+i)
	}

	// Load another item to trigger eviction
	itemID := int32(6)
	listLength := 200 // Requires 2 pages
	_, freedIDs, _, _ := pm.LoadItem(itemID, listLength, 5)

	if len(freedIDs) == 0 {
		t.Errorf("Expected some items to be evicted")
	}

	// Verify evicted items are no longer in pageTable
	for _, id := range freedIDs {
		if _, ok := pm.pageTable[id]; ok {
			t.Errorf("Evicted item %d still in pageTable", id)
		}
	}
}

func TestPageManager_ProtectedItems(t *testing.T) {
	cacheSize := 500
	pageSize := 100
	pmID := int32(0)
	pm := NewPageManager(cacheSize, pageSize, pmID)

	// Load an item
	itemID := int32(1)
	listLength := 100
	pm.LoadItem(itemID, listLength, 10)

	// Protect the item
	pm.SetProtected(itemID)
	if entry, ok := pm.pageTable[itemID]; !ok || entry.Protected != 1 {
		t.Errorf("Expected item %d to be protected with count 1", itemID)
	}

	// Try to evict by loading more items
	for i := 2; i <= 6; i++ {
		pm.LoadItem(int32(i), 100, 10)
	}

	// Verify protected item is still present
	if _, ok := pm.pageTable[itemID]; !ok {
		t.Errorf("Protected item %d was evicted", itemID)
	}

	// Unprotect and verify
	pm.RemoveProtected(itemID)
	if entry, ok := pm.pageTable[itemID]; !ok || entry.Protected != 0 {
		t.Errorf("Expected item %d to be unprotected", itemID)
	}
}

func TestMultiPageManager_LoadAndAccess(t *testing.T) {
	cacheSize := 1000
	pageSize := 100
	kvcacheNum := 3
	p0Scale := 0.7
	p1Scale := 0.8
	mpm := NewMultiPageManager(cacheSize, pageSize, kvcacheNum, p0Scale, p1Scale)

	// Load an item
	itemID := int32(1)
	listLength := 150
	priority := int32(10)
	pmID, pages := mpm.LoadItem(itemID, listLength, priority)

	if pmID < 0 || int(pmID) >= kvcacheNum {
		t.Errorf("Invalid pmID returned: %d", pmID)
	}
	if len(pages) != 2 {
		t.Errorf("Expected 2 pages allocated, got %d", len(pages))
	}

	// Access the item
	accessedPMID, accessedPages := mpm.AccessItem(itemID)
	if accessedPMID != pmID {
		t.Errorf("AccessItem returned pmID %d, expected %d", accessedPMID, pmID)
	}
	if len(accessedPages) != len(pages) {
		t.Errorf("AccessItem returned %d pages, expected %d", len(accessedPages), len(pages))
	}
}

func TestMultiPageManager_ConcurrentAccess(t *testing.T) {
	cacheSize := 1000
	pageSize := 100
	kvcacheNum := 2
	p0Scale := 0.7
	p1Scale := 0.8
	mpm := NewMultiPageManager(cacheSize, pageSize, kvcacheNum, p0Scale, p1Scale)

	var wg sync.WaitGroup
	numGoroutines := 10
	itemID := int32(1)
	listLength := 100
	priority := int32(10)

	// Load item initially
	mpm.LoadItem(itemID, listLength, priority)

	// Concurrently access the item
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pmID, pages := mpm.AccessItem(itemID)
			if pmID == -1 || len(pages) == 0 {
				t.Errorf("Failed to access item %d", itemID)
			}
		}()
	}

	wg.Wait()
}

func TestPageManager_PriorityAssignment(t *testing.T) {
	cacheSize := 1000
	pageSize := 100
	pmID := int32(0)
	pm := NewPageManager(cacheSize, pageSize, pmID)

	// Load items with different priorities
	items := []struct {
		id       int32
		length   int
		priority int
	}{
		{1, 100, 5},
		{2, 100, 10},
		{3, 100, 3},
		{4, 100, 8},
		{5, 100, 1},
	}

	for _, item := range items {
		pm.LoadItem(item.id, item.length, item.priority)
	}

	// Verify priority assignment
	for _, item := range items {
		if entry, ok := pm.pageTable[item.id]; ok {
			if entry.Priority < 0 || entry.Priority > 2 {
				t.Errorf("Item %d has invalid priority %d", item.id, entry.Priority)
			}
		} else {
			t.Errorf("Item %d not found in pageTable", item.id)
		}
	}
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
