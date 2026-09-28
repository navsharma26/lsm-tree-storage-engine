package lsm

import (
	"bytes"
	"container/heap"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// DefaultCompactionThreshold is the number of SSTables in a tier required to trigger compaction.
const DefaultCompactionThreshold = 4

// mergeItem represents a single entry in the K-Way Merge heap.
type mergeItem struct {
	key         []byte
	isTombstone bool
	valPtr      ValuePtr
	sstID       uint64
	iterIdx     int
}

// mergeHeap implements heap.Interface to select the smallest key across K iterators.
// If keys are identical, the entry with the higher sstID (newer) is prioritized.
type mergeHeap []*mergeItem

func (h mergeHeap) Len() int { return len(h) }

func (h mergeHeap) Less(i, j int) bool {
	cmp := bytes.Compare(h[i].key, h[j].key)
	if cmp != 0 {
		return cmp < 0 // ascending key order
	}
	// Newer SSTable takes precedence
	return h[i].sstID > h[j].sstID
}

func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *mergeHeap) Push(x any) {
	*h = append(*h, x.(*mergeItem))
}

func (h *mergeHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// MergeSSTables performs a K-Way Merge over the provided SSTable readers.
// - Deduplicates identical keys, retaining only the newest version (highest sstID).
// - Purges tombstones completely if isOldestHistorical is true; otherwise preserves them.
// - Outputs the merged data into an SSTable at outPath.
func MergeSSTables(tables []*SSTableReader, outPath string, sampleRate int, isOldestHistorical bool) (*SSTableReader, error) {
	if len(tables) == 0 {
		return nil, fmt.Errorf("lsm: no SSTables provided for compaction")
	}

	// 1. Create iterators for all K tables
	iters := make([]*SSTableIterator, len(tables))
	for i, tbl := range tables {
		tbl.Retain()
		defer tbl.Release()

		it, err := tbl.NewIterator()
		if err != nil {
			return nil, fmt.Errorf("lsm: failed opening iterator for SSTable %s: %w", tbl.Path(), err)
		}
		iters[i] = it
	}

	// 2. Initialize the Min-Heap
	h := &mergeHeap{}
	heap.Init(h)

	for i, it := range iters {
		if it.Next() {
			heap.Push(h, &mergeItem{
				key:         bytes.Clone(it.Key()),
				isTombstone: it.IsTombstone(),
				valPtr:      it.Value(),
				sstID:       tables[i].ID(),
				iterIdx:     i,
			})
		}
	}

	// 3. Prepare output SSTable writer
	writer, err := NewSSTableWriter(outPath, sampleRate)
	if err != nil {
		return nil, fmt.Errorf("lsm: failed creating output SSTable writer: %w", err)
	}

	var lastKey []byte

	// 4. K-Way Merge with deduplication and tombstone pruning
	for h.Len() > 0 {
		top := heap.Pop(h).(*mergeItem)

		// Replenish heap from the iterator that provided `top`
		if iters[top.iterIdx].Next() {
			heap.Push(h, &mergeItem{
				key:         bytes.Clone(iters[top.iterIdx].Key()),
				isTombstone: iters[top.iterIdx].IsTombstone(),
				valPtr:      iters[top.iterIdx].Value(),
				sstID:       tables[top.iterIdx].ID(),
				iterIdx:     top.iterIdx,
			})
		}

		// If this key was already emitted from a newer SSTable, discard older version
		if lastKey != nil && bytes.Equal(lastKey, top.key) {
			continue
		}

		// Drain all other duplicate entries of this key currently in the heap
		for h.Len() > 0 && bytes.Equal((*h)[0].key, top.key) {
			dup := heap.Pop(h).(*mergeItem)
			if iters[dup.iterIdx].Next() {
				heap.Push(h, &mergeItem{
					key:         bytes.Clone(iters[dup.iterIdx].Key()),
					isTombstone: iters[dup.iterIdx].IsTombstone(),
					valPtr:      iters[dup.iterIdx].Value(),
					sstID:       tables[dup.iterIdx].ID(),
					iterIdx:     dup.iterIdx,
				})
			}
		}

		lastKey = bytes.Clone(top.key)

		// Tombstone Handling:
		// Drop deleted keys completely if merging includes the oldest historical SSTables;
		// otherwise retain tombstone marker to mask older files.
		if top.isTombstone {
			if isOldestHistorical {
				// Purged completely from disk
				continue
			}
			if err := writer.Append(top.key, true, top.valPtr); err != nil {
				_ = writer.file.Close()
				return nil, err
			}
		} else {
			if err := writer.Append(top.key, false, top.valPtr); err != nil {
				_ = writer.file.Close()
				return nil, err
			}
		}
	}

	if err := writer.Finish(); err != nil {
		return nil, fmt.Errorf("lsm: failed finishing merged SSTable: %w", err)
	}

	return OpenSSTable(outPath)
}

// Compactor manages background size-tiered compaction across SSTables.
type Compactor struct {
	mu           sync.Mutex
	dir          string
	manifest     *Manifest
	sampleRate   int
	threshold    int
	triggerCh    chan struct{}
	stopCh       chan struct{}
	doneCh       chan struct{}
	wg           sync.WaitGroup
	onCompaction func(added *SSTableReader, removedIDs []uint64)
	getTableList func() []*SSTableReader
}

// NewCompactor constructs a Compactor and starts its background worker goroutine.
func NewCompactor(
	dir string,
	manifest *Manifest,
	sampleRate int,
	threshold int,
	getTableList func() []*SSTableReader,
	onCompaction func(added *SSTableReader, removedIDs []uint64),
) *Compactor {
	if threshold <= 1 {
		threshold = DefaultCompactionThreshold
	}
	if sampleRate <= 0 {
		sampleRate = DefaultSparseIndexSampleRate
	}

	c := &Compactor{
		dir:          dir,
		manifest:     manifest,
		sampleRate:   sampleRate,
		threshold:    threshold,
		triggerCh:    make(chan struct{}, 1),
		stopCh:       make(chan struct{}),
		doneCh:       make(chan struct{}),
		getTableList: getTableList,
		onCompaction: onCompaction,
	}

	c.wg.Add(1)
	go c.runLoop()

	return c
}

// Trigger notifies the background compactor to inspect the SSTable tier.
func (c *Compactor) Trigger() {
	select {
	case c.triggerCh <- struct{}{}:
	default:
	}
}

// CompactNow synchronously executes a compaction cycle if threshold is met.
// Returns the newly consolidated SSTableReader, or nil if no compaction occurred.
func (c *Compactor) CompactNow() (*SSTableReader, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	tables := c.getTableList()
	if len(tables) < c.threshold {
		return nil, nil
	}

	// Sort tables by ID ascending (oldest to newest)
	sortedTables := make([]*SSTableReader, len(tables))
	copy(sortedTables, tables)
	sort.Slice(sortedTables, func(i, j int) bool {
		return sortedTables[i].ID() < sortedTables[j].ID()
	})

	// Allocate new consolidated SSTable ID
	var newID uint64
	if c.manifest != nil {
		newID = c.manifest.AllocateSSTID()
	} else {
		newID = sortedTables[len(sortedTables)-1].ID() + 1
	}

	outPath := filepath.Join(c.dir, "sstable", fmt.Sprintf("%06d.sst", newID))

	// In size-tiered compaction of all current SSTables, this includes the oldest historical files
	isOldestHistorical := true

	newReader, err := MergeSSTables(sortedTables, outPath, c.sampleRate, isOldestHistorical)
	if err != nil {
		return nil, err
	}
	newReader.SetID(newID)

	removedIDs := make([]uint64, len(sortedTables))
	for i, t := range sortedTables {
		removedIDs[i] = t.ID()
	}

	// Update Manifest atomically
	if c.manifest != nil {
		stat, _ := os.Stat(outPath)
		var size int64
		if stat != nil {
			size = stat.Size()
		}
		newMeta := &SSTableMeta{
			ID:    newID,
			Level: 1,
			Path:  outPath,
			Size:  size,
		}
		if err := c.manifest.ApplyCompaction([]*SSTableMeta{newMeta}, removedIDs); err != nil {
			_ = newReader.Close()
			return nil, fmt.Errorf("lsm: failed applying compaction to manifest: %w", err)
		}
	}

	// Notify engine to swap active SSTable references and mark obsolete tables deleted
	if c.onCompaction != nil {
		c.onCompaction(newReader, removedIDs)
	}

	return newReader, nil
}

func (c *Compactor) runLoop() {
	defer c.wg.Done()

	for {
		select {
		case <-c.stopCh:
			return
		case <-c.triggerCh:
			_, _ = c.CompactNow()
		}
	}
}

// Close gracefully stops the background compaction worker.
func (c *Compactor) Close() error {
	select {
	case <-c.stopCh:
		return nil
	default:
		close(c.stopCh)
	}
	c.wg.Wait()
	return nil
}
