package lsm

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// FlushMemTableToSSTable takes an immutable frozen MemTable and writes its sorted
// entries into an SSTable file at sstPath with sparse indexing.
func FlushMemTableToSSTable(mem *MemTable, sstPath string, sampleRate int) (*SSTableReader, error) {
	writer, err := NewSSTableWriterWithExpectedKeys(sstPath, sampleRate, mem.Len())
	if err != nil {
		return nil, fmt.Errorf("lsm: failed to create SSTable writer for flush: %w", err)
	}

	it := mem.Iterator()
	it.SeekToFirst()
	for it.Valid() {
		k, vp, isDel, _ := it.Entry()
		if err := writer.Append(k, isDel, vp); err != nil {
			_ = writer.file.Close()
			return nil, fmt.Errorf("lsm: failed to append entry to SSTable during flush: %w", err)
		}
		it.Next()
	}

	if err := writer.Finish(); err != nil {
		return nil, fmt.Errorf("lsm: failed to finish SSTable during flush: %w", err)
	}

	return OpenSSTable(sstPath)
}

// FlushManager orchestrates active MemTable freezing, write redirection to a new
// MemTable with a fresh WAL, and background asynchronous flushing of frozen MemTables
// into immutable SSTables on disk.
type FlushManager struct {
	mu         sync.RWMutex
	dir        string
	vlog       *VLog
	activeMem  *MemTable
	immMem     *MemTable
	sstables   []*SSTableReader
	manifest   *Manifest
	compactor  *Compactor
	sampleRate int
	walSeq     uint64
	sstSeq     uint64
	flushMu    sync.Mutex
	wg         sync.WaitGroup
	closed     bool
}

// NewFlushManager initializes a FlushManager rooted at dir.
func NewFlushManager(dir string, vlog *VLog, sampleRate int) (*FlushManager, error) {
	if sampleRate <= 0 {
		sampleRate = DefaultSparseIndexSampleRate
	}

	walDir := filepath.Join(dir, "wal")
	sstDir := filepath.Join(dir, "sstable")
	if err := os.MkdirAll(walDir, 0755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(sstDir, 0755); err != nil {
		return nil, err
	}

	walPath := filepath.Join(walDir, "wal_000001.wal")
	wal, err := OpenWAL(walPath)
	if err != nil {
		return nil, err
	}

	manifest, err := OpenManifest(dir)
	if err != nil {
		_ = wal.Close()
		return nil, err
	}

	return &FlushManager{
		dir:        dir,
		vlog:       vlog,
		activeMem:  NewMemTable(wal),
		manifest:   manifest,
		sampleRate: sampleRate,
		walSeq:     1,
		sstSeq:     0,
	}, nil
}

// Put writes a key-value pair following the WiscKey decoupling architecture:
// 1. Appends value to VLog using buffered I/O, obtaining ValuePtr.
// 2. Writes key + ValuePtr to active MemTable (and its WAL).
func (fm *FlushManager) Put(key, val []byte) error {
	fm.mu.RLock()
	closed := fm.closed
	fm.mu.RUnlock()
	if closed {
		return os.ErrClosed
	}

	var vp ValuePtr
	var err error
	if fm.vlog != nil {
		vp, err = fm.vlog.Write(key, val)
		if err != nil {
			return fmt.Errorf("lsm: failed writing value to VLog: %w", err)
		}
	}

	fm.mu.RLock()
	mem := fm.activeMem
	fm.mu.RUnlock()

	return mem.Put(key, vp)
}

// Delete appends a tombstone for key into active MemTable (and WAL).
func (fm *FlushManager) Delete(key []byte) error {
	fm.mu.RLock()
	closed := fm.closed
	mem := fm.activeMem
	fm.mu.RUnlock()
	if closed {
		return os.ErrClosed
	}

	return mem.Delete(key)
}

// Get looks up a key across the LSM hierarchy:
// 1. Active MemTable
// 2. Frozen Immutable MemTable (if flush in progress)
// 3. SSTables on disk (newest to oldest)
// Once ValuePtr is found, retrieves the raw value from VLog via io.ReadAt.
func (fm *FlushManager) Get(key []byte) ([]byte, error) {
	fm.mu.RLock()
	defer fm.mu.RUnlock()

	if fm.closed {
		return nil, os.ErrClosed
	}

	// 1. Check Active MemTable
	vp, isDel, found := fm.activeMem.GetRecord(key)
	if found {
		if isDel {
			return nil, ErrKeyNotFound
		}
		if fm.vlog != nil {
			return fm.vlog.Read(vp)
		}
		return nil, nil
	}

	// 2. Check Immutable MemTable
	if fm.immMem != nil {
		vp, isDel, found = fm.immMem.GetRecord(key)
		if found {
			if isDel {
				return nil, ErrKeyNotFound
			}
			if fm.vlog != nil {
				return fm.vlog.Read(vp)
			}
			return nil, nil
		}
	}

	// 3. Check SSTables from newest to oldest
	for i := len(fm.sstables) - 1; i >= 0; i-- {
		sst := fm.sstables[i]
		sst.Retain()
		vp, isTombstone, err := sst.Get(key)
		_ = sst.Release()
		if err == nil {
			if isTombstone {
				return nil, ErrKeyNotFound
			}
			if fm.vlog != nil {
				return fm.vlog.Read(vp)
			}
			return nil, nil
		}
	}

	return nil, ErrKeyNotFound
}

// RotateMemTable freezes active MemTable, establishes a new active MemTable with a fresh WAL,
// and returns the frozen MemTable ready for flushing.
func (fm *FlushManager) RotateMemTable() (*MemTable, error) {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.closed {
		return nil, os.ErrClosed
	}
	if fm.immMem != nil {
		return nil, fmt.Errorf("lsm: flush already in progress")
	}

	// Create new WAL for subsequent writes
	newWALID := atomic.AddUint64(&fm.walSeq, 1)
	walPath := filepath.Join(fm.dir, "wal", fmt.Sprintf("wal_%06d.wal", newWALID))
	newWAL, err := OpenWAL(walPath)
	if err != nil {
		return nil, fmt.Errorf("lsm: failed creating new WAL: %w", err)
	}

	frozenMem := fm.activeMem
	fm.immMem = frozenMem
	fm.activeMem = NewMemTable(newWAL)

	return frozenMem, nil
}

// RotateAndFlushAsync freezes active MemTable, redirects new writes to a fresh MemTable+WAL,
// and flushes frozen MemTable to disk asynchronously in a background goroutine.
// Returns a completion error channel.
func (fm *FlushManager) RotateAndFlushAsync() (<-chan error, error) {
	frozenMem, err := fm.RotateMemTable()
	if err != nil {
		return nil, err
	}

	done := make(chan error, 1)
	fm.wg.Add(1)

	go func() {
		defer fm.wg.Done()
		err := fm.flushFrozen(frozenMem)
		done <- err
		close(done)
	}()

	return done, nil
}

// RotateAndFlushSync synchronously freezes and flushes the active MemTable.
func (fm *FlushManager) RotateAndFlushSync() (*SSTableReader, error) {
	frozenMem, err := fm.RotateMemTable()
	if err != nil {
		return nil, err
	}

	if err := fm.flushFrozen(frozenMem); err != nil {
		return nil, err
	}

	fm.mu.RLock()
	defer fm.mu.RUnlock()
	if len(fm.sstables) > 0 {
		return fm.sstables[len(fm.sstables)-1], nil
	}
	return nil, nil
}

func (fm *FlushManager) flushFrozen(frozenMem *MemTable) error {
	fm.flushMu.Lock()
	defer fm.flushMu.Unlock()

	var sstID uint64
	if fm.manifest != nil {
		sstID = fm.manifest.AllocateSSTID()
	} else {
		sstID = atomic.AddUint64(&fm.sstSeq, 1)
	}
	sstPath := filepath.Join(fm.dir, "sstable", fmt.Sprintf("%06d.sst", sstID))

	reader, err := FlushMemTableToSSTable(frozenMem, sstPath, fm.sampleRate)
	if err != nil {
		return err
	}
	reader.SetID(sstID)

	if fm.manifest != nil {
		stat, _ := os.Stat(sstPath)
		var size int64
		if stat != nil {
			size = stat.Size()
		}
		_ = fm.manifest.AddTable(&SSTableMeta{
			ID:    sstID,
			Level: 0,
			Path:  sstPath,
			Size:  size,
		})
	}

	// Safely install new SSTable and retire immMem
	fm.mu.Lock()
	fm.sstables = append(fm.sstables, reader)
	fm.immMem = nil
	fm.mu.Unlock()

	// Clean up old WAL of the flushed MemTable
	if oldWAL := frozenMem.WAL(); oldWAL != nil {
		walPath := oldWAL.Path()
		_ = oldWAL.Close()
		_ = os.Remove(walPath)
	}

	// Trigger compaction check if enabled
	if fm.compactor != nil {
		fm.compactor.Trigger()
	}

	return nil
}

// EnableCompaction configures and launches a background Compactor.
func (fm *FlushManager) EnableCompaction(threshold int) *Compactor {
	fm.mu.Lock()
	defer fm.mu.Unlock()

	if fm.compactor != nil {
		return fm.compactor
	}

	getTableList := func() []*SSTableReader {
		fm.mu.RLock()
		defer fm.mu.RUnlock()
		tables := make([]*SSTableReader, len(fm.sstables))
		copy(tables, fm.sstables)
		return tables
	}

	onCompaction := func(added *SSTableReader, removedIDs []uint64) {
		fm.mu.Lock()
		defer fm.mu.Unlock()

		removedMap := make(map[uint64]bool, len(removedIDs))
		for _, id := range removedIDs {
			removedMap[id] = true
		}

		var remaining []*SSTableReader
		for _, sst := range fm.sstables {
			if removedMap[sst.ID()] {
				_ = sst.MarkDeleted()
			} else {
				remaining = append(remaining, sst)
			}
		}

		remaining = append(remaining, added)
		fm.sstables = remaining
	}

	fm.compactor = NewCompactor(fm.dir, fm.manifest, fm.sampleRate, threshold, getTableList, onCompaction)
	return fm.compactor
}

// Compactor returns the active compactor instance, if enabled.
func (fm *FlushManager) Compactor() *Compactor {
	fm.mu.RLock()
	defer fm.mu.RUnlock()
	return fm.compactor
}

// Manifest returns the active manifest instance.
func (fm *FlushManager) Manifest() *Manifest {
	return fm.manifest
}

// ActiveMemTable returns current active MemTable.
func (fm *FlushManager) ActiveMemTable() *MemTable {
	fm.mu.RLock()
	defer fm.mu.RUnlock()
	return fm.activeMem
}

// ImmutableMemTable returns frozen MemTable currently being flushed, if any.
func (fm *FlushManager) ImmutableMemTable() *MemTable {
	fm.mu.RLock()
	defer fm.mu.RUnlock()
	return fm.immMem
}

// SSTables returns readers for all flushed SSTables.
func (fm *FlushManager) SSTables() []*SSTableReader {
	fm.mu.RLock()
	defer fm.mu.RUnlock()
	ssts := make([]*SSTableReader, len(fm.sstables))
	copy(ssts, fm.sstables)
	return ssts
}

// Close gracefully waits for ongoing flushes, syncs VLog, and closes all open files.
func (fm *FlushManager) Close() error {
	fm.mu.Lock()
	if fm.closed {
		fm.mu.Unlock()
		return nil
	}
	fm.closed = true
	fm.mu.Unlock()

	// Wait for background flushes to finish
	fm.wg.Wait()

	// Stop compactor if running
	if fm.compactor != nil {
		_ = fm.compactor.Close()
	}

	fm.mu.Lock()
	defer fm.mu.Unlock()

	var firstErr error
	if fm.activeMem != nil {
		if err := fm.activeMem.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	for _, sst := range fm.sstables {
		if err := sst.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if fm.vlog != nil {
		if err := fm.vlog.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}
