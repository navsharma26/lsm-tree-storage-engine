package lsm

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// ReadStats captures operational diagnostics during point lookups
// to verify zero-disk-I/O Bloom filter skips and optimal routing.
type ReadStats struct {
	ActiveMemHit     bool
	ImmMemHit        bool
	SSTablesExamined int
	BloomFilterSkips int
	BloomFilterHits  int
	DiskReads        int
}

// DB represents the unified LSM-tree storage engine with WiscKey key-value separation,
// embedded Bloom filters, and size-tiered background compaction.
type DB struct {
	mu            sync.RWMutex
	dir           string
	vlog          *VLog
	activeMem     *MemTable
	immMem        *MemTable
	sstables      []*SSTableReader
	manifest      *Manifest
	compactor     *Compactor
	sampleRate    int
	bloomDisabled bool
	closed        bool
}

// Options configure the DB engine parameters.
type Options struct {
	SparseIndexSampleRate int
	CompactionThreshold   int
	EnableCompaction      bool
	VLogFileID            uint32
	DisableBloomFilter    bool
}

// DefaultOptions returns standard production configuration.
func DefaultOptions() Options {
	return Options{
		SparseIndexSampleRate: DefaultSparseIndexSampleRate,
		CompactionThreshold:   DefaultCompactionThreshold,
		EnableCompaction:      true,
		VLogFileID:            1,
	}
}

// OpenDB opens or initializes a storage engine instance in dir.
func OpenDB(dir string, opt Options) (*DB, error) {
	if opt.SparseIndexSampleRate <= 0 {
		opt.SparseIndexSampleRate = DefaultSparseIndexSampleRate
	}
	if opt.CompactionThreshold <= 1 {
		opt.CompactionThreshold = DefaultCompactionThreshold
	}
	if opt.VLogFileID == 0 {
		opt.VLogFileID = 1
	}

	walDir := filepath.Join(dir, "wal")
	sstDir := filepath.Join(dir, "sstable")
	vlogDir := filepath.Join(dir, "vlog")

	for _, d := range []string{walDir, sstDir, vlogDir} {
		if err := os.MkdirAll(d, 0755); err != nil {
			return nil, fmt.Errorf("lsm: failed creating directory %s: %w", d, err)
		}
	}

	// 1. Open VLog
	vlogPath := filepath.Join(vlogDir, fmt.Sprintf("%06d.vlog", opt.VLogFileID))
	vlog, err := OpenVLog(vlogPath, opt.VLogFileID)
	if err != nil {
		return nil, fmt.Errorf("lsm: failed opening VLog: %w", err)
	}

	// 2. Open Manifest
	manifest, err := OpenManifest(dir)
	if err != nil {
		_ = vlog.Close()
		return nil, fmt.Errorf("lsm: failed opening Manifest: %w", err)
	}

	// 3. Scan and recover existing WAL files into Active MemTable
	walFiles, err := filepath.Glob(filepath.Join(walDir, "wal_*.wal"))
	if err != nil {
		_ = vlog.Close()
		return nil, fmt.Errorf("lsm: failed searching WAL directory: %w", err)
	}
	sort.Strings(walFiles)

	var activeWAL *WAL
	var recoveredEntries []LogEntry

	if len(walFiles) > 0 {
		for i, wf := range walFiles {
			var wID uint64
			if _, err := fmt.Sscanf(filepath.Base(wf), "wal_%d.wal", &wID); err == nil {
				manifest.EnsureNextSSTID(wID)
			}
			w, err := OpenWAL(wf)
			if err != nil {
				_ = vlog.Close()
				return nil, fmt.Errorf("lsm: failed opening existing WAL %s: %w", wf, err)
			}
			entries, err := w.Recover()
			if err != nil {
				_ = w.Close()
				_ = vlog.Close()
				return nil, fmt.Errorf("lsm: failed recovering WAL %s: %w", wf, err)
			}
			recoveredEntries = append(recoveredEntries, entries...)

			if i == len(walFiles)-1 {
				activeWAL = w
			} else {
				_ = w.Close()
			}
		}
	} else {
		walID := manifest.AllocateSSTID()
		walPath := filepath.Join(walDir, fmt.Sprintf("wal_%06d.wal", walID))
		w, err := OpenWAL(walPath)
		if err != nil {
			_ = vlog.Close()
			return nil, fmt.Errorf("lsm: failed opening WAL: %w", err)
		}
		activeWAL = w
	}

	activeMem := NewMemTable(activeWAL)
	for _, entry := range recoveredEntries {
		switch entry.Type {
		case RecordPut:
			activeMem.sl.Insert(entry.Key, entry.ValuePtr, false)
			activeMem.addSize(int64(len(entry.Key) + ValuePtrSize))
		case RecordDelete:
			activeMem.sl.Insert(entry.Key, entry.ValuePtr, true)
			activeMem.addSize(int64(len(entry.Key) + ValuePtrSize))
		}
	}

	// 4. Load existing SSTables tracked in Manifest
	var sstables []*SSTableReader
	for _, meta := range manifest.ActiveTables() {
		if _, err := os.Stat(meta.Path); err == nil {
			reader, err := OpenSSTable(meta.Path)
			if err == nil {
				reader.SetID(meta.ID)
				sstables = append(sstables, reader)
			}
		}
	}

	db := &DB{
		dir:           dir,
		vlog:          vlog,
		activeMem:     activeMem,
		sstables:      sstables,
		manifest:      manifest,
		sampleRate:    opt.SparseIndexSampleRate,
		bloomDisabled: opt.DisableBloomFilter,
	}

	if opt.DisableBloomFilter {
		for _, sst := range sstables {
			sst.SetBloomFilterEnabled(false)
		}
	}

	// 5. Optionally enable background Compactor
	if opt.EnableCompaction {
		db.enableCompactorLocked(opt.CompactionThreshold)
	}

	return db, nil
}

// Put writes key-value data decoupled into VLog and active MemTable.
func (db *DB) Put(key, val []byte) error {
	db.mu.RLock()
	if db.closed {
		db.mu.RUnlock()
		return os.ErrClosed
	}
	vlog := db.vlog
	mem := db.activeMem
	db.mu.RUnlock()

	vp, err := vlog.Write(key, val)
	if err != nil {
		return fmt.Errorf("lsm: failed writing value to VLog: %w", err)
	}

	return mem.Put(key, vp)
}

// Delete appends a tombstone for key into active MemTable (and WAL).
func (db *DB) Delete(key []byte) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return os.ErrClosed
	}
	return db.activeMem.Delete(key)
}

// SetBloomFilterEnabled toggles Bloom filter checks on or off across all SSTables.
// Useful for benchmarking read latency with vs without Bloom filtering.
func (db *DB) SetBloomFilterEnabled(enabled bool) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.bloomDisabled = !enabled
	for _, sst := range db.sstables {
		sst.SetBloomFilterEnabled(enabled)
	}
}

// IsBloomFilterEnabled reports whether Bloom filter checks are enabled.
func (db *DB) IsBloomFilterEnabled() bool {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return !db.bloomDisabled
}

// GetWithoutBloom executes the read path bypassing Bloom filter checks,
// forcing sparse index binary search and disk reads on SSTables.
func (db *DB) GetWithoutBloom(key []byte) ([]byte, error) {
	val, _, err := db.getWithStatsInternal(key, false)
	return val, err
}

// Get executes the unified read path:
// 1. Search active MemTable.
// 2. Search immutable MemTable (if flushing).
// 3. Search SSTables newest to oldest: check in-memory Bloom filter first.
//    If false, skip the file immediately with ZERO disk I/O.
// 4. If Bloom filter hits, binary search Sparse Index -> seek SSTable -> get ValuePtr -> fetch from VLog.
func (db *DB) Get(key []byte) ([]byte, error) {
	val, _, err := db.GetWithStats(key)
	return val, err
}

// GetWithStats performs Get(key) while recording routing statistics and disk reads.
func (db *DB) GetWithStats(key []byte) ([]byte, ReadStats, error) {
	db.mu.RLock()
	useBloom := !db.bloomDisabled
	db.mu.RUnlock()
	return db.getWithStatsInternal(key, useBloom)
}

func (db *DB) getWithStatsInternal(key []byte, useBloom bool) ([]byte, ReadStats, error) {
	var stats ReadStats
	if len(key) == 0 {
		return nil, stats, ErrKeyEmpty
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, stats, os.ErrClosed
	}

	// Step 1: Search Active MemTable
	vp, isDel, found := db.activeMem.GetRecord(key)
	if found {
		stats.ActiveMemHit = true
		if isDel {
			return nil, stats, ErrKeyNotFound
		}
		val, err := db.vlog.Read(vp)
		return val, stats, err
	}

	// Step 2: Search Immutable MemTable (if flushing)
	if db.immMem != nil {
		vp, isDel, found = db.immMem.GetRecord(key)
		if found {
			stats.ImmMemHit = true
			if isDel {
				return nil, stats, ErrKeyNotFound
			}
			val, err := db.vlog.Read(vp)
			return val, stats, err
		}
	}

	// Step 3: Search SSTables from newest to oldest
	for i := len(db.sstables) - 1; i >= 0; i-- {
		sst := db.sstables[i]
		stats.SSTablesExamined++

		// Check in-memory Bloom filter first!
		// If false: SKIP THE FILE IMMEDIATELY WITH ZERO DISK I/O!
		if useBloom && !sst.MayContain(key) {
			stats.BloomFilterSkips++
			continue
		}

		stats.BloomFilterHits++

		// Step 4: Bloom filter hits -> binary search Sparse Index -> seek SSTable -> get ValuePtr
		sst.Retain()
		var vp ValuePtr
		var isTombstone bool
		var bytesRead int
		var err error
		if useBloom {
			vp, isTombstone, bytesRead, err = sst.GetWithStats(key)
		} else {
			vp, isTombstone, bytesRead, err = sst.GetWithoutBloom(key)
		}
		_ = sst.Release()

		if bytesRead > 0 {
			stats.DiskReads++
		}

		if err == nil {
			if isTombstone {
				return nil, stats, ErrKeyNotFound
			}
			// Fetch actual value payload from VLog
			val, err := db.vlog.Read(vp)
			return val, stats, err
		}
	}

	return nil, stats, ErrKeyNotFound
}

// Flush synchronously freezes the active MemTable and writes it to an immutable SSTable.
func (db *DB) Flush() (*SSTableReader, error) {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return nil, os.ErrClosed
	}
	if db.immMem != nil {
		db.mu.Unlock()
		return nil, fmt.Errorf("lsm: flush already in progress")
	}

	// Allocate new WAL for active writes
	newWALID := db.manifest.AllocateSSTID()
	walPath := filepath.Join(db.dir, "wal", fmt.Sprintf("wal_%06d.wal", newWALID))
	newWAL, err := OpenWAL(walPath)
	if err != nil {
		db.mu.Unlock()
		return nil, err
	}

	// Ensure VLog is committed before freezing MemTable into SSTable
	if db.vlog != nil {
		if err := db.vlog.Flush(); err != nil {
			db.mu.Unlock()
			return nil, fmt.Errorf("lsm: failed flushing VLog during MemTable flush: %w", err)
		}
	}

	frozenMem := db.activeMem
	db.immMem = frozenMem
	db.activeMem = NewMemTable(newWAL)
	db.mu.Unlock()

	// Write SSTable to disk
	sstID := db.manifest.AllocateSSTID()
	sstPath := filepath.Join(db.dir, "sstable", fmt.Sprintf("%06d.sst", sstID))

	reader, err := FlushMemTableToSSTable(frozenMem, sstPath, db.sampleRate)
	if err != nil {
		return nil, err
	}
	reader.SetID(sstID)

	stat, _ := os.Stat(sstPath)
	var size int64
	if stat != nil {
		size = stat.Size()
	}
	_ = db.manifest.AddTable(&SSTableMeta{
		ID:    sstID,
		Level: 0,
		Path:  sstPath,
		Size:  size,
	})

	db.mu.Lock()
	if db.bloomDisabled {
		reader.SetBloomFilterEnabled(false)
	}
	db.sstables = append(db.sstables, reader)
	db.immMem = nil
	db.mu.Unlock()

	// Remove old WAL
	if oldWAL := frozenMem.WAL(); oldWAL != nil {
		p := oldWAL.Path()
		_ = oldWAL.Close()
		_ = os.Remove(p)
	}

	// Trigger compaction if enabled
	if db.compactor != nil {
		db.compactor.Trigger()
	}

	return reader, nil
}

// Compact triggers a synchronous compaction cycle.
func (db *DB) Compact() (*SSTableReader, error) {
	db.mu.RLock()
	compactor := db.compactor
	db.mu.RUnlock()

	if compactor == nil {
		return nil, fmt.Errorf("lsm: compaction not enabled")
	}
	return compactor.CompactNow()
}

// SSTables returns readers for all active SSTables.
func (db *DB) SSTables() []*SSTableReader {
	db.mu.RLock()
	defer db.mu.RUnlock()
	res := make([]*SSTableReader, len(db.sstables))
	copy(res, db.sstables)
	return res
}

// VLog returns the underlying Value Log.
func (db *DB) VLog() *VLog {
	return db.vlog
}

// Dir returns the root storage engine directory.
func (db *DB) Dir() string {
	return db.dir
}

// ActiveMemTable returns the current active MemTable.
func (db *DB) ActiveMemTable() *MemTable {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.activeMem
}

// Manifest returns the active Manifest tracker.
func (db *DB) Manifest() *Manifest {
	return db.manifest
}

// Sync commits both VLog and active MemTable's WAL to disk (fsync).
func (db *DB) Sync() error {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed {
		return os.ErrClosed
	}
	if db.vlog != nil {
		if err := db.vlog.Sync(); err != nil {
			return err
		}
	}
	if db.activeMem != nil {
		if err := db.activeMem.Sync(); err != nil {
			return err
		}
	}
	return nil
}

// Close gracefully stops workers, flushes VLog, and closes all open file descriptors.
func (db *DB) Close() error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return nil
	}
	db.closed = true
	db.mu.Unlock()

	if db.compactor != nil {
		_ = db.compactor.Close()
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	var firstErr error
	if db.activeMem != nil {
		if err := db.activeMem.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	for _, sst := range db.sstables {
		if err := sst.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if db.vlog != nil {
		if err := db.vlog.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

func (db *DB) enableCompactorLocked(threshold int) {
	getTableList := func() []*SSTableReader {
		db.mu.RLock()
		defer db.mu.RUnlock()
		tables := make([]*SSTableReader, len(db.sstables))
		copy(tables, db.sstables)
		return tables
	}

	onCompaction := func(added *SSTableReader, removedIDs []uint64) {
		db.mu.Lock()
		defer db.mu.Unlock()

		removedMap := make(map[uint64]bool, len(removedIDs))
		for _, id := range removedIDs {
			removedMap[id] = true
		}

		var remaining []*SSTableReader
		for _, sst := range db.sstables {
			if removedMap[sst.ID()] {
				_ = sst.MarkDeleted()
			} else {
				remaining = append(remaining, sst)
			}
		}

		if db.bloomDisabled {
			added.SetBloomFilterEnabled(false)
		}
		remaining = append(remaining, added)
		db.sstables = remaining
	}

	db.compactor = NewCompactor(db.dir, db.manifest, db.sampleRate, threshold, getTableList, onCompaction)
}
