package lsm

import (
	"sync"
	"sync/atomic"
)

const (
	// DefaultFlushThreshold is the standard 4MB threshold to trigger MemTable flush to L0 SSTable.
	DefaultFlushThreshold int64 = 4 * 1024 * 1024

	// MaxFlushThreshold is the maximum 8MB threshold before write stalls occur.
	MaxFlushThreshold int64 = 8 * 1024 * 1024
)

// MemTable is an in-memory sorted write buffer wrapping a SkipList and WAL.
// It tracks its current memory usage in bytes (`sizeBytes`) to detect when
// flush thresholds (e.g. 4MB/8MB) are reached.
type MemTable struct {
	mu        sync.RWMutex
	sl        *SkipList
	wal       *WAL
	sizeBytes int64
}

// NewMemTable constructs a new empty MemTable optionally associated with a WAL.
func NewMemTable(wal *WAL) *MemTable {
	return &MemTable{
		sl:  NewSkipList(),
		wal: wal,
	}
}

// NewMemTableFromWAL restores a MemTable from an existing WAL by replaying valid log entries.
func NewMemTableFromWAL(wal *WAL) (*MemTable, error) {
	mem := NewMemTable(wal)
	if wal == nil {
		return mem, nil
	}

	entries, err := wal.Recover()
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		switch entry.Type {
		case RecordPut:
			isUpdate := mem.sl.Insert(entry.Key, entry.ValuePtr, false)
			if !isUpdate {
				mem.addSize(int64(len(entry.Key) + ValuePtrSize))
			}
		case RecordDelete:
			isUpdate := mem.sl.Insert(entry.Key, entry.ValuePtr, true)
			if !isUpdate {
				mem.addSize(int64(len(entry.Key) + ValuePtrSize))
			}
		}
	}

	return mem, nil
}

// Put writes a key and ValuePtr to the MemTable.
// If a WAL is attached, it is logged first for durability before updating memory.
func (m *MemTable) Put(key []byte, valuePtr ValuePtr) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(key) == 0 {
		return ErrKeyEmpty
	}

	if m.wal != nil {
		if err := m.wal.Append(LogEntry{
			Type:     RecordPut,
			Key:      key,
			ValuePtr: valuePtr,
		}); err != nil {
			return err
		}
	}

	isUpdate := m.sl.Insert(key, valuePtr, false)
	if !isUpdate {
		m.addSize(int64(len(key) + ValuePtrSize))
	}

	return nil
}

// Delete marks a key as deleted (tombstone) in the MemTable.
// If a WAL is attached, a tombstone record is appended to the log first.
func (m *MemTable) Delete(key []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(key) == 0 {
		return ErrKeyEmpty
	}

	if m.wal != nil {
		if err := m.wal.Append(LogEntry{
			Type:     RecordDelete,
			Key:      key,
			ValuePtr: ValuePtr{},
		}); err != nil {
			return err
		}
	}

	isUpdate := m.sl.Insert(key, ValuePtr{}, true)
	if !isUpdate {
		m.addSize(int64(len(key) + ValuePtrSize))
	}

	return nil
}

// Get retrieves the ValuePtr for a key from the MemTable.
// Returns ErrKeyNotFound if the key does not exist or has been deleted.
func (m *MemTable) Get(key []byte) (ValuePtr, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(key) == 0 {
		return ValuePtr{}, ErrKeyEmpty
	}

	valPtr, isDeleted, found := m.sl.GetRecord(key)
	if !found || isDeleted {
		return ValuePtr{}, ErrKeyNotFound
	}

	return valPtr, nil
}

// GetRecord searches for key and returns (valuePtr, isDeleted, found).
// This allows caller to distinguish between a key that doesn't exist vs tombstone.
func (m *MemTable) GetRecord(key []byte) (ValuePtr, bool, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.sl.GetRecord(key)
}

// SizeBytes returns the approximate size in bytes of data stored in this MemTable.
func (m *MemTable) SizeBytes() int64 {
	return atomic.LoadInt64(&m.sizeBytes)
}

// IsThresholdReached reports whether the MemTable size has met or exceeded the given threshold.
func (m *MemTable) IsThresholdReached(threshold int64) bool {
	return m.SizeBytes() >= threshold
}

// Iterator returns an in-order sorted iterator over the MemTable entries.
func (m *MemTable) Iterator() *Iterator {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.sl.Iterator()
}

// Len returns the number of unique entries in the MemTable.
func (m *MemTable) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.sl.Len()
}

// WAL returns the underlying WAL instance, or nil if none was provided.
func (m *MemTable) WAL() *WAL {
	return m.wal
}

// Sync flushes the underlying WAL to disk if present.
func (m *MemTable) Sync() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.wal != nil {
		return m.wal.Sync()
	}
	return nil
}

// Close closes the attached WAL file.
func (m *MemTable) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.wal != nil {
		return m.wal.Close()
	}
	return nil
}

func (m *MemTable) addSize(delta int64) {
	atomic.AddInt64(&m.sizeBytes, delta)
}
