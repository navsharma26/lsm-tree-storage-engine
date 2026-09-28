package test

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"lsmtree/pkg/lsm"
)

// TestValuePtrEncodingDecoding verifies binary serialization of ValuePtr.
func TestValuePtrEncodingDecoding(t *testing.T) {
	testCases := []lsm.ValuePtr{
		{FileID: 0, Offset: 0, Len: 0},
		{FileID: 1, Offset: 1024, Len: 4096},
		{FileID: 42, Offset: 9876543210, Len: 65536},
		{FileID: ^uint32(0), Offset: ^uint64(0), Len: ^uint32(0)},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("Case_%d", i), func(t *testing.T) {
			encoded := tc.Encode()
			if len(encoded) != lsm.ValuePtrSize {
				t.Fatalf("expected encoded length %d, got %d", lsm.ValuePtrSize, len(encoded))
			}

			decoded, err := lsm.DecodeValuePtr(encoded)
			if err != nil {
				t.Fatalf("unexpected decode error: %v", err)
			}
			if decoded != tc {
				t.Fatalf("round-trip mismatch: expected %+v, got %+v", tc, decoded)
			}
		})
	}

	// Verify error handling for truncated input
	_, err := lsm.DecodeValuePtr([]byte{1, 2, 3})
	if err != lsm.ErrCorruptedRecord {
		t.Fatalf("expected ErrCorruptedRecord for short slice, got %v", err)
	}
}

// TestWALBasicOperations verifies normal append, sync, and clean recovery.
func TestWALBasicOperations(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "test.wal")

	wal, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	entries := []lsm.LogEntry{
		{Type: lsm.RecordPut, Key: []byte("user:1001"), ValuePtr: lsm.ValuePtr{FileID: 1, Offset: 0, Len: 128}},
		{Type: lsm.RecordPut, Key: []byte("user:1002"), ValuePtr: lsm.ValuePtr{FileID: 1, Offset: 128, Len: 256}},
		{Type: lsm.RecordDelete, Key: []byte("user:1001"), ValuePtr: lsm.ValuePtr{}},
		{Type: lsm.RecordPut, Key: []byte("user:1003"), ValuePtr: lsm.ValuePtr{FileID: 2, Offset: 0, Len: 512}},
	}

	for _, entry := range entries {
		if err := wal.Append(entry); err != nil {
			t.Fatalf("failed to append entry: %v", err)
		}
	}

	if err := wal.Sync(); err != nil {
		t.Fatalf("failed to sync WAL: %v", err)
	}

	if err := wal.Close(); err != nil {
		t.Fatalf("failed to close WAL: %v", err)
	}

	// Reopen and recover
	wal2, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to reopen WAL: %v", err)
	}
	defer wal2.Close()

	recovered, err := wal2.Recover()
	if err != nil {
		t.Fatalf("recovery failed: %v", err)
	}

	if len(recovered) != len(entries) {
		t.Fatalf("expected %d entries, got %d", len(entries), len(recovered))
	}

	for i, exp := range entries {
		got := recovered[i]
		if got.Type != exp.Type {
			t.Errorf("entry %d: expected type %v, got %v", i, exp.Type, got.Type)
		}
		if !bytes.Equal(got.Key, exp.Key) {
			t.Errorf("entry %d: expected key %s, got %s", i, exp.Key, got.Key)
		}
		if got.ValuePtr != exp.ValuePtr {
			t.Errorf("entry %d: expected valuePtr %+v, got %+v", i, exp.ValuePtr, got.ValuePtr)
		}
	}
}

// TestWALCrashRecoveryTruncatedPayload simulates a crash mid-write in the payload.
func TestWALCrashRecoveryTruncatedPayload(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "crash_payload.wal")

	wal, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("key_%05d", i)
		err := wal.Append(lsm.LogEntry{
			Type:     lsm.RecordPut,
			Key:      []byte(key),
			ValuePtr: lsm.ValuePtr{FileID: 1, Offset: uint64(i * 100), Len: 100},
		})
		if err != nil {
			t.Fatalf("failed to append: %v", err)
		}
	}
	wal.Close()

	// Simulate crash: truncate last 4 bytes of the last record's key payload
	stat, err := os.Stat(walPath)
	if err != nil {
		t.Fatalf("stat failed: %v", err)
	}
	truncatedSize := stat.Size() - 4
	if err := os.Truncate(walPath, truncatedSize); err != nil {
		t.Fatalf("truncate failed: %v", err)
	}

	// Reopen and recover: must halt cleanly and recover 9 complete records
	walRecovered, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}
	defer walRecovered.Close()

	entries, err := walRecovered.Recover()
	if err != nil {
		t.Fatalf("expected clean halt, got error: %v", err)
	}

	if len(entries) != 9 {
		t.Fatalf("expected 9 recovered entries, got %d", len(entries))
	}

	for i := 0; i < 9; i++ {
		expKey := fmt.Sprintf("key_%05d", i)
		if string(entries[i].Key) != expKey {
			t.Errorf("expected key %s, got %s", expKey, entries[i].Key)
		}
	}

	// Verify subsequent appends work cleanly from the valid offset
	err = walRecovered.Append(lsm.LogEntry{
		Type:     lsm.RecordPut,
		Key:      []byte("key_new_after_crash"),
		ValuePtr: lsm.ValuePtr{FileID: 2, Offset: 500, Len: 50},
	})
	if err != nil {
		t.Fatalf("failed to append after crash recovery: %v", err)
	}

	// Recover again to verify the new record is present
	entries2, err := walRecovered.Recover()
	if err != nil {
		t.Fatalf("second recovery failed: %v", err)
	}
	if len(entries2) != 10 {
		t.Fatalf("expected 10 entries after post-crash append, got %d", len(entries2))
	}
	if string(entries2[9].Key) != "key_new_after_crash" {
		t.Fatalf("expected last entry key_new_after_crash, got %s", entries2[9].Key)
	}
}

// TestWALCrashRecoveryTruncatedHeader simulates crash in the record header.
func TestWALCrashRecoveryTruncatedHeader(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "crash_header.wal")

	wal, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	for i := 0; i < 5; i++ {
		key := fmt.Sprintf("header_test_%03d", i)
		_ = wal.Append(lsm.LogEntry{
			Type:     lsm.RecordPut,
			Key:      []byte(key),
			ValuePtr: lsm.ValuePtr{FileID: 1, Offset: uint64(i * 10), Len: 10},
		})
	}
	wal.Close()

	// Truncate back into the header of the 5th record (leave only CRC + 5 bytes of header)
	stat, _ := os.Stat(walPath)
	recordSize := 4 + 19 + len("header_test_004")
	truncatedSize := stat.Size() - int64(recordSize) + 8 // 4B CRC + 4B partial header
	_ = os.Truncate(walPath, truncatedSize)

	walRec, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open: %v", err)
	}
	defer walRec.Close()

	entries, err := walRec.Recover()
	if err != nil {
		t.Fatalf("expected clean halt, got: %v", err)
	}
	if len(entries) != 4 {
		t.Fatalf("expected 4 entries recovered, got %d", len(entries))
	}
}

// TestWALCrashRecoveryCorruptedCRC simulates a bit flip / checksum mismatch.
func TestWALCrashRecoveryCorruptedCRC(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "corrupted_crc.wal")

	wal, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	for i := 0; i < 5; i++ {
		key := fmt.Sprintf("crc_test_%03d", i)
		_ = wal.Append(lsm.LogEntry{
			Type:     lsm.RecordPut,
			Key:      []byte(key),
			ValuePtr: lsm.ValuePtr{FileID: 1, Offset: uint64(i * 20), Len: 20},
		})
	}
	wal.Close()

	// Corrupt a byte in the 5th record's payload
	data, err := os.ReadFile(walPath)
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}
	// Flip the last byte of the file
	data[len(data)-1] ^= 0xFF
	if err := os.WriteFile(walPath, data, 0644); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	walRec, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open: %v", err)
	}
	defer walRec.Close()

	entries, err := walRec.Recover()
	if err != nil {
		t.Fatalf("expected clean halt on CRC corruption, got error: %v", err)
	}

	// Must halt cleanly and return the 4 uncorrupted prefix entries
	if len(entries) != 4 {
		t.Fatalf("expected 4 valid entries before corruption, got %d", len(entries))
	}
}

// TestSkipListSequential verifies basic operations, updates, tombstones, and iteration.
func TestSkipListSequential(t *testing.T) {
	sl := lsm.NewSkipList()

	// Insert elements
	keys := []string{"banana", "apple", "cherry", "date", "elderberry"}
	for i, k := range keys {
		sl.Insert([]byte(k), lsm.ValuePtr{FileID: 1, Offset: uint64(i * 10), Len: 10}, false)
	}

	if sl.Len() != len(keys) {
		t.Fatalf("expected length %d, got %d", len(keys), sl.Len())
	}

	// Get elements
	val, ok := sl.Get([]byte("apple"))
	if !ok || val.Offset != 10 {
		t.Fatalf("expected apple at offset 10, got ok=%v, val=%+v", ok, val)
	}

	// Non-existent key
	_, ok = sl.Get([]byte("fig"))
	if ok {
		t.Fatalf("expected fig not found")
	}

	// Update existing key
	sl.Insert([]byte("apple"), lsm.ValuePtr{FileID: 1, Offset: 999, Len: 10}, false)
	val, ok = sl.Get([]byte("apple"))
	if !ok || val.Offset != 999 {
		t.Fatalf("expected updated offset 999, got ok=%v, val=%+v", ok, val)
	}
	if sl.Len() != len(keys) {
		t.Fatalf("expected length unchanged after update, got %d", sl.Len())
	}

	// Delete key (tombstone)
	sl.Insert([]byte("cherry"), lsm.ValuePtr{}, true)
	_, ok = sl.Get([]byte("cherry"))
	if ok {
		t.Fatalf("expected cherry to be marked deleted")
	}

	// Verify GetRecord exposes deletion
	vp, isDeleted, found := sl.GetRecord([]byte("cherry"))
	if !found || !isDeleted {
		t.Fatalf("expected cherry found as tombstone, got found=%v, isDeleted=%v, vp=%+v", found, isDeleted, vp)
	}

	// In-order iteration
	expectedSorted := []string{"apple", "banana", "cherry", "date", "elderberry"}
	it := sl.Iterator()
	var iterated []string
	for it.Next() {
		iterated = append(iterated, string(it.Key()))
	}

	if len(iterated) != len(expectedSorted) {
		t.Fatalf("expected %d iterated items, got %d", len(expectedSorted), len(iterated))
	}
	for i, exp := range expectedSorted {
		if iterated[i] != exp {
			t.Errorf("at index %d: expected %s, got %s", i, exp, iterated[i])
		}
	}

	// Test Seek
	it.Seek([]byte("date"))
	if !it.Valid() || string(it.Key()) != "date" {
		t.Fatalf("expected seek to land on date, got %s", it.Key())
	}
}

// TestSkipListConcurrentWritesAndOrdering verifies thread-safety and sorted ordering
// under intensive concurrent read/write stress.
func TestSkipListConcurrentWritesAndOrdering(t *testing.T) {
	sl := lsm.NewSkipList()
	const numGoroutines = 16
	const itemsPerGoroutine = 1000
	const totalItems = numGoroutines * itemsPerGoroutine

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Concurrently insert items with randomly shuffled keys
	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			rnd := rand.New(rand.NewSource(time.Now().UnixNano() + int64(gid)))

			keys := make([]int, itemsPerGoroutine)
			for i := 0; i < itemsPerGoroutine; i++ {
				keys[i] = gid*itemsPerGoroutine + i
			}
			rnd.Shuffle(len(keys), func(i, j int) { keys[i], keys[j] = keys[j], keys[i] })

			for _, k := range keys {
				keyStr := fmt.Sprintf("key_%08d", k)
				sl.Insert([]byte(keyStr), lsm.ValuePtr{
					FileID: uint32(gid),
					Offset: uint64(k),
					Len:    64,
				}, false)
			}
		}(g)
	}

	// Concurrent readers reading during active writes
	stopReaders := make(chan struct{})
	var readerWg sync.WaitGroup
	for r := 0; r < 4; r++ {
		readerWg.Add(1)
		go func(rid int) {
			defer readerWg.Done()
			rnd := rand.New(rand.NewSource(time.Now().UnixNano() + int64(rid*1000)))
			for {
				select {
				case <-stopReaders:
					return
				default:
					randomKey := fmt.Sprintf("key_%08d", rnd.Intn(totalItems))
					sl.Get([]byte(randomKey))
					// Also test concurrent iterator creation
					if rnd.Intn(100) == 0 {
						it := sl.Iterator()
						if it.Next() {
							_ = it.Key()
						}
					}
				}
			}
		}(r)
	}

	wg.Wait()
	close(stopReaders)
	readerWg.Wait()

	if sl.Len() != totalItems {
		t.Fatalf("expected %d items, got %d", totalItems, sl.Len())
	}

	// Verify STRICT ascending order across entire SkipList
	it := sl.Iterator()
	var prevKey []byte
	count := 0

	for it.Next() {
		currKey := it.Key()
		if prevKey != nil {
			if bytes.Compare(prevKey, currKey) >= 0 {
				t.Fatalf("ordering violation: %s >= %s", prevKey, currKey)
			}
		}
		prevKey = bytes.Clone(currKey)
		count++
	}

	if count != totalItems {
		t.Fatalf("expected iterator count %d, got %d", totalItems, count)
	}

	// Spot-check random keys
	for i := 0; i < 100; i++ {
		k := rand.Intn(totalItems)
		keyStr := fmt.Sprintf("key_%08d", k)
		val, ok := sl.Get([]byte(keyStr))
		if !ok {
			t.Fatalf("expected key %s to exist", keyStr)
		}
		if val.Offset != uint64(k) {
			t.Fatalf("expected offset %d, got %d", k, val.Offset)
		}
	}
}

// TestMemTableByteCounterAndFlushThreshold verifies byte-size accounting and 4MB/8MB detection.
func TestMemTableByteCounterAndFlushThreshold(t *testing.T) {
	tmpDir := t.TempDir()
	walPath := filepath.Join(tmpDir, "memtable.wal")

	wal, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	mem := lsm.NewMemTable(wal)
	defer mem.Close()

	if mem.IsThresholdReached(lsm.DefaultFlushThreshold) {
		t.Fatalf("new memtable should not reach 4MB threshold")
	}

	// Each entry size = len(key) + ValuePtrSize (16)
	// We'll write entries until 4MB is reached
	const keyPrefix = "mem_test_key_with_sufficient_padding_for_testing_"
	entryIndex := 0

	for !mem.IsThresholdReached(lsm.DefaultFlushThreshold) {
		key := fmt.Sprintf("%s%08d", keyPrefix, entryIndex)
		err := mem.Put([]byte(key), lsm.ValuePtr{
			FileID: 1,
			Offset: uint64(entryIndex * 100),
			Len:    100,
		})
		if err != nil {
			t.Fatalf("failed to put in memtable: %v", err)
		}
		entryIndex++
	}

	currentSize := mem.SizeBytes()
	if currentSize < lsm.DefaultFlushThreshold {
		t.Fatalf("expected size >= 4MB (%d), got %d", lsm.DefaultFlushThreshold, currentSize)
	}

	if !mem.IsThresholdReached(lsm.DefaultFlushThreshold) {
		t.Fatalf("expected 4MB threshold reached")
	}

	// Continue writing until 8MB threshold
	for !mem.IsThresholdReached(lsm.MaxFlushThreshold) {
		key := fmt.Sprintf("%s%08d", keyPrefix, entryIndex)
		_ = mem.Put([]byte(key), lsm.ValuePtr{
			FileID: 1,
			Offset: uint64(entryIndex * 100),
			Len:    100,
		})
		entryIndex++
	}

	if mem.SizeBytes() < lsm.MaxFlushThreshold {
		t.Fatalf("expected size >= 8MB (%d), got %d", lsm.MaxFlushThreshold, mem.SizeBytes())
	}
	if !mem.IsThresholdReached(lsm.MaxFlushThreshold) {
		t.Fatalf("expected 8MB max threshold reached")
	}

	// Test Get from MemTable
	testKey := fmt.Sprintf("%s%08d", keyPrefix, 10)
	vp, err := mem.Get([]byte(testKey))
	if err != nil {
		t.Fatalf("failed to get key from memtable: %v", err)
	}
	if vp.Offset != 1000 {
		t.Fatalf("expected offset 1000, got %d", vp.Offset)
	}

	// Test Delete in MemTable
	if err := mem.Delete([]byte(testKey)); err != nil {
		t.Fatalf("failed to delete key: %v", err)
	}
	_, err = mem.Get([]byte(testKey))
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected ErrKeyNotFound after delete, got %v", err)
	}

	// Flush WAL and recover into a brand new MemTable
	if err := mem.Sync(); err != nil {
		t.Fatalf("sync failed: %v", err)
	}
	wal.Close()

	walForRecovery, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("failed to reopen WAL: %v", err)
	}
	defer walForRecovery.Close()

	recoveredMem, err := lsm.NewMemTableFromWAL(walForRecovery)
	if err != nil {
		t.Fatalf("failed to recover memtable from WAL: %v", err)
	}

	// Verify recovered state
	if recoveredMem.Len() != mem.Len() {
		t.Fatalf("expected %d entries in recovered memtable, got %d", mem.Len(), recoveredMem.Len())
	}

	// The deleted key should still be not found in recovered memtable
	_, err = recoveredMem.Get([]byte(testKey))
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected deleted key to return ErrKeyNotFound in recovered memtable, got %v", err)
	}

	// Another key should be found
	otherKey := fmt.Sprintf("%s%08d", keyPrefix, 20)
	vpOther, err := recoveredMem.Get([]byte(otherKey))
	if err != nil {
		t.Fatalf("expected key %s found in recovered memtable, got err %v", otherKey, err)
	}
	if vpOther.Offset != 2000 {
		t.Fatalf("expected offset 2000, got %d", vpOther.Offset)
	}
}
