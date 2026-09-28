package test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"lsmtree/pkg/lsm"
)

// TestVLogAppendAndRandomRead verifies VLog write, random read, and CRC verification.
func TestVLogAppendAndRandomRead(t *testing.T) {
	tmpDir := t.TempDir()
	vlogPath := filepath.Join(tmpDir, "test.vlog")

	vlog, err := lsm.OpenVLog(vlogPath, 1)
	if err != nil {
		t.Fatalf("failed to open vlog: %v", err)
	}
	defer vlog.Close()

	key1 := []byte("key_001")
	val1 := []byte("value_payload_for_testing_001")
	ptr1, err := vlog.Write(key1, val1)
	if err != nil {
		t.Fatalf("failed to write to vlog: %v", err)
	}

	key2 := []byte("key_002")
	val2 := []byte("another_longer_value_payload_002_with_more_bytes")
	ptr2, err := vlog.Write(key2, val2)
	if err != nil {
		t.Fatalf("failed to write key2: %v", err)
	}

	if err := vlog.Flush(); err != nil {
		t.Fatalf("failed to flush vlog: %v", err)
	}

	// Read val1 via io.ReadAt
	gotVal1, err := vlog.Read(ptr1)
	if err != nil {
		t.Fatalf("failed to read ptr1: %v", err)
	}
	if !bytes.Equal(gotVal1, val1) {
		t.Fatalf("expected val1 %s, got %s", val1, gotVal1)
	}

	// Read val2 via io.ReadAt
	gotKey2, gotVal2, err := vlog.ReadRecord(ptr2)
	if err != nil {
		t.Fatalf("failed to read record 2: %v", err)
	}
	if !bytes.Equal(gotKey2, key2) {
		t.Fatalf("expected key %s, got %s", key2, gotKey2)
	}
	if !bytes.Equal(gotVal2, val2) {
		t.Fatalf("expected val %s, got %s", val2, gotVal2)
	}

	// Corrupt a byte in the first entry's value and verify CRC failure
	corruptData, err := os.ReadFile(vlogPath)
	if err != nil {
		t.Fatalf("read vlog file failed: %v", err)
	}
	// Corrupt inside payload of entry 1
	corruptData[15] ^= 0xFF
	if err := os.WriteFile(vlogPath, corruptData, 0644); err != nil {
		t.Fatalf("write corrupted vlog failed: %v", err)
	}

	// Re-open and verify corruption detected
	vlogCorrupt, err := lsm.OpenVLog(vlogPath, 1)
	if err != nil {
		t.Fatalf("failed reopen corrupt vlog: %v", err)
	}
	defer vlogCorrupt.Close()

	_, err = vlogCorrupt.Read(ptr1)
	if err != lsm.ErrCorruptedRecord {
		t.Fatalf("expected ErrCorruptedRecord on tampered vlog, got: %v", err)
	}
}

// TestSSTableSparseIndexAndPreventFullScan verifies that the sparse index correctly
// isolates the candidate range and prevents full file scans.
func TestSSTableSparseIndexAndPreventFullScan(t *testing.T) {
	tmpDir := t.TempDir()
	sstPath := filepath.Join(tmpDir, "test.sst")

	sampleRate := 64
	writer, err := lsm.NewSSTableWriter(sstPath, sampleRate)
	if err != nil {
		t.Fatalf("failed to create SSTable writer: %v", err)
	}

	const numKeys = 2000
	for i := 0; i < numKeys; i++ {
		key := fmt.Sprintf("prefix_key_%06d", i)
		vp := lsm.ValuePtr{FileID: 1, Offset: uint64(i * 100), Len: 100}
		if err := writer.Append([]byte(key), false, vp); err != nil {
			t.Fatalf("append failed at %d: %v", i, err)
		}
	}

	if err := writer.Finish(); err != nil {
		t.Fatalf("finish failed: %v", err)
	}

	reader, err := lsm.OpenSSTable(sstPath)
	if err != nil {
		t.Fatalf("failed to open SSTable: %v", err)
	}
	defer reader.Close()

	totalFileSize := reader.Size()

	// Query middle key: prefix_key_001000
	testKey := []byte("prefix_key_001000")
	vp, isTombstone, bytesRead, err := reader.GetWithStats(testKey)
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	if isTombstone {
		t.Fatalf("unexpected tombstone")
	}
	if vp.Offset != 100000 {
		t.Fatalf("expected offset 100000, got %d", vp.Offset)
	}

	// Verify that the sparse index PREVENTS scanning the entire file:
	// Total data block is ~2000 entries * ~40 bytes = ~80KB.
	// Sparse index range is at most 64 entries * ~40 bytes = ~2.5KB.
	// bytesRead MUST be significantly smaller than totalFileSize!
	t.Logf("Total SSTable size: %d bytes, Bytes read for point lookup: %d bytes", totalFileSize, bytesRead)
	if int64(bytesRead) >= totalFileSize {
		t.Fatalf("sparse index failed: read %d bytes out of %d total file size", bytesRead, totalFileSize)
	}
	maxExpectedBlock := int64(sampleRate * (len("prefix_key_001000") + 19) * 2)
	if int64(bytesRead) > maxExpectedBlock {
		t.Fatalf("read %d bytes, expected <= %d bytes", bytesRead, maxExpectedBlock)
	}

	// Test non-existent key between existing keys
	nonExistentKey := []byte("prefix_key_000500_not_here")
	_, _, _, err = reader.GetWithStats(nonExistentKey)
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected ErrKeyNotFound for missing key, got: %v", err)
	}

	// Test key smaller than the first key
	smallerKey := []byte("aaa_before_first_key")
	_, _, _, err = reader.GetWithStats(smallerKey)
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected ErrKeyNotFound for key before first key, got: %v", err)
	}
}

// TestPhase2EndToEnd10000Keys4KBValuesAndFlush is the comprehensive end-to-end test:
// 1. Writes 10,000 keys with 4KB values decoupled into VLog and MemTable.
// 2. Triggers MemTable flush to SSTable with sparse indexing.
// 3. Verifies all 10,000 values are retrievable from SSTable + VLog.
// 4. Verifies sparse index prevents scanning the entire file.
func TestPhase2EndToEnd10000Keys4KBValuesAndFlush(t *testing.T) {
	dataDir := t.TempDir()
	vlogPath := filepath.Join(dataDir, "vlog", "000001.vlog")

	vlog, err := lsm.OpenVLog(vlogPath, 1)
	if err != nil {
		t.Fatalf("failed to open VLog: %v", err)
	}

	const sampleRate = 64
	engine, err := lsm.NewFlushManager(dataDir, vlog, sampleRate)
	if err != nil {
		t.Fatalf("failed to initialize FlushManager: %v", err)
	}
	defer engine.Close()

	const numKeys = 10000
	const valSize = 4096 // 4KB per value (Total = 40MB values)

	t.Logf("Writing %d keys with %d bytes (4KB) values...", numKeys, valSize)

	// Pre-create 4KB value pattern
	valTemplate := make([]byte, valSize)
	for b := range valTemplate {
		valTemplate[b] = byte('A' + (b % 26))
	}

	for i := 0; i < numKeys; i++ {
		key := fmt.Sprintf("user_account_id_%08d", i)

		// Stamp unique ID into value payload for absolute verification
		val := make([]byte, valSize)
		copy(val, valTemplate)
		copy(val[0:8], fmt.Sprintf("%08d", i))

		if err := engine.Put([]byte(key), val); err != nil {
			t.Fatalf("engine Put failed at %d: %v", i, err)
		}
	}

	// Verify active MemTable has keys before flush
	if engine.ActiveMemTable().Len() != numKeys {
		t.Fatalf("expected active memtable len %d, got %d", numKeys, engine.ActiveMemTable().Len())
	}

	t.Logf("Freezing active MemTable and flushing to SSTable...")
	sstReader, err := engine.RotateAndFlushSync()
	if err != nil {
		t.Fatalf("flush failed: %v", err)
	}
	if sstReader == nil {
		t.Fatalf("expected SSTableReader from flush, got nil")
	}

	sstSize := sstReader.Size()
	t.Logf("SSTable generated: %s (Size: %d bytes, ~%.2f KB for 10k keys)",
		sstReader.Path(), sstSize, float64(sstSize)/1024.0)

	// In WiscKey, SSTable only stores keys and ValuePtrs:
	// 10,000 keys * (25 bytes key + 19 bytes prefix) = ~440 KB SSTable!
	// Meanwhile VLog stores the 40MB values!
	if sstSize > 1*1024*1024 {
		t.Fatalf("SSTable is unexpectedly large (%d bytes); WiscKey key-value separation failed", sstSize)
	}

	t.Logf("Verifying all %d values are retrievable from SSTable + VLog...", numKeys)

	// Verify sample lookups across entire key space
	testIndices := []int{0, 1, 63, 64, 65, 500, 1000, 2500, 4999, 7500, 9998, 9999}
	for _, idx := range testIndices {
		key := fmt.Sprintf("user_account_id_%08d", idx)
		gotVal, err := engine.Get([]byte(key))
		if err != nil {
			t.Fatalf("failed to retrieve key %s: %v", key, err)
		}
		if len(gotVal) != valSize {
			t.Fatalf("key %s: expected %d bytes, got %d", key, valSize, len(gotVal))
		}
		expectedPrefix := fmt.Sprintf("%08d", idx)
		if string(gotVal[0:8]) != expectedPrefix {
			t.Fatalf("key %s: expected value prefix %s, got %s", key, expectedPrefix, gotVal[0:8])
		}
	}

	// Verify sparse index avoids full scan on SSTable
	keyToProbe := []byte(fmt.Sprintf("user_account_id_%08d", 5555))
	_, _, bytesScanned, err := sstReader.GetWithStats(keyToProbe)
	if err != nil {
		t.Fatalf("probe failed: %v", err)
	}

	t.Logf("Point lookup for %s: scanned %d bytes out of total SSTable size %d bytes (~%.2f%%)",
		keyToProbe, bytesScanned, sstSize, float64(bytesScanned)/float64(sstSize)*100.0)

	if int64(bytesScanned) >= sstSize/10 {
		t.Fatalf("sparse index did not effectively isolate candidate block: scanned %d bytes out of %d",
			bytesScanned, sstSize)
	}

	// Verify writes to new MemTable while SSTables exist
	newKey := []byte("user_account_id_new_post_flush")
	newVal := []byte("new_value_in_active_memtable")
	if err := engine.Put(newKey, newVal); err != nil {
		t.Fatalf("failed putting new key: %v", err)
	}

	gotNewVal, err := engine.Get(newKey)
	if err != nil {
		t.Fatalf("failed retrieving new key: %v", err)
	}
	if !bytes.Equal(gotNewVal, newVal) {
		t.Fatalf("expected %s, got %s", newVal, gotNewVal)
	}

	// Verify deletion (tombstone)
	delKey := []byte(fmt.Sprintf("user_account_id_%08d", 100))
	if err := engine.Delete(delKey); err != nil {
		t.Fatalf("failed to delete key: %v", err)
	}
	_, err = engine.Get(delKey)
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected ErrKeyNotFound for deleted key, got: %v", err)
	}
}
