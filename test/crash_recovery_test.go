package test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"lsmtree/pkg/lsm"
)

// TestCrashRecovery_AbruptKillWithTornWALWrite simulates a sudden power loss
// or kernel panic mid-write. It verifies:
// a) DB restarts without panic.
// b) Zero corruption of previously acknowledged transactions across SSTables and WAL.
// c) The uncommitted / torn write tail record is cleanly dropped via CRC validation.
func TestCrashRecovery_AbruptKillWithTornWALWrite(t *testing.T) {
	dir := t.TempDir()

	db, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 16,
		CompactionThreshold:   5,
		EnableCompaction:      false,
		VLogFileID:            1,
	})
	if err != nil {
		t.Fatalf("OpenDB failed: %v", err)
	}

	// 1. Write 2,000 keys and flush to create SSTable 1 & 2
	committedKeys := make(map[string]string)
	for i := 0; i < 1000; i++ {
		k := fmt.Sprintf("persisted_user_%06d", i)
		v := fmt.Sprintf("profile_value_tier1_%06d", i)
		if err := db.Put([]byte(k), []byte(v)); err != nil {
			t.Fatalf("Put failed: %v", err)
		}
		committedKeys[k] = v
	}
	if _, err := db.Flush(); err != nil {
		t.Fatalf("Flush 1 failed: %v", err)
	}

	for i := 1000; i < 2000; i++ {
		k := fmt.Sprintf("persisted_user_%06d", i)
		v := fmt.Sprintf("profile_value_tier2_%06d", i)
		if err := db.Put([]byte(k), []byte(v)); err != nil {
			t.Fatalf("Put failed: %v", err)
		}
		committedKeys[k] = v
	}
	if _, err := db.Flush(); err != nil {
		t.Fatalf("Flush 2 failed: %v", err)
	}

	// 2. Write 500 keys to Active MemTable (backed by active WAL)
	for i := 2000; i < 2500; i++ {
		k := fmt.Sprintf("persisted_user_%06d", i)
		v := fmt.Sprintf("profile_value_active_%06d", i)
		if err := db.Put([]byte(k), []byte(v)); err != nil {
			t.Fatalf("Put failed: %v", err)
		}
		committedKeys[k] = v
	}

	// 3. Find the active WAL file on disk
	walDir := filepath.Join(dir, "wal")
	walFiles, err := filepath.Glob(filepath.Join(walDir, "wal_*.wal"))
	if err != nil || len(walFiles) == 0 {
		t.Fatalf("failed finding active WAL files: %v", err)
	}
	sort.Strings(walFiles)
	activeWALPath := walFiles[len(walFiles)-1]

	statBefore, err := os.Stat(activeWALPath)
	if err != nil {
		t.Fatalf("stat active WAL failed: %v", err)
	}
	validSizeBeforeCorruption := statBefore.Size()

	// 4. Simulate ABRUPT CRASH (power cut / SIGKILL):
	// Do NOT call db.Close() - leaving OS buffers unflushed and files open.
	// Inject a torn write: incomplete record header + partial garbage bytes
	tornGarbage := []byte{0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x00, 0x08, 0xAA, 0xBB}
	if err := InjectTornWrite(activeWALPath, tornGarbage); err != nil {
		t.Fatalf("failed injecting torn write: %v", err)
	}

	statAfter, err := os.Stat(activeWALPath)
	if err != nil || statAfter.Size() <= validSizeBeforeCorruption {
		t.Fatalf("expected WAL size to grow with injected garbage: before=%d, after=%d",
			validSizeBeforeCorruption, statAfter.Size())
	}

	// 5. Reopen the database on the exact same directory
	dbReopened, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 16,
		CompactionThreshold:   5,
		EnableCompaction:      false,
		VLogFileID:            1,
	})
	if err != nil {
		t.Fatalf("OpenDB failed after crash: %v", err)
	}
	defer dbReopened.Close()

	// a) Verify active WAL was truncated to remove the corrupted tail bytes
	statReopened, err := os.Stat(activeWALPath)
	if err != nil {
		t.Fatalf("stat reopened WAL failed: %v", err)
	}
	if statReopened.Size() != validSizeBeforeCorruption {
		t.Fatalf("WAL was not truncated to valid offset: got %d, expected %d",
			statReopened.Size(), validSizeBeforeCorruption)
	}

	// b) Verify ZERO corruption for all 2,500 previously acknowledged transactions
	for k, expectedVal := range committedKeys {
		val, err := dbReopened.Get([]byte(k))
		if err != nil {
			t.Fatalf("Get failed for acknowledged key %s: %v", k, err)
		}
		if string(val) != expectedVal {
			t.Fatalf("data corruption for key %s: got %s, expected %s", k, string(val), expectedVal)
		}
	}

	// c) Verify new writes succeed on reopened engine
	newKey := []byte("new_key_post_recovery_0001")
	newVal := []byte("new_val_post_recovery_0001")
	if err := dbReopened.Put(newKey, newVal); err != nil {
		t.Fatalf("Put failed on reopened DB: %v", err)
	}

	retrieved, err := dbReopened.Get(newKey)
	if err != nil || !bytes.Equal(retrieved, newVal) {
		t.Fatalf("failed retrieving newly written key on reopened DB: %v", err)
	}

	t.Logf("Crash recovery succeeded: verified %d acknowledged keys intact, torn write truncated", len(committedKeys))
}

// TestCrashRecovery_CorruptedTailCRC verifies that if the CRC32 of the last record
// is corrupted (e.g., bit-rot or partial disk write), the engine halts recovery
// cleanly, drops the corrupted record, and preserves all preceding transactions.
func TestCrashRecovery_CorruptedTailCRC(t *testing.T) {
	dir := t.TempDir()

	walPath := filepath.Join(dir, "wal_000001.wal")
	wal, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("OpenWAL failed: %v", err)
	}

	count := 100
	var lastRecordOffset int64

	// Write 100 entries to WAL
	for i := 0; i < count; i++ {
		key := []byte(fmt.Sprintf("wal_key_%04d", i))
		vp := lsm.ValuePtr{FileID: 1, Offset: uint64(i * 100), Len: 50}

		if i == count-1 {
			lastRecordOffset = wal.Size()
		}

		if err := wal.Append(lsm.LogEntry{
			Type:     lsm.RecordPut,
			Key:      key,
			ValuePtr: vp,
		}); err != nil {
			t.Fatalf("wal.Append failed: %v", err)
		}
	}
	_ = wal.Close()

	// Corrupt the CRC32 of the 100th record
	if err := CorruptFileTailCRC(walPath, lastRecordOffset); err != nil {
		t.Fatalf("CorruptFileTailCRC failed: %v", err)
	}

	// Reopen WAL and execute recovery
	walReopened, err := lsm.OpenWAL(walPath)
	if err != nil {
		t.Fatalf("OpenWAL failed on corrupted WAL: %v", err)
	}
	defer walReopened.Close()

	recoveredEntries, err := walReopened.Recover()
	if err != nil {
		t.Fatalf("Recover failed: %v", err)
	}

	// First 99 records must be recovered cleanly; the 100th corrupted record must be dropped
	if len(recoveredEntries) != count-1 {
		t.Fatalf("expected %d recovered entries, got %d", count-1, len(recoveredEntries))
	}

	for i := 0; i < count-1; i++ {
		expectedKey := []byte(fmt.Sprintf("wal_key_%04d", i))
		if !bytes.Equal(recoveredEntries[i].Key, expectedKey) {
			t.Fatalf("recovered key mismatch at index %d: got %s, expected %s",
				i, string(recoveredEntries[i].Key), string(expectedKey))
		}
	}

	// Check file was truncated to last valid record
	if walReopened.Size() != lastRecordOffset {
		t.Fatalf("expected WAL size to be truncated to %d, got %d", lastRecordOffset, walReopened.Size())
	}
}

// TestCrashRecovery_FaultFilePanicSimulation tests the FaultFile wrapper directly:
// verifies that sudden panics during writes leave earlier writes recoverable.
func TestCrashRecovery_FaultFilePanicSimulation(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "panic_test.wal")

	rawFile, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}

	faultFile := NewFaultFile(rawFile)
	// Allow 500 bytes before panicking abruptly
	faultFile.PanicAfterNBytes(500)

	didPanic := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				didPanic = true
			}
		}()

		// Write in 120-byte chunks until limit is exceeded
		chunk := make([]byte, 120)
		for i := range chunk {
			chunk[i] = byte(i)
		}

		for {
			_, err := faultFile.Write(chunk)
			if err != nil {
				break
			}
		}
	}()

	if !didPanic {
		t.Fatalf("expected FaultFile to panic mid-write")
	}

	_ = rawFile.Close()

	// Verify file on disk has exactly 500 bytes (halted abruptly at boundary)
	stat, err := os.Stat(filePath)
	if err != nil {
		t.Fatalf("stat failed: %v", err)
	}
	if stat.Size() != 500 {
		t.Fatalf("expected file size 500 bytes, got %d", stat.Size())
	}
}

// TestCrashRecovery_ContinuousWritesAndConcurrentFlushes simulates a realistic
// high-concurrency production workload where writes, MemTable rotations, and
// SSTable flushes occur simultaneously, followed by an abrupt simulated crash.
func TestCrashRecovery_ContinuousWritesAndConcurrentFlushes(t *testing.T) {
	dir := t.TempDir()

	db, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 16,
		CompactionThreshold:   5,
		EnableCompaction:      true,
		VLogFileID:            1,
	})
	if err != nil {
		t.Fatalf("OpenDB failed: %v", err)
	}

	var committedMu sync.Mutex
	committed := make(map[string]string)

	var stopWorkers int32
	var wg sync.WaitGroup

	numWriters := 4
	totalWritesPerWorker := 750

	// Launch concurrent write workers
	for w := 0; w < numWriters; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < totalWritesPerWorker; i++ {
				if atomic.LoadInt32(&stopWorkers) == 1 {
					return
				}
				key := fmt.Sprintf("txn_key_w%d_%06d", workerID, i)
				val := fmt.Sprintf("txn_val_w%d_%06d_%d", workerID, i, time.Now().UnixNano())

				if err := db.Put([]byte(key), []byte(val)); err != nil {
					return
				}

				committedMu.Lock()
				committed[key] = val
				committedMu.Unlock()
			}
		}(w)
	}

	// Periodically trigger synchronous flushes concurrently
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 5; i++ {
			if atomic.LoadInt32(&stopWorkers) == 1 {
				return
			}
			time.Sleep(15 * time.Millisecond)
			_, _ = db.Flush()
		}
	}()

	// Wait for workers to complete or reach steady state
	wg.Wait()

	// Simulate crash: do NOT close DB gracefully
	committedMu.Lock()
	snapshotCommitted := make(map[string]string, len(committed))
	for k, v := range committed {
		snapshotCommitted[k] = v
	}
	committedMu.Unlock()

	// Inject partial corrupted write to active WAL
	walFiles, _ := filepath.Glob(filepath.Join(dir, "wal", "wal_*.wal"))
	if len(walFiles) > 0 {
		sort.Strings(walFiles)
		activeWAL := walFiles[len(walFiles)-1]
		_ = InjectTornWrite(activeWAL, []byte{0xDE, 0xAD, 0xBE, 0xEF})
	}

	// Reopen engine
	dbRecovered, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 16,
		CompactionThreshold:   5,
		EnableCompaction:      true,
		VLogFileID:            1,
	})
	if err != nil {
		t.Fatalf("OpenDB failed after concurrent crash: %v", err)
	}
	defer dbRecovered.Close()

	// Assert zero corruption of all committed transactions
	corruptedCount := 0
	for k, expectedVal := range snapshotCommitted {
		val, err := dbRecovered.Get([]byte(k))
		if err != nil {
			t.Fatalf("Get failed for acknowledged key %s: %v", k, err)
		}
		if string(val) != expectedVal {
			corruptedCount++
		}
	}

	if corruptedCount > 0 {
		t.Fatalf("detected %d corrupted transactions after recovery", corruptedCount)
	}

	t.Logf("Concurrent crash recovery passed: verified all %d acknowledged transactions intact", len(snapshotCommitted))
}
