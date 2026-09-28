package test

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"lsmtree/pkg/lsm"
)

// TestKWayMergeUnit directly tests MergeSSTables with deduplication, ordering,
// and tombstone purge logic.
func TestKWayMergeUnit(t *testing.T) {
	tmpDir := t.TempDir()

	// SSTable 1 (oldest, ID=1): keys [a, b, c, d]
	sstPath1 := filepath.Join(tmpDir, "000001.sst")
	w1, _ := lsm.NewSSTableWriter(sstPath1, 16)
	_ = w1.Append([]byte("a"), false, lsm.ValuePtr{FileID: 1, Offset: 10, Len: 10})
	_ = w1.Append([]byte("b"), false, lsm.ValuePtr{FileID: 1, Offset: 20, Len: 10})
	_ = w1.Append([]byte("c"), false, lsm.ValuePtr{FileID: 1, Offset: 30, Len: 10})
	_ = w1.Append([]byte("d"), false, lsm.ValuePtr{FileID: 1, Offset: 40, Len: 10})
	_ = w1.Finish()
	r1, _ := lsm.OpenSSTable(sstPath1)
	r1.SetID(1)
	defer r1.Close()

	// SSTable 2 (middle, ID=2): keys [b (updated), c (deleted), e (new)]
	sstPath2 := filepath.Join(tmpDir, "000002.sst")
	w2, _ := lsm.NewSSTableWriter(sstPath2, 16)
	_ = w2.Append([]byte("b"), false, lsm.ValuePtr{FileID: 1, Offset: 200, Len: 10})
	_ = w2.Append([]byte("c"), true, lsm.ValuePtr{}) // tombstone
	_ = w2.Append([]byte("e"), false, lsm.ValuePtr{FileID: 1, Offset: 50, Len: 10})
	_ = w2.Finish()
	r2, _ := lsm.OpenSSTable(sstPath2)
	r2.SetID(2)
	defer r2.Close()

	// SSTable 3 (newest, ID=3): keys [a (deleted), d (updated)]
	sstPath3 := filepath.Join(tmpDir, "000003.sst")
	w3, _ := lsm.NewSSTableWriter(sstPath3, 16)
	_ = w3.Append([]byte("a"), true, lsm.ValuePtr{}) // tombstone
	_ = w3.Append([]byte("d"), false, lsm.ValuePtr{FileID: 1, Offset: 400, Len: 10})
	_ = w3.Finish()
	r3, _ := lsm.OpenSSTable(sstPath3)
	r3.SetID(3)
	defer r3.Close()

	// Merge all 3 SSTables with isOldestHistorical = true (tombstones purged)
	mergedPath := filepath.Join(tmpDir, "merged_000004.sst")
	mergedReader, err := lsm.MergeSSTables([]*lsm.SSTableReader{r1, r2, r3}, mergedPath, 16, true)
	if err != nil {
		t.Fatalf("merge failed: %v", err)
	}
	defer mergedReader.Close()

	// "a" was deleted in newest table (r3) -> should be purged completely
	_, _, err = mergedReader.Get([]byte("a"))
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected key 'a' to be purged, got err: %v", err)
	}

	// "b" was updated in r2 to offset 200 -> should return offset 200
	vpB, isTombB, err := mergedReader.Get([]byte("b"))
	if err != nil || isTombB || vpB.Offset != 200 {
		t.Fatalf("expected key 'b' at offset 200, got: vp=%+v, tomb=%v, err=%v", vpB, isTombB, err)
	}

	// "c" was deleted in r2 -> should be purged completely
	_, _, err = mergedReader.Get([]byte("c"))
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected key 'c' to be purged, got err: %v", err)
	}

	// "d" was updated in r3 to offset 400 -> should return offset 400
	vpD, isTombD, err := mergedReader.Get([]byte("d"))
	if err != nil || isTombD || vpD.Offset != 400 {
		t.Fatalf("expected key 'd' at offset 400, got: vp=%+v, tomb=%v, err=%v", vpD, isTombD, err)
	}

	// "e" was inserted in r2 -> should return offset 50
	vpE, isTombE, err := mergedReader.Get([]byte("e"))
	if err != nil || isTombE || vpE.Offset != 50 {
		t.Fatalf("expected key 'e' at offset 50, got: vp=%+v, tomb=%v, err=%v", vpE, isTombE, err)
	}

	// Verify in-order traversal of merged table
	it, err := mergedReader.NewIterator()
	if err != nil {
		t.Fatalf("failed creating iterator: %v", err)
	}
	var keysSeen []string
	for it.Next() {
		keysSeen = append(keysSeen, string(it.Key()))
	}
	expectedKeys := []string{"b", "d", "e"}
	if len(keysSeen) != len(expectedKeys) {
		t.Fatalf("expected %v, got %v", expectedKeys, keysSeen)
	}
	for i, k := range expectedKeys {
		if keysSeen[i] != k {
			t.Errorf("at index %d: expected %s, got %s", i, k, keysSeen[i])
		}
	}
}

// TestPhase3Overwrite1000Keys5TimesAndCompaction fulfills the exact user requirement:
// "Overwrite the same 1,000 keys 5 times across 5 flushes. Assert file count drops
// post-compaction and dead keys are purged."
func TestPhase3Overwrite1000Keys5TimesAndCompaction(t *testing.T) {
	tmpDir := t.TempDir()
	vlogPath := filepath.Join(tmpDir, "vlog", "000001.vlog")

	vlog, err := lsm.OpenVLog(vlogPath, 1)
	if err != nil {
		t.Fatalf("failed opening vlog: %v", err)
	}

	engine, err := lsm.NewFlushManager(tmpDir, vlog, 16)
	if err != nil {
		t.Fatalf("failed creating engine: %v", err)
	}
	defer engine.Close()

	const numKeys = 1000
	const numRounds = 5

	t.Logf("Overwriting %d keys %d times across %d flushes...", numKeys, numRounds, numRounds)

	for round := 1; round <= numRounds; round++ {
		valSuffix := fmt.Sprintf("_v%d", round)
		for i := 0; i < numKeys; i++ {
			key := fmt.Sprintf("key_%05d", i)
			val := fmt.Sprintf("val_%05d%s", i, valSuffix)
			if err := engine.Put([]byte(key), []byte(val)); err != nil {
				t.Fatalf("round %d put failed: %v", round, err)
			}
		}

		// In the 5th round, delete the first 100 keys (key_00000 to key_00099)
		if round == numRounds {
			for i := 0; i < 100; i++ {
				delKey := fmt.Sprintf("key_%05d", i)
				if err := engine.Delete([]byte(delKey)); err != nil {
					t.Fatalf("delete failed at %d: %v", i, err)
				}
			}
		}

		// Flush active MemTable to an immutable SSTable on disk
		_, err := engine.RotateAndFlushSync()
		if err != nil {
			t.Fatalf("round %d flush failed: %v", round, err)
		}
	}

	// Verify we initially have 5 SSTables before compaction
	initialSSTs := engine.SSTables()
	if len(initialSSTs) != numRounds {
		t.Fatalf("expected %d SSTables before compaction, got %d", numRounds, len(initialSSTs))
	}
	t.Logf("Before compaction: %d active SSTables on disk", len(initialSSTs))

	// Enable compaction and execute K-Way merge
	compactor := engine.EnableCompaction(4)
	mergedSST, err := compactor.CompactNow()
	if err != nil {
		t.Fatalf("compaction failed: %v", err)
	}
	if mergedSST == nil {
		t.Fatalf("expected compacted SSTable, got nil")
	}

	// ASSERTION 1: File count drops post-compaction (5 -> 1)
	postCompactionSSTs := engine.SSTables()
	t.Logf("Post-compaction: active SSTable count dropped from %d to %d", len(initialSSTs), len(postCompactionSSTs))
	if len(postCompactionSSTs) != 1 {
		t.Fatalf("expected SSTable count to drop to 1, got %d", len(postCompactionSSTs))
	}

	// Verify Manifest tracks exactly 1 active SSTable
	manifestTables := engine.Manifest().ActiveTables()
	if len(manifestTables) != 1 {
		t.Fatalf("expected manifest to track 1 table, got %d", len(manifestTables))
	}

	// ASSERTION 2: Obsolete SSTable files were physically deleted from disk
	for _, oldSST := range initialSSTs {
		if oldSST.ID() == mergedSST.ID() {
			continue
		}
		if _, err := os.Stat(oldSST.Path()); !os.IsNotExist(err) {
			t.Errorf("expected obsolete SSTable %s to be deleted from disk", oldSST.Path())
		}
	}

	// ASSERTION 3: Dead keys (0..99) are completely PURGED from SSTable
	for i := 0; i < 100; i++ {
		deadKey := []byte(fmt.Sprintf("key_%05d", i))
		_, err := engine.Get(deadKey)
		if err != lsm.ErrKeyNotFound {
			t.Errorf("expected dead key %s to be purged, got: %v", deadKey, err)
		}
		// Directly check merged SSTable: tombstone was dropped completely
		_, isTomb, _, err := mergedSST.GetWithStats(deadKey)
		if err != lsm.ErrKeyNotFound {
			t.Errorf("dead key %s should not exist in merged SSTable, got isTomb=%v, err=%v", deadKey, isTomb, err)
		}
	}

	// ASSERTION 4: Remaining 900 keys (100..999) return newest value (v5)
	for i := 100; i < numKeys; i++ {
		key := fmt.Sprintf("key_%05d", i)
		expectedVal := fmt.Sprintf("val_%05d_v5", i)
		gotVal, err := engine.Get([]byte(key))
		if err != nil {
			t.Fatalf("failed retrieving active key %s: %v", key, err)
		}
		if string(gotVal) != expectedVal {
			t.Fatalf("key %s: expected %s, got %s", key, expectedVal, string(gotVal))
		}
	}
}

// TestConcurrentReadsDuringCompaction verifies thread-safe file swapping and ref-counting
// when queries run concurrently while a background compaction deletes old files.
func TestConcurrentReadsDuringCompaction(t *testing.T) {
	tmpDir := t.TempDir()
	vlogPath := filepath.Join(tmpDir, "vlog", "000001.vlog")

	vlog, err := lsm.OpenVLog(vlogPath, 1)
	if err != nil {
		t.Fatalf("failed opening vlog: %v", err)
	}

	engine, err := lsm.NewFlushManager(tmpDir, vlog, 16)
	if err != nil {
		t.Fatalf("failed creating engine: %v", err)
	}
	defer engine.Close()

	compactor := engine.EnableCompaction(4)

	const numKeys = 500
	for round := 1; round <= 4; round++ {
		for i := 0; i < numKeys; i++ {
			k := fmt.Sprintf("concurrent_key_%04d", i)
			v := fmt.Sprintf("val_round_%d", round)
			_ = engine.Put([]byte(k), []byte(v))
		}
		_, _ = engine.RotateAndFlushSync()
	}

	stopReaders := make(chan struct{})
	var readerWg sync.WaitGroup

	// Launch concurrent readers
	for r := 0; r < 8; r++ {
		readerWg.Add(1)
		go func(rid int) {
			defer readerWg.Done()
			rnd := rand.New(rand.NewSource(time.Now().UnixNano() + int64(rid)))
			for {
				select {
				case <-stopReaders:
					return
				default:
					k := fmt.Sprintf("concurrent_key_%04d", rnd.Intn(numKeys))
					val, err := engine.Get([]byte(k))
					if err != nil {
						t.Errorf("concurrent read error: %v", err)
						return
					}
					if len(val) == 0 {
						t.Errorf("empty value read for key %s", k)
						return
					}
				}
			}
		}(r)
	}

	// Trigger compaction concurrently while readers are actively reading
	time.Sleep(10 * time.Millisecond)
	_, err = compactor.CompactNow()
	if err != nil {
		t.Fatalf("compaction during concurrent reads failed: %v", err)
	}

	// Allow readers to read from the post-compaction state
	time.Sleep(20 * time.Millisecond)
	close(stopReaders)
	readerWg.Wait()
}

// TestBackgroundCompactionThresholdWorker verifies automatic background compaction
// triggered via channel notification when SSTable count reaches threshold (>= 4).
func TestBackgroundCompactionThresholdWorker(t *testing.T) {
	tmpDir := t.TempDir()
	vlogPath := filepath.Join(tmpDir, "vlog", "000001.vlog")

	vlog, err := lsm.OpenVLog(vlogPath, 1)
	if err != nil {
		t.Fatalf("failed opening vlog: %v", err)
	}

	engine, err := lsm.NewFlushManager(tmpDir, vlog, 16)
	if err != nil {
		t.Fatalf("failed creating engine: %v", err)
	}
	defer engine.Close()

	// Enable background compactor with threshold = 4
	_ = engine.EnableCompaction(4)

	// Flush 4 distinct batches
	for batch := 1; batch <= 4; batch++ {
		for i := 0; i < 100; i++ {
			k := fmt.Sprintf("bg_key_%02d_%04d", batch, i)
			v := fmt.Sprintf("bg_val_%02d_%04d", batch, i)
			_ = engine.Put([]byte(k), []byte(v))
		}
		_, err := engine.RotateAndFlushSync()
		if err != nil {
			t.Fatalf("flush %d failed: %v", batch, err)
		}
	}

	// Wait for background worker to automatically process compaction trigger
	deadline := time.Now().Add(3 * time.Second)
	compacted := false
	for time.Now().Before(deadline) {
		if len(engine.SSTables()) == 1 {
			compacted = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	if !compacted {
		t.Fatalf("background compaction did not consolidate tables within deadline, count=%d", len(engine.SSTables()))
	}

	// Verify all data is retrievable post background compaction
	for batch := 1; batch <= 4; batch++ {
		for i := 0; i < 100; i++ {
			k := fmt.Sprintf("bg_key_%02d_%04d", batch, i)
			expectedV := fmt.Sprintf("bg_val_%02d_%04d", batch, i)
			gotV, err := engine.Get([]byte(k))
			if err != nil {
				t.Fatalf("failed retrieving key %s: %v", k, err)
			}
			if string(gotV) != expectedV {
				t.Fatalf("key %s: expected %s, got %s", k, expectedV, string(gotV))
			}
		}
	}
}
