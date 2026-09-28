package main

import (
	"fmt"
	"os"
	"path/filepath"

	"lsmtree/pkg/lsm"
)

func main() {
	fmt.Println("==================================================================")
	fmt.Println("🚀 LSM-Tree Storage Engine with WiscKey Key-Value Separation")
	fmt.Println("==================================================================")

	dataDir := "./data_demo"
	_ = os.RemoveAll(dataDir) // Clean start for demo
	defer os.RemoveAll(dataDir)

	// 1. Open Storage Engine
	fmt.Println("\n[Step 1] Initializing engine in:", dataDir)
	opts := lsm.DefaultOptions()
	opts.SparseIndexSampleRate = 16
	opts.CompactionThreshold = 4

	db, err := lsm.OpenDB(dataDir, opts)
	if err != nil {
		fmt.Printf("❌ Failed to open DB: %v\n", err)
		return
	}
	defer db.Close()
	fmt.Println("   ✔ Storage Engine successfully opened!")

	// 2. Insert records (WiscKey KV Separation)
	fmt.Println("\n[Step 2] Writing keys (WiscKey Decoupling: Values -> VLog, Keys -> MemTable + WAL)...")
	for i := 1; i <= 100; i++ {
		key := []byte(fmt.Sprintf("user_account_%04d", i))
		val := []byte(fmt.Sprintf("{\"user_id\": %d, \"name\": \"User %d\", \"plan\": \"Enterprise\"}", i, i))
		if err := db.Put(key, val); err != nil {
			fmt.Printf("❌ Put failed: %v\n", err)
			return
		}
	}
	fmt.Println("   ✔ Wrote 100 records into Active MemTable & VLog.")

	// 3. Read directly from Active MemTable
	fmt.Println("\n[Step 3] Point lookup from Active MemTable (RAM)...")
	key50 := []byte("user_account_0050")
	val50, stats, err := db.GetWithStats(key50)
	if err != nil {
		fmt.Printf("❌ Get failed: %v\n", err)
		return
	}
	fmt.Printf("   ✔ Key: %s\n", string(key50))
	fmt.Printf("   ✔ Value: %s\n", string(val50))
	fmt.Printf("   ✔ Diagnostics: ActiveMemHit=%v, SSTablesExamined=%d, DiskReads=%d\n",
		stats.ActiveMemHit, stats.SSTablesExamined, stats.DiskReads)

	// 4. Flush MemTable to an immutable SSTable on disk
	fmt.Println("\n[Step 4] Freezing Active MemTable and flushing to immutable SSTable on disk...")
	sstReader, err := db.Flush()
	if err != nil {
		fmt.Printf("❌ Flush failed: %v\n", err)
		return
	}
	fmt.Printf("   ✔ Generated SSTable: %s (Size: %d bytes)\n", sstReader.Path(), sstReader.Size())
	fmt.Printf("   ✔ Embedded Bloom Filter: %d bits, %d hash iterations\n",
		sstReader.BloomFilter().M(), sstReader.BloomFilter().K())

	// 5. Read from SSTable through 4-step hierarchy
	fmt.Println("\n[Step 5] Point lookup through Unified 4-Step Read Path...")
	key75 := []byte("user_account_0075")
	val75, stats, err := db.GetWithStats(key75)
	if err != nil {
		fmt.Printf("❌ Get failed: %v\n", err)
		return
	}
	fmt.Printf("   ✔ Key: %s\n", string(key75))
	fmt.Printf("   ✔ Value: %s\n", string(val75))
	fmt.Printf("   ✔ Diagnostics: ActiveMemHit=%v, BloomFilterHits=%d, DiskReads=%d\n",
		stats.ActiveMemHit, stats.BloomFilterHits, stats.DiskReads)

	// 6. Fast Negative Lookup: Bloom Filter Skip (Zero Disk I/O)
	fmt.Println("\n[Step 6] Negative lookup for non-existent key (Bloom Filter Routing)...")
	missingKey := []byte("non_existent_key_9999")
	_, stats, err = db.GetWithStats(missingKey)
	if err == lsm.ErrKeyNotFound {
		fmt.Printf("   ✔ Key '%s' correctly not found!\n", string(missingKey))
		fmt.Printf("   ✔ Diagnostics: BloomFilterSkips=%d, DiskReads=%d (ZERO DISK I/O!)\n",
			stats.BloomFilterSkips, stats.DiskReads)
	}

	// 7. Delete key (Tombstone Masking)
	fmt.Println("\n[Step 7] Deleting key 'user_account_0050' (Tombstone masking)...")
	if err := db.Delete(key50); err != nil {
		fmt.Printf("❌ Delete failed: %v\n", err)
		return
	}
	_, _, err = db.GetWithStats(key50)
	if err == lsm.ErrKeyNotFound {
		fmt.Println("   ✔ Key successfully deleted and masked with tombstone record!")
	}

	// 8. Reopening & Crash Recovery Demo
	fmt.Println("\n[Step 8] Testing Crash Recovery & WAL Replay on restart...")
	// Write new key to WAL without flushing to SSTable
	crashKey := []byte("unflushed_persistent_key")
	crashVal := []byte("this_value_was_persisted_in_wal")
	if err := db.Put(crashKey, crashVal); err != nil {
		fmt.Printf("❌ Put failed: %v\n", err)
		return
	}

	// Close database
	_ = db.Close()

	// Reopen database from same directory
	dbReopened, err := lsm.OpenDB(dataDir, opts)
	if err != nil {
		fmt.Printf("❌ OpenDB failed during recovery: %v\n", err)
		return
	}
	defer dbReopened.Close()

	recoveredVal, err := dbReopened.Get(crashKey)
	if err != nil {
		fmt.Printf("❌ Failed to recover key: %v\n", err)
		return
	}
	fmt.Printf("   ✔ Successfully recovered key '%s' from WAL on startup!\n", string(crashKey))
	fmt.Printf("   ✔ Recovered Value: %s\n", string(recoveredVal))

	// Verify SSTable files on disk
	sstFiles, _ := filepath.Glob(filepath.Join(dataDir, "sstable", "*.sst"))
	walFiles, _ := filepath.Glob(filepath.Join(dataDir, "wal", "*.wal"))
	vlogFiles, _ := filepath.Glob(filepath.Join(dataDir, "vlog", "*.vlog"))

	fmt.Println("\n==================================================================")
	fmt.Println("📊 Engine State on Disk:")
	fmt.Printf("   • SSTable Files: %d (%v)\n", len(sstFiles), sstFiles)
	fmt.Printf("   • WAL Files:     %d (%v)\n", len(walFiles), walFiles)
	fmt.Printf("   • VLog Files:    %d (%v)\n", len(vlogFiles), vlogFiles)
	fmt.Println("==================================================================")
	fmt.Println("🎉 DEMO COMPLETED SUCCESSFULLY!")
	fmt.Println("==================================================================")
}
