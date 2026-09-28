package test

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"lsmtree/pkg/lsm"
)

// TestBloomFilterOptimalMath validates the mathematical formulas for m and k.
func TestBloomFilterOptimalMath(t *testing.T) {
	n := 10000
	p := 0.01 // 1% false positive rate

	m, k := lsm.CalculateOptimalParams(n, p)

	// Theoretical:
	// m = - 10000 * ln(0.01) / (ln(2)^2) ≈ 95851 bits (~11982 bytes)
	// k = (m / n) * ln(2) ≈ 9.585 * 0.69315 ≈ 6.64 => round to 7
	if m < 95000 || m > 97000 {
		t.Fatalf("unexpected m bits: %d, expected ~95851", m)
	}
	if k != 7 {
		t.Fatalf("unexpected k hash count: %d, expected 7", k)
	}

	bf := lsm.NewBloomFilter(n, p)
	if bf.M() != m {
		t.Fatalf("bf.M mismatch: got %d, expected %d", bf.M(), m)
	}
	if bf.K() != k {
		t.Fatalf("bf.K mismatch: got %d, expected %d", bf.K(), k)
	}

	// Test serialization round-trip
	for i := 0; i < 1000; i++ {
		key := []byte(fmt.Sprintf("key_%06d", i))
		bf.Add(key)
	}

	encoded := bf.Encode()
	if len(encoded) != bf.SizeInBytes() {
		t.Fatalf("encoded length %d != SizeInBytes %d", len(encoded), bf.SizeInBytes())
	}

	decoded, err := lsm.DecodeBloomFilter(encoded)
	if err != nil {
		t.Fatalf("DecodeBloomFilter failed: %v", err)
	}
	if decoded.M() != bf.M() || decoded.K() != bf.K() {
		t.Fatalf("decoded parameters mismatch: m=%d/%d, k=%d/%d", decoded.M(), bf.M(), decoded.K(), bf.K())
	}

	// Verify all 1000 keys exist in decoded filter
	for i := 0; i < 1000; i++ {
		key := []byte(fmt.Sprintf("key_%06d", i))
		if !decoded.Contains(key) {
			t.Fatalf("decoded filter missing key %s", key)
		}
	}
}

// TestBloomFilterEmpiricalFalsePositiveRate verifies zero false negatives
// and confirms empirical false positive rate conforms to target p <= 1.5% for p=0.01.
func TestBloomFilterEmpiricalFalsePositiveRate(t *testing.T) {
	n := 20000
	p := 0.01 // 1% target FPR
	bf := lsm.NewBloomFilter(n, p)

	// 1. Insert 20,000 distinct keys
	insertedKeys := make([][]byte, n)
	for i := 0; i < n; i++ {
		key := []byte(fmt.Sprintf("inserted_user_id_%08d", i))
		insertedKeys[i] = key
		bf.Add(key)
	}

	// 2. Zero False Negatives: Every inserted key MUST return true
	for i, key := range insertedKeys {
		if !bf.Contains(key) {
			t.Fatalf("false negative detected at index %d for key %s", i, string(key))
		}
	}

	// 3. Test 20,000 non-existent keys to measure empirical false positive rate
	numQueries := 20000
	falsePositives := 0
	for i := 0; i < numQueries; i++ {
		nonExistentKey := []byte(fmt.Sprintf("non_existent_id_%08d", i))
		if bf.Contains(nonExistentKey) {
			falsePositives++
		}
	}

	empiricalFPR := float64(falsePositives) / float64(numQueries)
	t.Logf("Empirical False Positive Rate: %.4f (%d / %d), Target: %.4f", empiricalFPR, falsePositives, numQueries, p)

	// Target is 1%, allow margin up to 1.5% due to random distribution
	if empiricalFPR > 0.015 {
		t.Fatalf("empirical FPR %.4f exceeded allowed threshold of 0.015", empiricalFPR)
	}
}

// TestSSTableBloomFilterEmbeddedAndNegativeSkips verifies that SSTables embed
// the Bloom filter, the 24-byte footer is correctly formatted, and negative lookups
// skip disk I/O completely.
func TestSSTableBloomFilterEmbeddedAndNegativeSkips(t *testing.T) {
	dir := t.TempDir()
	sstPath := filepath.Join(dir, "000001.sst")

	count := 2000
	writer, err := lsm.NewSSTableWriterWithExpectedKeys(sstPath, 32, count)
	if err != nil {
		t.Fatalf("failed creating SSTableWriter: %v", err)
	}

	for i := 0; i < count; i++ {
		key := []byte(fmt.Sprintf("entity_key_%06d", i))
		vp := lsm.ValuePtr{FileID: 1, Offset: uint64(i * 100), Len: 50}
		if err := writer.Append(key, false, vp); err != nil {
			t.Fatalf("writer.Append failed: %v", err)
		}
	}

	if err := writer.Finish(); err != nil {
		t.Fatalf("writer.Finish failed: %v", err)
	}

	// Verify file size and footer
	stat, err := os.Stat(sstPath)
	if err != nil {
		t.Fatalf("stat failed: %v", err)
	}
	if stat.Size() < 24 {
		t.Fatalf("SSTable size %d smaller than 24-byte footer", stat.Size())
	}

	reader, err := lsm.OpenSSTable(sstPath)
	if err != nil {
		t.Fatalf("OpenSSTable failed: %v", err)
	}
	defer reader.Close()

	if reader.BloomFilter() == nil {
		t.Fatalf("expected embedded Bloom filter to be loaded, got nil")
	}

	// 1. Verify positive lookups
	for i := 0; i < 100; i++ {
		key := []byte(fmt.Sprintf("entity_key_%06d", i*20))
		if !reader.MayContain(key) {
			t.Fatalf("MayContain false negative for existing key %s", key)
		}
		vp, isTomb, bytesRead, err := reader.GetWithStats(key)
		if err != nil {
			t.Fatalf("GetWithStats failed for key %s: %v", key, err)
		}
		if isTomb {
			t.Fatalf("unexpected tombstone for key %s", key)
		}
		if vp.Offset != uint64(i*20*100) {
			t.Fatalf("unexpected ValuePtr offset: got %d, expected %d", vp.Offset, i*20*100)
		}
		if bytesRead <= 0 {
			t.Fatalf("expected disk read > 0 for existing key, got %d", bytesRead)
		}
	}

	// 2. Verify negative lookups with Bloom Filter: Zero Disk I/O
	skippedDiskCount := 0
	numNegative := 1000
	for i := 0; i < numNegative; i++ {
		missingKey := []byte(fmt.Sprintf("missing_key_%08d", i))
		_, _, bytesRead, err := reader.GetWithStats(missingKey)
		if err != lsm.ErrKeyNotFound {
			t.Fatalf("expected ErrKeyNotFound, got %v", err)
		}
		if bytesRead == 0 {
			skippedDiskCount++
		}
	}

	t.Logf("Bloom Filter negative skips with zero disk I/O: %d / %d (%.2f%%)",
		skippedDiskCount, numNegative, float64(skippedDiskCount)/float64(numNegative)*100)

	if skippedDiskCount < 980 {
		t.Fatalf("expected >= 98%% Bloom filter skips with zero disk I/O, got %d / %d", skippedDiskCount, numNegative)
	}

	// 3. Disable Bloom filter: Verify disk I/O is forced
	reader.SetBloomFilterEnabled(false)
	forcedDiskCount := 0
	for i := 0; i < 50; i++ {
		missingKey := []byte(fmt.Sprintf("missing_key_force_disk_%08d", i))
		_, _, bytesRead, err := reader.GetWithStats(missingKey)
		if err != lsm.ErrKeyNotFound {
			t.Fatalf("expected ErrKeyNotFound, got %v", err)
		}
		// Since key falls within range of sparse index, disk block must be read
		if bytesRead > 0 {
			forcedDiskCount++
		}
	}
	t.Logf("Without Bloom filter: forced disk reads: %d / 50", forcedDiskCount)
	if forcedDiskCount == 0 {
		t.Fatalf("expected forced disk reads when Bloom filter is disabled")
	}
}

// TestDBUnifiedReadPath_4StepHierarchy verifies the 4-step read routing:
// 1. Active MemTable
// 2. Immutable MemTable
// 3. SSTables (skipping non-matching SSTables with zero disk I/O via Bloom filters)
// 4. Sparse index binary search -> SSTable seek -> VLog value fetch
func TestDBUnifiedReadPath_4StepHierarchy(t *testing.T) {
	dir := t.TempDir()

	db, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 16,
		CompactionThreshold:   10, // Prevent auto-compaction during test
		EnableCompaction:      false,
		VLogFileID:            1,
	})
	if err != nil {
		t.Fatalf("OpenDB failed: %v", err)
	}
	defer db.Close()

	// Tier 1: Write SSTable 1 (Keys 000 - 099)
	for i := 0; i < 100; i++ {
		k := []byte(fmt.Sprintf("key_tier1_%04d", i))
		v := []byte(fmt.Sprintf("val_tier1_%04d", i))
		if err := db.Put(k, v); err != nil {
			t.Fatalf("Put failed: %v", err)
		}
	}
	if _, err := db.Flush(); err != nil {
		t.Fatalf("Flush 1 failed: %v", err)
	}

	// Tier 2: Write SSTable 2 (Keys 100 - 199)
	for i := 100; i < 200; i++ {
		k := []byte(fmt.Sprintf("key_tier2_%04d", i))
		v := []byte(fmt.Sprintf("val_tier2_%04d", i))
		if err := db.Put(k, v); err != nil {
			t.Fatalf("Put failed: %v", err)
		}
	}
	if _, err := db.Flush(); err != nil {
		t.Fatalf("Flush 2 failed: %v", err)
	}

	// Tier 3: Write to Active MemTable (Keys 200 - 299)
	for i := 200; i < 300; i++ {
		k := []byte(fmt.Sprintf("key_tier3_%04d", i))
		v := []byte(fmt.Sprintf("val_tier3_%04d", i))
		if err := db.Put(k, v); err != nil {
			t.Fatalf("Put failed: %v", err)
		}
	}

	// --- Step 1: Active MemTable Point Lookup ---
	activeKey := []byte("key_tier3_0250")
	val, stats, err := db.GetWithStats(activeKey)
	if err != nil {
		t.Fatalf("Get activeKey failed: %v", err)
	}
	if !stats.ActiveMemHit {
		t.Fatalf("expected ActiveMemHit == true for %s", activeKey)
	}
	if stats.SSTablesExamined != 0 || stats.DiskReads != 0 {
		t.Fatalf("active MemTable hit should not touch SSTables or disk, got stats: %+v", stats)
	}
	if !bytes.Equal(val, []byte("val_tier3_0250")) {
		t.Fatalf("unexpected value: got %s, expected val_tier3_0250", string(val))
	}

	// --- Step 2 & 3: SSTable Lookups with Bloom Filter Routing ---
	// Query SSTable 2 key: Should skip SSTable 1 (older) or hit SSTable 2 directly
	sst2Key := []byte("key_tier2_0150")
	val, stats, err = db.GetWithStats(sst2Key)
	if err != nil {
		t.Fatalf("Get sst2Key failed: %v", err)
	}
	if stats.ActiveMemHit || stats.ImmMemHit {
		t.Fatalf("expected SSTable hit, not memtable hit: %+v", stats)
	}
	if stats.BloomFilterHits < 1 {
		t.Fatalf("expected Bloom filter hit for %s, stats: %+v", sst2Key, stats)
	}
	if !bytes.Equal(val, []byte("val_tier2_0150")) {
		t.Fatalf("unexpected value: got %s, expected val_tier2_0150", string(val))
	}

	// Query SSTable 1 key: Newer SSTable 2 must be SKIPPED via Bloom filter!
	sst1Key := []byte("key_tier1_0050")
	val, stats, err = db.GetWithStats(sst1Key)
	if err != nil {
		t.Fatalf("Get sst1Key failed: %v", err)
	}
	if stats.BloomFilterSkips < 1 {
		t.Fatalf("expected SSTable 2 to be skipped by Bloom filter for key %s, stats: %+v", sst1Key, stats)
	}
	if !bytes.Equal(val, []byte("val_tier1_0050")) {
		t.Fatalf("unexpected value: got %s, expected val_tier1_0050", string(val))
	}

	// --- Step 4: Non-Existent Key Lookup: All SSTables Skipped with ZERO Disk I/O ---
	missingKey := []byte("key_non_existent_9999")
	_, stats, err = db.GetWithStats(missingKey)
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected ErrKeyNotFound for %s, got: %v", missingKey, err)
	}
	if stats.ActiveMemHit || stats.ImmMemHit {
		t.Fatalf("unexpected mem hit for non-existent key: %+v", stats)
	}
	// Both SSTables should be skipped immediately with zero disk reads
	if stats.BloomFilterSkips != 2 {
		t.Fatalf("expected 2 BloomFilterSkips, got %d (stats: %+v)", stats.BloomFilterSkips, stats)
	}
	if stats.DiskReads != 0 {
		t.Fatalf("expected ZERO disk reads for non-existent key, got %d", stats.DiskReads)
	}

	// --- Step 5: Deletion Tombstone in Active MemTable ---
	if err := db.Delete([]byte("key_tier1_0050")); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	_, stats, err = db.GetWithStats([]byte("key_tier1_0050"))
	if err != lsm.ErrKeyNotFound {
		t.Fatalf("expected ErrKeyNotFound for deleted key, got: %v", err)
	}
	if !stats.ActiveMemHit {
		t.Fatalf("expected ActiveMemHit for tombstone: %+v", stats)
	}
	if stats.SSTablesExamined != 0 {
		t.Fatalf("tombstone in active MemTable must mask older SSTables without examining them: %+v", stats)
	}
}

// TestDBZeroDiskIOOnNegativeLookups writes multiple SSTables and queries 5,000 non-existent
// keys to verify that > 98% of lookups result in ZERO disk I/O.
func TestDBZeroDiskIOOnNegativeLookups(t *testing.T) {
	dir := t.TempDir()

	db, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 32,
		CompactionThreshold:   10,
		EnableCompaction:      false,
		VLogFileID:            1,
	})
	if err != nil {
		t.Fatalf("OpenDB failed: %v", err)
	}
	defer db.Close()

	// Write 5 SSTables, each with 1,000 keys (5,000 keys total)
	totalTables := 5
	keysPerTable := 1000
	for table := 0; table < totalTables; table++ {
		for i := 0; i < keysPerTable; i++ {
			idx := table*keysPerTable + i
			k := []byte(fmt.Sprintf("account_user_id_%08d", idx))
			v := []byte(fmt.Sprintf("payload_data_%08d", idx))
			if err := db.Put(k, v); err != nil {
				t.Fatalf("Put failed: %v", err)
			}
		}
		if _, err := db.Flush(); err != nil {
			t.Fatalf("Flush %d failed: %v", table, err)
		}
	}

	if len(db.SSTables()) != totalTables {
		t.Fatalf("expected %d SSTables, got %d", totalTables, len(db.SSTables()))
	}

	// Query 1,000 random non-existent keys
	queryCount := 1000
	totalSSTableChecks := queryCount * totalTables
	totalSkips := 0
	totalDiskReads := 0

	for i := 0; i < queryCount; i++ {
		missingKey := []byte(fmt.Sprintf("unknown_query_key_%08d", i))
		_, stats, err := db.GetWithStats(missingKey)
		if err != lsm.ErrKeyNotFound {
			t.Fatalf("expected ErrKeyNotFound, got %v", err)
		}
		totalSkips += stats.BloomFilterSkips
		totalDiskReads += stats.DiskReads
	}

	skipPercentage := float64(totalSkips) / float64(totalSSTableChecks) * 100
	t.Logf("Total SSTable Checks: %d, Total Bloom Skips: %d (%.2f%%), Total Disk Reads: %d",
		totalSSTableChecks, totalSkips, skipPercentage, totalDiskReads)

	if skipPercentage < 98.0 {
		t.Fatalf("expected >= 98%% Bloom filter skips, got %.2f%%", skipPercentage)
	}
	if totalDiskReads > 100 { // < 2.0% false positive disk reads out of 5000 checks
		t.Fatalf("too many disk reads for negative lookups: %d", totalDiskReads)
	}
}

// -----------------------------------------------------------------------------
// BENCHMARKS: Read Latency for Non-Existent Keys WITH vs WITHOUT Bloom Filter
// -----------------------------------------------------------------------------

var benchDB *lsm.DB
var benchNonExistentKeys [][]byte

func initBenchmarkDB(b *testing.B) (*lsm.DB, [][]byte) {
	b.Helper()
	dir, err := os.MkdirTemp("", "lsm_bench_phase4_*")
	if err != nil {
		b.Fatalf("failed creating temp dir: %v", err)
	}

	db, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 32,
		CompactionThreshold:   20,
		EnableCompaction:      false,
		VLogFileID:            1,
	})
	if err != nil {
		b.Fatalf("OpenDB failed: %v", err)
	}

	// Write 5 SSTables, each with 2,000 keys (10,000 total keys on disk)
	numTables := 5
	keysPerTable := 2000
	for t := 0; t < numTables; t++ {
		for i := 0; i < keysPerTable; i++ {
			idx := t*keysPerTable + i
			k := []byte(fmt.Sprintf("bench_key_id_%08d", idx))
			v := []byte(fmt.Sprintf("bench_value_payload_%08d", idx))
			if err := db.Put(k, v); err != nil {
				b.Fatalf("Put failed: %v", err)
			}
		}
		if _, err := db.Flush(); err != nil {
			b.Fatalf("Flush failed: %v", err)
		}
	}

	// Pre-generate 10,000 non-existent keys
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	nonExistentKeys := make([][]byte, 10000)
	for i := 0; i < len(nonExistentKeys); i++ {
		nonExistentKeys[i] = []byte(fmt.Sprintf("non_existent_random_%012d", rng.Int63()))
	}

	return db, nonExistentKeys
}

// BenchmarkGetNonExistentKeys_WithBloom benchmarks point lookups for non-existent
// keys when Bloom filter checks are ACTIVE.
// All SSTables are skipped in-memory with ZERO disk I/O.
func BenchmarkGetNonExistentKeys_WithBloom(b *testing.B) {
	db, keys := initBenchmarkDB(b)
	defer func() {
		_ = db.Close()
		_ = os.RemoveAll(db.Dir())
	}()

	keyCount := len(keys)
	db.SetBloomFilterEnabled(true)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		k := keys[i%keyCount]
		_, err := db.Get(k)
		if err != lsm.ErrKeyNotFound {
			b.Fatalf("expected ErrKeyNotFound, got %v", err)
		}
	}
}

// BenchmarkGetNonExistentKeys_WithoutBloom benchmarks point lookups for non-existent
// keys when Bloom filter checks are BYPASSED.
// Every SSTable requires sparse index binary search and disk ReadAt I/O.
func BenchmarkGetNonExistentKeys_WithoutBloom(b *testing.B) {
	db, keys := initBenchmarkDB(b)
	defer func() {
		_ = db.Close()
		_ = os.RemoveAll(db.Dir())
	}()

	keyCount := len(keys)
	db.SetBloomFilterEnabled(false)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		k := keys[i%keyCount]
		_, err := db.GetWithoutBloom(k)
		if err != lsm.ErrKeyNotFound {
			b.Fatalf("expected ErrKeyNotFound, got %v", err)
		}
	}
}

// BenchmarkBloomFilter_Add measures insertion throughput of the pure bitwise Bloom filter.
func BenchmarkBloomFilter_Add(b *testing.B) {
	bf := lsm.NewBloomFilter(b.N, 0.01)
	keys := make([][]byte, 1000)
	for i := 0; i < len(keys); i++ {
		keys[i] = []byte(fmt.Sprintf("perf_key_%08d", i))
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		bf.Add(keys[i%1000])
	}
}

// BenchmarkBloomFilter_Contains_Hit measures point lookup latency on existing keys.
func BenchmarkBloomFilter_Contains_Hit(b *testing.B) {
	n := 10000
	bf := lsm.NewBloomFilter(n, 0.01)
	keys := make([][]byte, n)
	for i := 0; i < n; i++ {
		k := []byte(fmt.Sprintf("bench_hit_key_%08d", i))
		keys[i] = k
		bf.Add(k)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if !bf.Contains(keys[i%n]) {
			b.Fatalf("unexpected false negative")
		}
	}
}

// BenchmarkBloomFilter_Contains_Miss measures point lookup latency on non-existent keys.
func BenchmarkBloomFilter_Contains_Miss(b *testing.B) {
	n := 10000
	bf := lsm.NewBloomFilter(n, 0.01)
	for i := 0; i < n; i++ {
		bf.Add([]byte(fmt.Sprintf("bench_exist_key_%08d", i)))
	}

	missingKeys := make([][]byte, n)
	for i := 0; i < n; i++ {
		missingKeys[i] = []byte(fmt.Sprintf("bench_missing_key_%08d", i))
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = bf.Contains(missingKeys[i%n])
	}
}
