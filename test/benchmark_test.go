package test

import (
	"crypto/rand"
	"fmt"
	mrand "math/rand"
	"os"
	"testing"
	"time"

	"lsmtree/pkg/lsm"
)

// BenchmarkSequentialWrites measures sequential write throughput in ops/sec and MB/sec.
func BenchmarkSequentialWrites(b *testing.B) {
	dir, err := os.MkdirTemp("", "lsm_bench_seq_writes_*")
	if err != nil {
		b.Fatalf("failed creating temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	db, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 32,
		CompactionThreshold:   10,
		EnableCompaction:      true,
		VLogFileID:            1,
	})
	if err != nil {
		b.Fatalf("OpenDB failed: %v", err)
	}
	defer db.Close()

	valSize := 1024 // 1 KB payload
	valPayload := make([]byte, valSize)
	_, _ = rand.Read(valPayload)

	keySize := 24
	totalBytesPerOp := keySize + valSize
	b.SetBytes(int64(totalBytesPerOp))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		key := []byte(fmt.Sprintf("seq_write_key_%010d", i))
		if err := db.Put(key, valPayload); err != nil {
			b.Fatalf("Put failed at %d: %v", i, err)
		}
	}
}

// BenchmarkRandomReads evaluates point lookup latency and throughput comparing
// hot working set keys (in memory / recent SSTables) vs cold keys (older SSTables on disk).
func BenchmarkRandomReads(b *testing.B) {
	dir, err := os.MkdirTemp("", "lsm_bench_random_reads_*")
	if err != nil {
		b.Fatalf("failed creating temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	db, err := lsm.OpenDB(dir, lsm.Options{
		SparseIndexSampleRate: 32,
		CompactionThreshold:   10,
		EnableCompaction:      false,
		VLogFileID:            1,
	})
	if err != nil {
		b.Fatalf("OpenDB failed: %v", err)
	}
	defer db.Close()

	totalKeys := 20000
	valSize := 512
	valPayload := make([]byte, valSize)
	_, _ = rand.Read(valPayload)

	keys := make([][]byte, totalKeys)
	for i := 0; i < totalKeys; i++ {
		keys[i] = []byte(fmt.Sprintf("bench_read_key_%08d", i))
		if err := db.Put(keys[i], valPayload); err != nil {
			b.Fatalf("Put failed: %v", err)
		}
		// Periodically flush to create multiple SSTables on disk
		if (i+1)%4000 == 0 {
			if _, err := db.Flush(); err != nil {
				b.Fatalf("Flush failed: %v", err)
			}
		}
	}

	rng := mrand.New(mrand.NewSource(time.Now().UnixNano()))

	// Sub-benchmark 1: Hot Keys (80% queries hit latest 5% of dataset in memory)
	b.Run("HotKeys", func(b *testing.B) {
		b.SetBytes(int64(valSize))
		b.ResetTimer()
		b.ReportAllocs()

		hotWindowStart := int(float64(totalKeys) * 0.95)
		hotWindowLen := totalKeys - hotWindowStart

		for i := 0; i < b.N; i++ {
			var k []byte
			if rng.Float64() < 0.80 {
				// 80% chance: pick from hot window
				idx := hotWindowStart + rng.Intn(hotWindowLen)
				k = keys[idx]
			} else {
				// 20% chance: pick from rest of database
				idx := rng.Intn(hotWindowStart)
				k = keys[idx]
			}

			_, err := db.Get(k)
			if err != nil {
				b.Fatalf("Get failed for hot key: %v", err)
			}
		}
	})

	// Sub-benchmark 2: Cold Keys (Uniform random queries across entire dataset)
	b.Run("ColdKeys", func(b *testing.B) {
		b.SetBytes(int64(valSize))
		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			idx := rng.Intn(totalKeys)
			k := keys[idx]

			_, err := db.Get(k)
			if err != nil {
				b.Fatalf("Get failed for cold key: %v", err)
			}
		}
	})
}

// BenchmarkLargeValueWiscKeyAdvantage demonstrates the write amplification benefits
// of WiscKey KV separation across 128B, 4KB, and 64KB value payloads.
// In standard LSMs, large values trigger 10x-30x write amplification as they pass
// through multi-tier compaction. In WiscKey, large values are appended once to VLog
// while the LSM-tree only stores lightweight 16-byte ValuePtr metadata.
func BenchmarkLargeValueWiscKeyAdvantage(b *testing.B) {
	sizes := []struct {
		name string
		size int
	}{
		{"Values_128B", 128},
		{"Values_4KB", 4 * 1024},
		{"Values_64KB", 64 * 1024},
	}

	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			dir, err := os.MkdirTemp("", fmt.Sprintf("lsm_bench_wisckey_%s_*", tc.name))
			if err != nil {
				b.Fatalf("failed creating temp dir: %v", err)
			}
			defer os.RemoveAll(dir)

			db, err := lsm.OpenDB(dir, lsm.Options{
				SparseIndexSampleRate: 32,
				CompactionThreshold:   10,
				EnableCompaction:      true,
				VLogFileID:            1,
			})
			if err != nil {
				b.Fatalf("OpenDB failed: %v", err)
			}
			defer db.Close()

			payload := make([]byte, tc.size)
			_, _ = rand.Read(payload)

			b.SetBytes(int64(tc.size + 24))
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				k := []byte(fmt.Sprintf("wisckey_key_%010d", i))
				if err := db.Put(k, payload); err != nil {
					b.Fatalf("Put failed: %v", err)
				}
			}
		})
	}
}
