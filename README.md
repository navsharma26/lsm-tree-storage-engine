# LSM-Tree Storage Engine with WiscKey Key-Value Separation

A production-grade Log-Structured Merge-tree (LSM-tree) storage engine implementing the **WiscKey** key-value separation architecture in pure Go (standard library only, zero external dependencies).

---

## Architecture Overview

```
                                  WRITE PATH (Put / Delete)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
             [Values decoupled]                               [Keys decoupled]
                      ▼                                               ▼
          Append-Only Value Log (VLog)                     Active MemTable (SkipList)
          [CRC32][KeyLen][ValLen][Key][Val]                           │
                      │                                               ▼
              Returns ValuePtr                         Append-Only Binary WAL
          (FileID, Offset, Len)                        [CRC32][Type][KeyLen][ValuePtr][Key]
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                    (MemTable Threshold Met: 4MB/8MB)
                                              ▼
                                   Immutable MemTable Flush
                                              │
                                              ▼
                                    SSTable File on Disk
                         ┌─────────────────────────────────────────┐
                         │ Data Block: Sorted [Key, ValuePtr]      │
                         ├─────────────────────────────────────────┤
                         │ Sparse Index: Sampled Keys & Offsets    │
                         ├─────────────────────────────────────────┤
                         │ Bloom Filter: Bitset & Optimal k Hashes │
                         ├─────────────────────────────────────────┤
                         │ Footer (24B): [IndexOff][IndexLen][BLen]│
                         └─────────────────────────────────────────┘
                                              │
                                              ▼
                               Size-Tiered Background Compactor
                             (K-Way Merge with container/heap)
```

### Core Components

1. **WiscKey Key-Value Separation ([`pkg/lsm/vlog.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/vlog.go))**:
   - Decouples large values from the LSM-tree. Raw values are appended once to the append-only Value Log (`VLog`), returning a 16-byte [`ValuePtr`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/types.go#L37-L42) (`FileID uint32`, `Offset uint64`, `Len uint32`).
   - The LSM-tree stores only `(Key, ValuePtr)`. This reduces write amplification from typical $10\times\text{--}30\times$ down to nearly $1\times$, saturating storage bandwidth on large payloads.
   - Each VLog entry stores the Key alongside Value (`[CRC32][KeyLen][ValLen][Key][Value]`) for crash consistency and garbage collection.

2. **MemTable & Concurrency-Safe SkipList ([`pkg/lsm/memtable.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/memtable.go), [`pkg/lsm/skiplist.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/skiplist.go))**:
   - Lock-free / concurrency-safe probabilistic SkipList ($p=0.5, \text{maxLevel}=16$).
   - Atomic memory tracking (`sizeBytes`) with flush threshold detection (`DefaultFlushThreshold = 4MB`, `MaxFlushThreshold = 8MB`).
   - Snapshot-isolated forward iterator supporting binary search `Seek`.

3. **Append-Only Write-Ahead Log ([`pkg/lsm/wal.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/wal.go))**:
   - Binary format: `[CRC32 (4B)][RecordType (1B)][KeyLen (2B)][ValuePtr (16B)][Payload (KeyLen bytes)]`.
   - Strict crash recovery: Scans sequentially to EOF, detects truncated/torn writes, and automatically truncates the log to the last valid byte offset.

4. **SSTable with Sparse Index & Embedded Bloom Filter ([`pkg/lsm/sstable.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/sstable.go), [`pkg/lsm/bloom.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/bloom.go))**:
   - Keys are streamed into an embedded Bloom filter during SSTable generation.
   - Sparse index samples keys every $N$ entries (default 64) for bounded memory usage.
   - Fixed 24-byte Footer: `[IndexOffset (8B)][IndexLen (8B)][BloomLen (8B)]`.
   - Thread-safe ref-counting (`Retain`/`Release`/`MarkDeleted`) ensures active readers are never disrupted by background compaction unlinking files.

5. **Optimal Bloom Filters ([`pkg/lsm/bloom.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/bloom.go))**:
   - Pure bitwise operations on `[]byte`.
   - Optimal parameters calculation:
     $$m = - \frac{n \ln(p)}{(\ln 2)^2}, \quad k = \frac{m}{n} \ln 2$$
   - Double hashing technique:
     $$g_i(x) = (h_1(x) + i \cdot h_2(x)) \pmod m$$
     via FNV-1a 64-bit with `sync.Pool` allocation reuse for **0 allocs/op**.

6. **Size-Tiered Compactor with K-Way Merge ([`pkg/lsm/compaction.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/compaction.go), [`pkg/lsm/manifest.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/pkg/lsm/manifest.go))**:
   - Min-heap multi-way merge (`container/heap`) across $K$ SSTables.
   - Deduplication retaining the newest version based on SSTable file ID / timestamp.
   - Tombstone purging: Deletion tombstones are safely eliminated only when compaction covers the oldest historical tier.
   - Atomic state transitions recorded in `MANIFEST.json` via temp-file rename.

---

## Unified Read Path (`Get(key)`)

```
1. Active MemTable (RAM)
   └─► Found? Return value from VLog via ValuePtr.
2. Immutable MemTable (if flushing)
   └─► Found? Return value from VLog via ValuePtr.
3. SSTables on Disk (Newest to Oldest)
   ├─► Check In-Memory Bloom Filter:
   │   └─► Negative (99% probability)? SKIP FILE IMMEDIATELY (Zero Disk I/O).
   └─► Bloom Filter Hit?
       ├─► Binary Search Sparse Index (Find byte offset range)
       ├─► Seek SSTable & Scan Range
       └─► Found? Fetch value payload from VLog via io.ReadAt.
4. Return ErrKeyNotFound if missing across all tiers.
```

---

## Durability & Fault-Tolerance

A dedicated fault-injection test harness ([`test/fault_fs.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/test/fault_fs.go), [`test/crash_recovery_test.go`](file:///Users/navneetsharma/LSM-Tree%20Storage%20Engine%20/test/crash_recovery_test.go)) validates engine behavior under sudden kernel panics, power cuts, and torn disk writes:

- **Torn Write Halting**: If power cuts mid-record, `wal.Recover()` detects invalid CRC32 or unexpected EOF, cleanly halts, and truncates the log to the last valid byte offset.
- **Zero Data Corruption**: 100% of acknowledged transactions preceding the crash are restored and verified intact.
- **Unflushed Buffer Protection**: Both WAL and VLog ensure records reach the OS buffer cache synchronously, maintaining durability across process restarts.

---

## Benchmark Results

*Hardware: Apple M2 (8 cores, arm64), macOS.*

### 1. Sequential Write Throughput

| Benchmark | Operations | Latency | Bandwidth | Memory / Op | Allocations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BenchmarkSequentialWrites` | 121,174 ops | **8,310 ns/op** | **126.12 MB/s** | 1,480 B/op | 8 allocs/op |

### 2. Random Read Performance (Hot vs Cold)

| Workload | Operations | Latency | Bandwidth | Allocations |
| :--- | :--- | :--- | :--- | :--- |
| `RandomReads/HotKeys` (80% in 5% RAM working set) | 380,931 ops | **3,054 ns/op** | **167.63 MB/s** | 3 allocs/op |
| `RandomReads/ColdKeys` (Uniform across older SSTables) | 355,448 ops | **3,320 ns/op** | **154.20 MB/s** | 3 allocs/op |

### 3. Bloom Filter Negative Lookup Routing

| Lookup Configuration | Operations | Latency | Disk I/O Reads | Allocations |
| :--- | :--- | :--- | :--- | :--- |
| **`GetNonExistentKeys_WithBloom`** | **1,636,159 ops** | **728.6 ns/op** | **0 (Zero Disk I/O)** | **0 allocs/op** |
| `GetNonExistentKeys_WithoutBloom` | 224,079 ops | 4,546.0 ns/op | 5 reads/op | 5 allocs/op |
| **Speedup** | **$6.24\times$ faster** | | **$100\%$ I/O reduction** | |

### 4. WiscKey Value Scaling & Write Amplification Advantage

| Value Size | Operations | Latency | Write Bandwidth | Memory / Op |
| :--- | :--- | :--- | :--- | :--- |
| **128 B** | 185,134 ops | 5,889 ns/op | 25.81 MB/s | 488 B/op |
| **4 KB** | 90,079 ops | 13,311 ns/op | **309.52 MB/s** | 5,194 B/op |
| **64 KB** | 22,282 ops | 69,660 ns/op | **941.14 MB/s** | 74,095 B/op |

*Observation: As value size scales from 128B to 64KB, throughput scales to **941 MB/s** because large values bypass LSM compaction amplification.*

---

## Running Tests & Benchmarks

### Run Full Test Suite with Race Detector
```bash
go test -v -race ./...
```

### Run Crash Recovery & Fault-Tolerance Tests
```bash
go test -v -race -run=TestCrashRecovery ./test/...
```

### Run Benchmarks
```bash
go test -bench=. -benchmem -run=^$ ./test/...
```
