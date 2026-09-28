package lsm

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
)

const (
	// DefaultSparseIndexSampleRate stores every Nth key in the sparse index block.
	DefaultSparseIndexSampleRate = 64

	// sstableFooterSize is the fixed 24-byte trailer: [IndexOffset (8B)][IndexLen (8B)][BloomLen (8B)].
	sstableFooterSize = 24

	// sstableDataEntryPrefix is KeyLen (2B) + IsTombstone (1B) + ValuePtr (16B) = 19 bytes.
	sstableDataEntryPrefix = 2 + 1 + ValuePtrSize
)

// IndexEntry represents a sampled key and its corresponding byte offset within the Data Block.
type IndexEntry struct {
	Key    []byte
	Offset uint64
}

// SSTableWriter constructs an immutable SSTable file on disk with sparse indexing.
//
// File Layout:
// [Data Block: sorted entries]
// [Sparse Index Block: sampled keys and offsets]
// [Bloom Filter Block: serialized BloomFilter]
// [Footer: IndexOffset (8B) + IndexLen (8B) + BloomLen (8B)]
type SSTableWriter struct {
	file       *os.File
	writer     *bufio.Writer
	sampleRate int
	count      int
	offset     uint64
	indexes    []IndexEntry
	bloom      *BloomFilter
	keys       [][]byte
	path       string
	closed     bool
}

// NewSSTableWriter creates a new SSTableWriter writing to the specified path.
func NewSSTableWriter(path string, sampleRate int) (*SSTableWriter, error) {
	return NewSSTableWriterWithExpectedKeys(path, sampleRate, 0)
}

// NewSSTableWriterWithExpectedKeys creates a new SSTableWriter with a pre-sized Bloom filter
// allowing keys to be streamed directly into the Bloom filter during insertion.
func NewSSTableWriterWithExpectedKeys(path string, sampleRate int, expectedKeys int) (*SSTableWriter, error) {
	if sampleRate <= 0 {
		sampleRate = DefaultSparseIndexSampleRate
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("lsm: failed to create SSTable directory: %w", err)
	}

	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("lsm: failed to create SSTable file: %w", err)
	}

	var bf *BloomFilter
	if expectedKeys > 0 {
		bf = NewBloomFilter(expectedKeys, 0.01)
	}

	return &SSTableWriter{
		file:       file,
		writer:     bufio.NewWriterSize(file, 256*1024),
		sampleRate: sampleRate,
		bloom:      bf,
		path:       path,
	}, nil
}

// Append writes a key and its ValuePtr or tombstone to the SSTable Data Block.
// Keys MUST be passed in strictly ascending sorted order.
func (w *SSTableWriter) Append(key []byte, isTombstone bool, vp ValuePtr) error {
	if w.closed {
		return os.ErrClosed
	}

	keyLen := len(key)
	if keyLen == 0 {
		return ErrKeyEmpty
	}
	if keyLen > math.MaxUint16 {
		return ErrKeyTooLarge
	}

	// Sample key into Sparse Index Block every sampleRate entries
	if w.count%w.sampleRate == 0 {
		w.indexes = append(w.indexes, IndexEntry{
			Key:    bytes.Clone(key),
			Offset: w.offset,
		})
	}

	// Data Block entry layout:
	// [KeyLen (2B)][Key (KeyLen bytes)][IsTombstone (1B)][FileID (4B)][Offset (8B)][Len (4B)]
	entrySize := sstableDataEntryPrefix + keyLen
	buf := make([]byte, entrySize)

	binary.BigEndian.PutUint16(buf[0:2], uint16(keyLen))
	copy(buf[2:2+keyLen], key)

	if isTombstone {
		buf[2+keyLen] = 0x01
	} else {
		buf[2+keyLen] = 0x00
	}

	vp.EncodeTo(buf[3+keyLen : 19+keyLen])

	if _, err := w.writer.Write(buf); err != nil {
		return fmt.Errorf("lsm: failed to write SSTable entry: %w", err)
	}

	// Stream key into Bloom filter
	if w.bloom != nil {
		w.bloom.Add(key)
	} else {
		w.keys = append(w.keys, bytes.Clone(key))
	}
	w.offset += uint64(entrySize)
	w.count++
	return nil
}

// Finish writes the Sparse Index Block, appends the Bloom Filter, appends the 24-byte Footer, and closes the file.
func (w *SSTableWriter) Finish() error {
	if w.closed {
		return nil
	}
	w.closed = true

	indexOffset := w.offset

	// 1. Write Sparse Index Block:
	// Sequence of [KeyLen (2B)][Key (KeyLen bytes)][Offset (8B)]
	for _, idx := range w.indexes {
		kLen := len(idx.Key)
		idxBuf := make([]byte, 2+kLen+8)
		binary.BigEndian.PutUint16(idxBuf[0:2], uint16(kLen))
		copy(idxBuf[2:2+kLen], idx.Key)
		binary.BigEndian.PutUint64(idxBuf[2+kLen:10+kLen], idx.Offset)

		if _, err := w.writer.Write(idxBuf); err != nil {
			_ = w.file.Close()
			return fmt.Errorf("lsm: failed to write sparse index block: %w", err)
		}
		w.offset += uint64(len(idxBuf))
	}

	indexLen := w.offset - indexOffset

	// 2. Generate and append serialized Bloom Filter Block
	if w.bloom == nil {
		w.bloom = NewBloomFilter(len(w.keys), 0.01)
		for _, k := range w.keys {
			w.bloom.Add(k)
		}
		w.keys = nil
	}
	bloomData := w.bloom.Encode()
	bloomLen := uint64(len(bloomData))

	if bloomLen > 0 {
		if _, err := w.writer.Write(bloomData); err != nil {
			_ = w.file.Close()
			return fmt.Errorf("lsm: failed to write bloom filter block: %w", err)
		}
		w.offset += bloomLen
	}

	// 3. Write Footer (fixed 24 bytes): [IndexOffset (8B)][IndexLen (8B)][BloomLen (8B)]
	footer := make([]byte, sstableFooterSize)
	binary.BigEndian.PutUint64(footer[0:8], indexOffset)
	binary.BigEndian.PutUint64(footer[8:16], indexLen)
	binary.BigEndian.PutUint64(footer[16:24], bloomLen)

	if _, err := w.writer.Write(footer); err != nil {
		_ = w.file.Close()
		return fmt.Errorf("lsm: failed to write SSTable footer: %w", err)
	}
	w.offset += sstableFooterSize

	if err := w.writer.Flush(); err != nil {
		_ = w.file.Close()
		return err
	}

	if err := w.file.Sync(); err != nil {
		_ = w.file.Close()
		return err
	}

	return w.file.Close()
}

// SSTableReader provides fast read-only point lookups and range scans on an SSTable
// using binary search over the in-memory sparse index.
type SSTableReader struct {
	mu            sync.RWMutex
	file          *os.File
	path          string
	id            uint64
	refCount      int32
	deleted       bool
	size          int64
	indexOffset   uint64
	indexLen      uint64
	bloomLen      uint64
	sparseIndex   []IndexEntry
	bloom         *BloomFilter
	bloomDisabled bool
}

// OpenSSTable opens an immutable SSTable file and loads its sparse index block into memory.
func OpenSSTable(path string) (*SSTableReader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("lsm: failed to open SSTable %s: %w", path, err)
	}

	stat, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, err
	}

	size := stat.Size()
	if size < sstableFooterSize {
		_ = file.Close()
		return nil, ErrCorruptedRecord
	}

	// 1. Read fixed 24-byte Footer at file tail
	footer := make([]byte, sstableFooterSize)
	if _, err := file.ReadAt(footer, size-sstableFooterSize); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("lsm: failed to read SSTable footer: %w", err)
	}

	indexOffset := binary.BigEndian.Uint64(footer[0:8])
	indexLen := binary.BigEndian.Uint64(footer[8:16])
	bloomLen := binary.BigEndian.Uint64(footer[16:24])

	if indexOffset+indexLen+bloomLen+sstableFooterSize != uint64(size) {
		_ = file.Close()
		return nil, fmt.Errorf("lsm: invalid SSTable footer metadata: %w", ErrCorruptedRecord)
	}

	// 2. Read Sparse Index Block
	indexBuf := make([]byte, indexLen)
	if _, err := file.ReadAt(indexBuf, int64(indexOffset)); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("lsm: failed to read sparse index block: %w", err)
	}

	// 3. Parse Sparse Index into memory
	var sparseIndex []IndexEntry
	pos := 0
	for pos < len(indexBuf) {
		if pos+2 > len(indexBuf) {
			_ = file.Close()
			return nil, ErrCorruptedRecord
		}
		kLen := int(binary.BigEndian.Uint16(indexBuf[pos : pos+2]))
		pos += 2
		if pos+kLen+8 > len(indexBuf) {
			_ = file.Close()
			return nil, ErrCorruptedRecord
		}
		key := indexBuf[pos : pos+kLen]
		pos += kLen
		offset := binary.BigEndian.Uint64(indexBuf[pos : pos+8])
		pos += 8

		sparseIndex = append(sparseIndex, IndexEntry{
			Key:    bytes.Clone(key),
			Offset: offset,
		})
	}

	// 4. Read and decode Bloom Filter Block
	var bloom *BloomFilter
	if bloomLen > 0 {
		bloomBuf := make([]byte, bloomLen)
		if _, err := file.ReadAt(bloomBuf, int64(indexOffset+indexLen)); err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("lsm: failed reading bloom filter block: %w", err)
		}
		bf, err := DecodeBloomFilter(bloomBuf)
		if err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("lsm: failed decoding bloom filter: %w", err)
		}
		bloom = bf
	}

	return &SSTableReader{
		file:        file,
		path:        path,
		refCount:    1,
		size:        size,
		indexOffset: indexOffset,
		indexLen:    indexLen,
		bloomLen:    bloomLen,
		sparseIndex: sparseIndex,
		bloom:       bloom,
	}, nil
}

// SetBloomFilterEnabled enables or disables the bloom filter check for this reader.
// When disabled, MayContain always returns true to simulate non-filtered SSTable lookups.
func (r *SSTableReader) SetBloomFilterEnabled(enabled bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.bloomDisabled = !enabled
}

// IsBloomFilterEnabled reports whether bloom filter checking is active for this reader.
func (r *SSTableReader) IsBloomFilterEnabled() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return !r.bloomDisabled
}

// MayContain uses the embedded Bloom filter to check whether key may exist.
// Returns false if key definitely does not exist.
func (r *SSTableReader) MayContain(key []byte) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if !r.bloomDisabled && r.bloom != nil {
		return r.bloom.Contains(key)
	}
	return true
}

// BloomFilter returns the reader's Bloom filter instance.
func (r *SSTableReader) BloomFilter() *BloomFilter {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.bloom
}

// Get performs a point lookup for key:
// 1. Binary searches the in-memory sparse index to find the candidate byte offset range.
// 2. Seeks to offset and scans linearly within that small range to find the exact ValuePtr.
// Returns (valuePtr, isTombstone, error).
func (r *SSTableReader) Get(key []byte) (ValuePtr, bool, error) {
	vp, isTombstone, _, err := r.GetWithStats(key)
	return vp, isTombstone, err
}

// GetWithStats performs Get while also returning the number of bytes read from disk.
// If the key is not in the Bloom filter, it returns ErrKeyNotFound with 0 bytes read.
func (r *SSTableReader) GetWithStats(key []byte) (ValuePtr, bool, int, error) {
	if len(key) == 0 {
		return ValuePtr{}, false, 0, ErrKeyEmpty
	}

	// 1. Fast negative lookup: check in-memory Bloom filter first!
	// If false, skip the file immediately with ZERO disk I/O!
	if !r.MayContain(key) {
		return ValuePtr{}, false, 0, ErrKeyNotFound
	}

	return r.GetWithoutBloom(key)
}

// GetWithoutBloom performs a point lookup bypassing the Bloom filter,
// forcing binary search of the sparse index and disk I/O.
func (r *SSTableReader) GetWithoutBloom(key []byte) (ValuePtr, bool, int, error) {
	if len(key) == 0 {
		return ValuePtr{}, false, 0, ErrKeyEmpty
	}

	if len(r.sparseIndex) == 0 {
		return ValuePtr{}, false, 0, ErrKeyNotFound
	}

	// If key is strictly smaller than the very first key sampled in the SSTable, it cannot exist
	if bytes.Compare(key, r.sparseIndex[0].Key) < 0 {
		return ValuePtr{}, false, 0, ErrKeyNotFound
	}

	// Binary search sparse index: find largest idx such that sparseIndex[idx].Key <= key
	idx := sort.Search(len(r.sparseIndex), func(i int) bool {
		return bytes.Compare(r.sparseIndex[i].Key, key) > 0
	}) - 1

	if idx < 0 {
		return ValuePtr{}, false, 0, ErrKeyNotFound
	}

	startOffset := r.sparseIndex[idx].Offset
	var endOffset uint64
	if idx+1 < len(r.sparseIndex) {
		endOffset = r.sparseIndex[idx+1].Offset
	} else {
		endOffset = r.indexOffset // Data Block boundary
	}

	if endOffset <= startOffset {
		return ValuePtr{}, false, 0, ErrKeyNotFound
	}

	bytesToRead := int(endOffset - startOffset)
	dataBuf := make([]byte, bytesToRead)
	if _, err := r.file.ReadAt(dataBuf, int64(startOffset)); err != nil && err != io.EOF {
		return ValuePtr{}, false, bytesToRead, fmt.Errorf("lsm: failed reading data block range: %w", err)
	}

	// Scan linearly within the isolated byte offset range
	pos := 0
	for pos < len(dataBuf) {
		if pos+sstableDataEntryPrefix > len(dataBuf) {
			break
		}
		kLen := int(binary.BigEndian.Uint16(dataBuf[pos : pos+2]))
		entryStart := pos + 2
		entryEnd := entryStart + kLen
		if entryEnd+1+ValuePtrSize > len(dataBuf) {
			break
		}

		entryKey := dataBuf[entryStart:entryEnd]
		isTombstone := dataBuf[entryEnd] == 0x01
		vpBuf := dataBuf[entryEnd+1 : entryEnd+1+ValuePtrSize]
		vp, _ := DecodeValuePtr(vpBuf)

		pos = entryEnd + 1 + ValuePtrSize

		cmp := bytes.Compare(entryKey, key)
		if cmp == 0 {
			if isTombstone {
				return ValuePtr{}, true, bytesToRead, nil
			}
			return vp, false, bytesToRead, nil
		}
		if cmp > 0 {
			// Keys are strictly sorted; candidate key does not exist
			break
		}
	}

	return ValuePtr{}, false, bytesToRead, ErrKeyNotFound
}

// Close closes the underlying SSTable file descriptor.
func (r *SSTableReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.file != nil {
		err := r.file.Close()
		r.file = nil
		return err
	}
	return nil
}

// SparseIndex returns a copy of the in-memory sparse index entries.
func (r *SSTableReader) SparseIndex() []IndexEntry {
	entries := make([]IndexEntry, len(r.sparseIndex))
	copy(entries, r.sparseIndex)
	return entries
}

// Size returns total size in bytes of the SSTable file.
func (r *SSTableReader) Size() int64 {
	return r.size
}

// Path returns the filepath of the SSTable.
func (r *SSTableReader) Path() string {
	return r.path
}

// ID returns the numeric sequence ID of the SSTable.
func (r *SSTableReader) ID() uint64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.id
}

// SetID sets the numeric sequence ID of the SSTable.
func (r *SSTableReader) SetID(id uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.id = id
}

// Retain increments the reference counter for concurrent read safety.
func (r *SSTableReader) Retain() {
	atomic.AddInt32(&r.refCount, 1)
}

// Release decrements the reference counter. If the reference count drops
// to zero and the file was marked for deletion, it closes and deletes the file.
func (r *SSTableReader) Release() error {
	if atomic.AddInt32(&r.refCount, -1) == 0 {
		return r.destroy()
	}
	return nil
}

// MarkDeleted marks the SSTable as obsolete and releases the caller's base reference.
func (r *SSTableReader) MarkDeleted() error {
	r.mu.Lock()
	r.deleted = true
	r.mu.Unlock()
	return r.Release()
}

func (r *SSTableReader) destroy() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var err error
	if r.file != nil {
		err = r.file.Close()
		r.file = nil
	}
	if r.deleted && r.path != "" {
		_ = os.Remove(r.path)
	}
	return err
}

// NewIterator creates a sequential iterator over the entire Data Block of the SSTable.
func (r *SSTableReader) NewIterator() (*SSTableIterator, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.file == nil {
		return nil, os.ErrClosed
	}

	sr := io.NewSectionReader(r.file, 0, int64(r.indexOffset))
	return &SSTableIterator{
		reader: r,
		r:      bufio.NewReaderSize(sr, 64*1024),
		limit:  int64(r.indexOffset),
	}, nil
}

// SSTableIterator streams through all key-value entries in an SSTable Data Block.
type SSTableIterator struct {
	reader   *SSTableReader
	r        *bufio.Reader
	offset   int64
	limit    int64
	currKey  []byte
	currTomb bool
	currVP   ValuePtr
	valid    bool
	err      error
}

// Next advances the iterator to the next entry.
func (it *SSTableIterator) Next() bool {
	if it.offset >= it.limit {
		it.valid = false
		return false
	}

	var kLenBuf [2]byte
	if _, err := io.ReadFull(it.r, kLenBuf[:]); err != nil {
		if err != io.EOF {
			it.err = err
		}
		it.valid = false
		return false
	}
	kLen := int(binary.BigEndian.Uint16(kLenBuf[:]))

	key := make([]byte, kLen)
	if _, err := io.ReadFull(it.r, key); err != nil {
		it.err = err
		it.valid = false
		return false
	}

	var tombBuf [1]byte
	if _, err := io.ReadFull(it.r, tombBuf[:]); err != nil {
		it.err = err
		it.valid = false
		return false
	}
	isTomb := tombBuf[0] == 0x01

	var vpBuf [ValuePtrSize]byte
	if _, err := io.ReadFull(it.r, vpBuf[:]); err != nil {
		it.err = err
		it.valid = false
		return false
	}
	vp, _ := DecodeValuePtr(vpBuf[:])

	it.offset += int64(sstableDataEntryPrefix + kLen)
	it.currKey = key
	it.currTomb = isTomb
	it.currVP = vp
	it.valid = true
	return true
}

// Key returns the current entry's key.
func (it *SSTableIterator) Key() []byte {
	return it.currKey
}

// IsTombstone reports whether the current entry is a deletion tombstone.
func (it *SSTableIterator) IsTombstone() bool {
	return it.currTomb
}

// Value returns the ValuePtr associated with the current entry.
func (it *SSTableIterator) Value() ValuePtr {
	return it.currVP
}

// Valid reports whether the iterator is positioned at a valid entry.
func (it *SSTableIterator) Valid() bool {
	return it.valid
}

// Error returns any fatal error encountered during iteration.
func (it *SSTableIterator) Error() error {
	return it.err
}

// Close closes the iterator.
func (it *SSTableIterator) Close() error {
	it.valid = false
	return nil
}
