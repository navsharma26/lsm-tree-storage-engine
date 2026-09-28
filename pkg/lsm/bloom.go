package lsm

import (
	"encoding/binary"
	"errors"
	"hash"
	"hash/fnv"
	"math"
	"sync"
)

// BloomFilter implements an in-memory space-efficient probabilistic set
// using pure bitwise operations on []byte and double hashing.
type BloomFilter struct {
	m      uint32 // Bit array size
	k      uint8  // Number of hash iterations
	bitset []byte // Raw bit array storage
}

var fnvPool = sync.Pool{
	New: func() any {
		return fnv.New64a()
	},
}

// CalculateOptimalParams computes the optimal bit size m and hash count k
// given expected elements n and target false positive rate p.
//
// Formulas:
// m = - (n * ln(p)) / (ln(2)^2)
// k = (m / n) * ln(2)
func CalculateOptimalParams(n int, p float64) (m uint32, k uint8) {
	if n <= 0 {
		n = 1
	}
	if p <= 0 || p >= 1.0 {
		p = 0.01 // default 1% false positive rate
	}

	ln2 := math.Ln2
	ln2Sq := ln2 * ln2

	// m = - (n * ln(p)) / (ln(2)^2)
	optimalBits := -float64(n) * math.Log(p) / ln2Sq
	if optimalBits < 64 {
		optimalBits = 64
	}
	if optimalBits > float64(math.MaxUint32) {
		optimalBits = float64(math.MaxUint32)
	}
	m = uint32(optimalBits)

	// k = (m / n) * ln(2)
	optimalK := (float64(m) / float64(n)) * ln2
	kRound := uint8(math.Round(optimalK))
	if kRound < 1 {
		kRound = 1
	}
	if kRound > 30 {
		kRound = 30
	}
	k = kRound

	return m, k
}

// NewBloomFilter constructs an empty BloomFilter tailored for n elements
// with false positive probability p (e.g. 0.01 for 1%).
func NewBloomFilter(n int, p float64) *BloomFilter {
	m, k := CalculateOptimalParams(n, p)
	byteCount := (m + 7) / 8
	return &BloomFilter{
		m:      m,
		k:      k,
		bitset: make([]byte, byteCount),
	}
}

// Add sets the k bits for key using double hashing:
// g_i(x) = (h1(x) + i * h2(x)) mod m
func (bf *BloomFilter) Add(key []byte) {
	if bf == nil || bf.m == 0 {
		return
	}

	h1, h2 := bf.getHashes(key)
	m := uint64(bf.m)

	for i := uint8(0); i < bf.k; i++ {
		bitIndex := (uint64(h1) + uint64(i)*uint64(h2)) % m
		bytePos := bitIndex / 8
		bitOffset := bitIndex % 8
		bf.bitset[bytePos] |= (1 << bitOffset)
	}
}

// Contains checks if key might be present in the set.
// If it returns false, the key is GUARANTEED not present (zero false negatives).
// If it returns true, the key might be present (with false positive rate <= p).
func (bf *BloomFilter) Contains(key []byte) bool {
	if bf == nil || bf.m == 0 || len(bf.bitset) == 0 {
		return true // fail-safe: if no filter, must search storage
	}

	h1, h2 := bf.getHashes(key)
	m := uint64(bf.m)

	for i := uint8(0); i < bf.k; i++ {
		bitIndex := (uint64(h1) + uint64(i)*uint64(h2)) % m
		bytePos := bitIndex / 8
		bitOffset := bitIndex % 8
		if (bf.bitset[bytePos] & (1 << bitOffset)) == 0 {
			return false
		}
	}
	return true
}

// Encode serializes the Bloom filter into binary:
// [k (1B)][m (4B)][bitset bytes]
func (bf *BloomFilter) Encode() []byte {
	if bf == nil {
		return nil
	}
	buf := make([]byte, 1+4+len(bf.bitset))
	buf[0] = bf.k
	binary.BigEndian.PutUint32(buf[1:5], bf.m)
	copy(buf[5:], bf.bitset)
	return buf
}

// DecodeBloomFilter deserializes a BloomFilter from its binary representation.
func DecodeBloomFilter(data []byte) (*BloomFilter, error) {
	if len(data) < 5 {
		return nil, errors.New("lsm: bloom filter buffer too short")
	}

	k := data[0]
	m := binary.BigEndian.Uint32(data[1:5])
	expectedBytes := int((m + 7) / 8)
	if len(data[5:]) < expectedBytes {
		return nil, ErrCorruptedRecord
	}

	bitset := make([]byte, expectedBytes)
	copy(bitset, data[5:5+expectedBytes])

	return &BloomFilter{
		m:      m,
		k:      k,
		bitset: bitset,
	}, nil
}

// SizeInBytes returns the total memory/disk footprint of the BloomFilter.
func (bf *BloomFilter) SizeInBytes() int {
	if bf == nil {
		return 0
	}
	return 5 + len(bf.bitset)
}

// M returns the total bit array size.
func (bf *BloomFilter) M() uint32 {
	if bf == nil {
		return 0
	}
	return bf.m
}

// K returns the number of hash iterations.
func (bf *BloomFilter) K() uint8 {
	if bf == nil {
		return 0
	}
	return bf.k
}

// FalsePositiveRate computes the theoretical false positive rate for n inserted keys:
// p = (1 - e^(-k * n / m))^k
func (bf *BloomFilter) FalsePositiveRate(n int) float64 {
	if bf == nil || bf.m == 0 || n <= 0 {
		return 1.0
	}
	exponent := -float64(bf.k) * float64(n) / float64(bf.m)
	return math.Pow(1.0-math.Exp(exponent), float64(bf.k))
}

// getHashes generates two independent 32-bit hash seeds from key using FNV-1a 64-bit.
func (bf *BloomFilter) getHashes(key []byte) (uint32, uint32) {
	h := fnvPool.Get().(hash.Hash64)
	h.Reset()
	h.Write(key)
	hash64 := h.Sum64()
	fnvPool.Put(h)

	h1 := uint32(hash64)
	h2 := uint32(hash64 >> 32)
	if h2 == 0 {
		h2 = 0x9e3779b9 // golden ratio constant to ensure non-zero step
	}
	return h1, h2
}
