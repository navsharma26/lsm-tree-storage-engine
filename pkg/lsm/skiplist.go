package lsm

import (
	"bytes"
	"math/rand"
	"sort"
	"sync"
	"time"
)

const (
	// DefaultMaxLevel is the maximum height of the SkipList.
	DefaultMaxLevel = 16

	// DefaultProbability is the probability p=0.5 for level promotion.
	DefaultProbability = 0.5
)

// Node represents a single element in the SkipList.
type Node struct {
	key       []byte
	valuePtr  ValuePtr
	isDeleted bool
	forward   []*Node
}

// Key returns a copy of the node's key.
func (n *Node) Key() []byte {
	return bytes.Clone(n.key)
}

// Value returns the ValuePtr associated with the node.
func (n *Node) Value() ValuePtr {
	return n.valuePtr
}

// IsDeleted reports whether the node represents a tombstone.
func (n *Node) IsDeleted() bool {
	return n.isDeleted
}

// SkipList is a thread-safe probabilistic data structure supporting concurrent
// reads, writes, and in-order iteration.
type SkipList struct {
	mu          sync.RWMutex
	head        *Node
	maxLevel    int
	level       int
	length      int
	probability float64
	rnd         *rand.Rand
	rndMu       sync.Mutex
}

// NewSkipList constructs a new SkipList with p=0.5 and maxLevel=16.
func NewSkipList() *SkipList {
	return NewSkipListWithParams(DefaultMaxLevel, DefaultProbability)
}

// NewSkipListWithParams constructs a SkipList with custom maxLevel and probability.
func NewSkipListWithParams(maxLevel int, probability float64) *SkipList {
	if maxLevel <= 0 {
		maxLevel = DefaultMaxLevel
	}
	if probability <= 0 || probability >= 1.0 {
		probability = DefaultProbability
	}

	head := &Node{
		forward: make([]*Node, maxLevel),
	}

	return &SkipList{
		head:        head,
		maxLevel:    maxLevel,
		level:       1,
		length:      0,
		probability: probability,
		rnd:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// randomLevel generates a random level for a new node with geometric distribution (p=0.5).
func (sl *SkipList) randomLevel() int {
	sl.rndMu.Lock()
	defer sl.rndMu.Unlock()

	lvl := 1
	for lvl < sl.maxLevel && sl.rnd.Float64() < sl.probability {
		lvl++
	}
	return lvl
}

// Insert adds or updates an entry in the SkipList.
// If the key already exists, its ValuePtr and isDeleted flag are updated in-place.
// Returns whether the insertion was an update of an existing key.
func (sl *SkipList) Insert(key []byte, valuePtr ValuePtr, isDeleted bool) bool {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	update := make([]*Node, sl.maxLevel)
	curr := sl.head

	// Traverse from top level down to level 0
	for i := sl.level - 1; i >= 0; i-- {
		for curr.forward[i] != nil && bytes.Compare(curr.forward[i].key, key) < 0 {
			curr = curr.forward[i]
		}
		update[i] = curr
	}

	// Examine candidate at level 0
	target := curr.forward[0]
	if target != nil && bytes.Equal(target.key, key) {
		// Key exists: in-place update
		target.valuePtr = valuePtr
		target.isDeleted = isDeleted
		return true
	}

	// Key does not exist: generate probabilistic level
	lvl := sl.randomLevel()
	if lvl > sl.level {
		for i := sl.level; i < lvl; i++ {
			update[i] = sl.head
		}
		sl.level = lvl
	}

	// Create and splice new node
	newNode := &Node{
		key:       bytes.Clone(key),
		valuePtr:  valuePtr,
		isDeleted: isDeleted,
		forward:   make([]*Node, lvl),
	}

	for i := 0; i < lvl; i++ {
		newNode.forward[i] = update[i].forward[i]
		update[i].forward[i] = newNode
	}

	sl.length++
	return false
}

// Get looks up a key in the SkipList.
// If the key is found and not deleted, returns (valuePtr, true).
// If the key is not found or marked deleted, returns (ValuePtr{}, false).
func (sl *SkipList) Get(key []byte) (ValuePtr, bool) {
	sl.mu.RLock()
	defer sl.mu.RUnlock()

	curr := sl.head
	for i := sl.level - 1; i >= 0; i-- {
		for curr.forward[i] != nil && bytes.Compare(curr.forward[i].key, key) < 0 {
			curr = curr.forward[i]
		}
	}

	target := curr.forward[0]
	if target != nil && bytes.Equal(target.key, key) {
		if target.isDeleted {
			return ValuePtr{}, false
		}
		return target.valuePtr, true
	}

	return ValuePtr{}, false
}

// GetRecord searches for a key and returns (valuePtr, isDeleted, found).
// This provides visibility into tombstone status for LSM-tree level searches.
func (sl *SkipList) GetRecord(key []byte) (ValuePtr, bool, bool) {
	sl.mu.RLock()
	defer sl.mu.RUnlock()

	curr := sl.head
	for i := sl.level - 1; i >= 0; i-- {
		for curr.forward[i] != nil && bytes.Compare(curr.forward[i].key, key) < 0 {
			curr = curr.forward[i]
		}
	}

	target := curr.forward[0]
	if target != nil && bytes.Equal(target.key, key) {
		return target.valuePtr, target.isDeleted, true
	}

	return ValuePtr{}, false, false
}

// Len returns the number of unique keys in the SkipList.
func (sl *SkipList) Len() int {
	sl.mu.RLock()
	defer sl.mu.RUnlock()
	return sl.length
}

// Iterator returns an in-order iterator over a point-in-time snapshot of the SkipList.
// This guarantees thread safety during iteration even while concurrent writes proceed.
func (sl *SkipList) Iterator() *Iterator {
	sl.mu.RLock()
	defer sl.mu.RUnlock()

	nodes := make([]*Node, 0, sl.length)
	curr := sl.head.forward[0]
	for curr != nil {
		nodes = append(nodes, curr)
		curr = curr.forward[0]
	}

	return &Iterator{
		nodes:  nodes,
		cursor: -1,
	}
}

// Iterator provides sorted in-order traversal of SkipList nodes.
// Supports both `for it.Next() { ... }` and `for it.SeekToFirst(); it.Valid(); it.Next() { ... }`.
type Iterator struct {
	nodes   []*Node
	cursor  int
	started bool
}

// SeekToFirst positions the iterator at the first entry.
func (it *Iterator) SeekToFirst() {
	it.cursor = 0
	it.started = true
}

// Seek positions the iterator at the first entry with key >= target.
func (it *Iterator) Seek(target []byte) {
	it.started = true
	it.cursor = sort.Search(len(it.nodes), func(i int) bool {
		return bytes.Compare(it.nodes[i].key, target) >= 0
	})
}

// Valid reports whether the iterator is currently positioned at a valid entry.
func (it *Iterator) Valid() bool {
	return it.cursor >= 0 && it.cursor < len(it.nodes)
}

// Next advances the iterator to the next entry.
// Returns true if the next position is valid.
func (it *Iterator) Next() bool {
	if !it.started {
		it.started = true
		it.cursor = 0
	} else {
		it.cursor++
	}
	return it.Valid()
}

// Key returns the key of the current entry, or nil if invalid.
func (it *Iterator) Key() []byte {
	if !it.Valid() {
		return nil
	}
	return it.nodes[it.cursor].key
}

// Value returns the ValuePtr of the current entry, or ValuePtr{} if invalid.
func (it *Iterator) Value() ValuePtr {
	if !it.Valid() {
		return ValuePtr{}
	}
	return it.nodes[it.cursor].valuePtr
}

// IsDeleted reports whether the current entry is marked as deleted.
func (it *Iterator) IsDeleted() bool {
	if !it.Valid() {
		return false
	}
	return it.nodes[it.cursor].isDeleted
}

// Entry returns all fields of the current node at once.
func (it *Iterator) Entry() (key []byte, val ValuePtr, isDeleted bool, valid bool) {
	if !it.Valid() {
		return nil, ValuePtr{}, false, false
	}
	n := it.nodes[it.cursor]
	return n.key, n.valuePtr, n.isDeleted, true
}
