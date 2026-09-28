package lsm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// SSTableMeta holds metadata for an active SSTable tracked by the Manifest.
type SSTableMeta struct {
	ID    uint64 `json:"id"`
	Level int    `json:"level"`
	Path  string `json:"path"`
	Size  int64  `json:"size"`
}

// manifestData is the serializable structure stored in the manifest file.
type manifestData struct {
	NextSSTID uint64         `json:"next_sst_id"`
	Tables    []*SSTableMeta `json:"tables"`
}

// Manifest tracks active SSTable file IDs, tiers, and levels, providing
// atomic commits of state changes across flushes and compactions.
type Manifest struct {
	mu        sync.RWMutex
	dir       string
	path      string
	nextSSTID uint64
	tables    map[uint64]*SSTableMeta
}

// OpenManifest loads an existing manifest or creates a new one in dir.
func OpenManifest(dir string) (*Manifest, error) {
	manifestDir := filepath.Join(dir, "manifest")
	if err := os.MkdirAll(manifestDir, 0755); err != nil {
		return nil, fmt.Errorf("lsm: failed creating manifest directory: %w", err)
	}

	manifestPath := filepath.Join(manifestDir, "MANIFEST.json")
	m := &Manifest{
		dir:       dir,
		path:      manifestPath,
		tables:    make(map[uint64]*SSTableMeta),
		nextSSTID: 1,
	}

	if _, err := os.Stat(manifestPath); err == nil {
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			return nil, fmt.Errorf("lsm: failed reading manifest: %w", err)
		}

		var md manifestData
		if err := json.Unmarshal(data, &md); err != nil {
			return nil, fmt.Errorf("lsm: corrupted manifest json: %w", err)
		}

		m.nextSSTID = md.NextSSTID
		for _, t := range md.Tables {
			m.tables[t.ID] = t
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	} else {
		// New manifest
		if err := m.saveLocked(); err != nil {
			return nil, err
		}
	}

	return m, nil
}

// AllocateSSTID atomically returns a new unique SSTable sequence ID.
func (m *Manifest) AllocateSSTID() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := m.nextSSTID
	m.nextSSTID++
	_ = m.saveLocked()
	return id
}

// EnsureNextSSTID ensures nextSSTID is at least minID + 1.
func (m *Manifest) EnsureNextSSTID(minID uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if minID >= m.nextSSTID {
		m.nextSSTID = minID + 1
		_ = m.saveLocked()
	}
}

// AddTable registers a newly flushed SSTable in the manifest.
func (m *Manifest) AddTable(meta *SSTableMeta) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.tables[meta.ID] = meta
	if meta.ID >= m.nextSSTID {
		m.nextSSTID = meta.ID + 1
	}
	return m.saveLocked()
}

// RemoveTable unregisters an obsolete SSTable.
func (m *Manifest) RemoveTable(id uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.tables, id)
	return m.saveLocked()
}

// ApplyCompaction atomically registers newly created consolidated SSTables
// and removes obsolete SSTables in a single persistent transaction.
func (m *Manifest) ApplyCompaction(added []*SSTableMeta, removed []uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, id := range removed {
		delete(m.tables, id)
	}
	for _, meta := range added {
		m.tables[meta.ID] = meta
		if meta.ID >= m.nextSSTID {
			m.nextSSTID = meta.ID + 1
		}
	}

	return m.saveLocked()
}

// ActiveTables returns all currently active SSTables sorted by ID ascending.
func (m *Manifest) ActiveTables() []*SSTableMeta {
	m.mu.RLock()
	defer m.mu.RUnlock()

	tables := make([]*SSTableMeta, 0, len(m.tables))
	for _, t := range m.tables {
		tables = append(tables, &SSTableMeta{
			ID:    t.ID,
			Level: t.Level,
			Path:  t.Path,
			Size:  t.Size,
		})
	}

	sort.Slice(tables, func(i, j int) bool {
		return tables[i].ID < tables[j].ID
	})
	return tables
}

// TableCount returns the total number of active SSTables.
func (m *Manifest) TableCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.tables)
}

// saveLocked writes the manifest atomically via temp file rename.
func (m *Manifest) saveLocked() error {
	tables := make([]*SSTableMeta, 0, len(m.tables))
	for _, t := range m.tables {
		tables = append(tables, t)
	}
	sort.Slice(tables, func(i, j int) bool {
		return tables[i].ID < tables[j].ID
	})

	md := manifestData{
		NextSSTID: m.nextSSTID,
		Tables:    tables,
	}

	data, err := json.MarshalIndent(md, "", "  ")
	if err != nil {
		return err
	}

	tmpPath := fmt.Sprintf("%s.tmp", m.path)
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("lsm: failed writing temp manifest: %w", err)
	}

	// Atomic rename ensures crash-consistency
	if err := os.Rename(tmpPath, m.path); err != nil {
		return fmt.Errorf("lsm: failed renaming manifest file: %w", err)
	}

	return nil
}
