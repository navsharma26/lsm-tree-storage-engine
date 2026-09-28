package lsm

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

const (
	// walHeaderSize is the size of the record header excluding CRC32:
	// RecordType (1B) + KeyLen (2B) + ValuePtr (16B) = 19 bytes.
	walHeaderSize = 1 + 2 + ValuePtrSize

	// walPrefixSize is CRC32 (4B) + walHeaderSize (19B) = 23 bytes.
	walPrefixSize = 4 + walHeaderSize
)

// WAL represents an append-only Write-Ahead Log supporting crash recovery.
//
// Record Layout on disk:
// [CRC32 Checksum (4B)][RecordType (1B)][KeyLen (2B)][ValuePtr (16B)][Payload (KeyLen bytes)]
//
// Checksum is computed over:
// [RecordType (1B)][KeyLen (2B)][ValuePtr (16B)][Payload (KeyLen bytes)]
type WAL struct {
	mu     sync.Mutex
	file   *os.File
	path   string
	size   int64
	closed bool
}

// OpenWAL opens or creates a Write-Ahead Log at the specified path.
// It opens the file with read-write access and append mode.
func OpenWAL(path string) (*WAL, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("lsm: failed to create WAL directory: %w", err)
	}

	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("lsm: failed to open WAL file: %w", err)
	}

	stat, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("lsm: failed to stat WAL file: %w", err)
	}

	return &WAL{
		file: file,
		path: path,
		size: stat.Size(),
	}, nil
}

// Append writes a single LogEntry to the WAL.
// The operation is thread-safe and updates the current log size.
func (w *WAL) Append(record LogEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return ErrWALClosed
	}

	keyLen := len(record.Key)
	if keyLen == 0 {
		return ErrKeyEmpty
	}
	if keyLen > math.MaxUint16 {
		return ErrKeyTooLarge
	}
	if record.Type != RecordPut && record.Type != RecordDelete {
		return fmt.Errorf("lsm: invalid record type 0x%02X: %w", uint8(record.Type), ErrCorruptedRecord)
	}

	// Total record size = 4 (CRC32) + 19 (Header) + keyLen (Payload)
	totalSize := walPrefixSize + keyLen
	buf := make([]byte, totalSize)

	// Populate Header starting at offset 4
	header := buf[4:]
	header[0] = byte(record.Type)
	binary.BigEndian.PutUint16(header[1:3], uint16(keyLen))
	record.ValuePtr.EncodeTo(header[3:19])
	copy(header[19:], record.Key)

	// Compute CRC32 over [RecordType (1B)][KeyLen (2B)][ValuePtr (16B)][Payload]
	checksum := crc32.ChecksumIEEE(header)
	binary.BigEndian.PutUint32(buf[0:4], checksum)

	// Append to file
	n, err := w.file.Write(buf)
	if err != nil {
		return fmt.Errorf("lsm: failed to write to WAL: %w", err)
	}
	w.size += int64(n)

	return nil
}

// Recover scans the WAL sequentially from offset 0 to EOF, verifying each record's
// CRC32 checksum. If a CRC32 failure or truncated partial write is encountered (e.g.
// following an abrupt system crash), it halts cleanly, truncates the log to the last
// known valid offset, and returns the valid entries recovered.
func (w *WAL) Recover() ([]LogEntry, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil, ErrWALClosed
	}

	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("lsm: failed to seek WAL to start: %w", err)
	}

	reader := bufio.NewReader(w.file)
	var entries []LogEntry
	var validOffset int64

	crcBuf := make([]byte, 4)
	headerBuf := make([]byte, walHeaderSize)

	for {
		// 1. Read CRC32 (4 bytes)
		_, err := io.ReadFull(reader, crcBuf)
		if err == io.EOF {
			// Clean EOF reached at record boundary
			break
		}
		if err == io.ErrUnexpectedEOF {
			// Partial record at tail (crash during CRC write): halt cleanly
			break
		}
		if err != nil {
			return nil, fmt.Errorf("lsm: WAL recovery read error: %w", err)
		}
		expectedCRC := binary.BigEndian.Uint32(crcBuf)

		// 2. Read Header: RecordType (1B) + KeyLen (2B) + ValuePtr (16B) = 19 bytes
		_, err = io.ReadFull(reader, headerBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			// Partial record at tail: halt cleanly
			break
		}
		if err != nil {
			return nil, fmt.Errorf("lsm: WAL recovery read error: %w", err)
		}

		recType := RecordType(headerBuf[0])
		if recType != RecordPut && recType != RecordDelete {
			// Invalid record type indicates torn write or corruption: halt cleanly
			break
		}

		keyLen := int(binary.BigEndian.Uint16(headerBuf[1:3]))
		valPtr, err := DecodeValuePtr(headerBuf[3:19])
		if err != nil {
			// Corrupted ValuePtr payload: halt cleanly
			break
		}

		// 3. Read Payload (KeyLen bytes)
		keyBuf := make([]byte, keyLen)
		_, err = io.ReadFull(reader, keyBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			// Partial record at tail (crash during key write): halt cleanly
			break
		}
		if err != nil {
			return nil, fmt.Errorf("lsm: WAL recovery read error: %w", err)
		}

		// 4. Compute and verify CRC32 Checksum
		h := crc32.NewIEEE()
		h.Write(headerBuf)
		h.Write(keyBuf)
		actualCRC := h.Sum32()

		if actualCRC != expectedCRC {
			// Checksum mismatch (corrupted data): halt cleanly
			break
		}

		// Record is valid
		entries = append(entries, LogEntry{
			Type:     recType,
			Key:      keyBuf,
			ValuePtr: valPtr,
		})

		validOffset += int64(walPrefixSize + keyLen)
	}

	// Truncate any uncommitted/torn write residue at the tail of the log
	if err := w.file.Truncate(validOffset); err != nil {
		return nil, fmt.Errorf("lsm: failed to truncate WAL to valid offset: %w", err)
	}

	if _, err := w.file.Seek(validOffset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("lsm: failed to seek to valid offset: %w", err)
	}

	w.size = validOffset
	return entries, nil
}

// Sync commits the current contents of the WAL to stable storage (fsync).
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return ErrWALClosed
	}
	return w.file.Sync()
}

// Close flushes and closes the WAL file.
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil
	}
	w.closed = true
	if err := w.file.Sync(); err != nil {
		_ = w.file.Close()
		return err
	}
	return w.file.Close()
}

// Size returns the current valid size in bytes of the WAL file.
func (w *WAL) Size() int64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.size
}

// Path returns the filesystem path of the WAL.
func (w *WAL) Path() string {
	return w.path
}
