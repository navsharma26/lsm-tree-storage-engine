package lsm

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// vlogHeaderSize: CRC32 (4B) + KeyLen (2B) + ValLen (4B) = 10 bytes.
const vlogHeaderSize = 10

// VLog represents an append-only Value Log for the WiscKey storage architecture.
//
// Entry format:
// [CRC32 (4B)][KeyLen (2B)][ValLen (4B)][Key (KeyLen bytes)][Value (ValLen bytes)]
//
// Checksum is computed over:
// [KeyLen (2B)][ValLen (4B)][Key][Value]
type VLog struct {
	mu            sync.Mutex
	file          *os.File
	writer        *bufio.Writer
	fileID        uint32
	offset        uint64
	flushedOffset uint64
	path          string
	closed        bool
}

// OpenVLog opens or creates a Value Log file with the specified fileID.
func OpenVLog(path string, fileID uint32) (*VLog, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("lsm: failed to create VLog directory: %w", err)
	}

	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("lsm: failed to open VLog file: %w", err)
	}

	stat, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("lsm: failed to stat VLog file: %w", err)
	}

	size := uint64(stat.Size())
	return &VLog{
		file:          file,
		writer:        bufio.NewWriterSize(file, 256*1024), // 256KB buffer for high write throughput
		fileID:        fileID,
		offset:        size,
		flushedOffset: size,
		path:          path,
	}, nil
}

// Write appends a key-value record to the VLog using buffered I/O.
// Returns a ValuePtr referencing the entry offset and value length.
func (v *VLog) Write(key, val []byte) (ValuePtr, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed {
		return ValuePtr{}, os.ErrClosed
	}

	keyLen := len(key)
	if keyLen == 0 {
		return ValuePtr{}, ErrKeyEmpty
	}
	if keyLen > math.MaxUint16 {
		return ValuePtr{}, ErrKeyTooLarge
	}
	valLen := len(val)
	if uint64(valLen) > math.MaxUint32 {
		return ValuePtr{}, fmt.Errorf("lsm: value length %d exceeds uint32 max", valLen)
	}

	entryOffset := v.offset
	entrySize := vlogHeaderSize + keyLen + valLen
	buf := make([]byte, entrySize)

	// Header: KeyLen (2B) + ValLen (4B)
	binary.BigEndian.PutUint16(buf[4:6], uint16(keyLen))
	binary.BigEndian.PutUint32(buf[6:10], uint32(valLen))

	// Payload: Key + Value
	copy(buf[10:10+keyLen], key)
	copy(buf[10+keyLen:], val)

	// Checksum over [KeyLen (2B)][ValLen (4B)][Key][Value]
	checksum := crc32.ChecksumIEEE(buf[4:])
	binary.BigEndian.PutUint32(buf[0:4], checksum)

	// Buffered write
	if _, err := v.writer.Write(buf); err != nil {
		return ValuePtr{}, fmt.Errorf("lsm: failed to write to VLog buffer: %w", err)
	}
	if err := v.writer.Flush(); err != nil {
		return ValuePtr{}, fmt.Errorf("lsm: failed to flush VLog buffer: %w", err)
	}

	v.offset += uint64(entrySize)
	v.flushedOffset = v.offset

	return ValuePtr{
		FileID: v.fileID,
		Offset: entryOffset,
		Len:    uint32(valLen),
	}, nil
}

// Read reads a value from the VLog using random access io.ReadAt.
// It verifies the CRC32 checksum before returning the value slice.
func (v *VLog) Read(ptr ValuePtr) ([]byte, error) {
	_, val, err := v.ReadRecord(ptr)
	return val, err
}

// ReadRecord retrieves both the key and the value from the VLog at ptr.Offset.
// This is essential for WiscKey garbage collection and recovery.
func (v *VLog) ReadRecord(ptr ValuePtr) ([]byte, []byte, error) {
	// Ensure buffered data is committed to the file if reading newly written records
	v.ensureFlushed(ptr.Offset)

	// Read header: [CRC32 (4B)][KeyLen (2B)][ValLen (4B)] = 10 bytes
	header := make([]byte, vlogHeaderSize)
	if _, err := v.file.ReadAt(header, int64(ptr.Offset)); err != nil {
		return nil, nil, fmt.Errorf("lsm: failed to read VLog header at offset %d: %w", ptr.Offset, err)
	}

	expectedCRC := binary.BigEndian.Uint32(header[0:4])
	keyLen := int(binary.BigEndian.Uint16(header[4:6]))
	valLen := int(binary.BigEndian.Uint32(header[6:10]))

	// Read payload: [Key][Value]
	payload := make([]byte, keyLen+valLen)
	if _, err := v.file.ReadAt(payload, int64(ptr.Offset)+vlogHeaderSize); err != nil {
		return nil, nil, fmt.Errorf("lsm: failed to read VLog payload at offset %d: %w", ptr.Offset+vlogHeaderSize, err)
	}

	// Verify CRC32
	h := crc32.NewIEEE()
	h.Write(header[4:10])
	h.Write(payload)
	if h.Sum32() != expectedCRC {
		return nil, nil, ErrCorruptedRecord
	}

	key := payload[:keyLen]
	val := payload[keyLen:]
	return key, val, nil
}

// Flush flushes buffered write data to the underlying OS file.
func (v *VLog) Flush() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed {
		return os.ErrClosed
	}

	if err := v.writer.Flush(); err != nil {
		return err
	}
	v.flushedOffset = v.offset
	return nil
}

// Sync flushes buffered data and commits changes to stable storage (fsync).
func (v *VLog) Sync() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed {
		return os.ErrClosed
	}

	if err := v.writer.Flush(); err != nil {
		return err
	}
	v.flushedOffset = v.offset
	return v.file.Sync()
}

// Close flushes and closes the VLog file.
func (v *VLog) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed {
		return nil
	}
	v.closed = true

	if err := v.writer.Flush(); err != nil {
		_ = v.file.Close()
		return err
	}
	if err := v.file.Sync(); err != nil {
		_ = v.file.Close()
		return err
	}
	return v.file.Close()
}

// FileID returns the VLog file identifier.
func (v *VLog) FileID() uint32 {
	return v.fileID
}

// Offset returns the current write offset of the VLog.
func (v *VLog) Offset() uint64 {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.offset
}

// Path returns the filesystem path of the VLog file.
func (v *VLog) Path() string {
	return v.path
}

func (v *VLog) ensureFlushed(offset uint64) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if offset >= v.flushedOffset && !v.closed {
		_ = v.writer.Flush()
		v.flushedOffset = v.offset
	}
}
