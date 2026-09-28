package lsm

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// RecordType specifies the operation type for a WAL or MemTable record.
type RecordType uint8

const (
	// RecordPut indicates a standard write/update operation.
	RecordPut RecordType = 0x01

	// RecordDelete indicates a tombstone deletion operation.
	RecordDelete RecordType = 0x02
)

// String returns a human-readable representation of RecordType.
func (rt RecordType) String() string {
	switch rt {
	case RecordPut:
		return "PUT"
	case RecordDelete:
		return "DELETE"
	default:
		return fmt.Sprintf("UNKNOWN(0x%02X)", uint8(rt))
	}
}

// ValuePtr represents a pointer to a value stored in the Value Log (vLog)
// as part of the WiscKey key-value separation architecture.
// Total encoded size is 16 bytes: FileID (4B) + Offset (8B) + Len (4B).
type ValuePtr struct {
	FileID uint32 // ID of the vLog file.
	Offset uint64 // Byte offset in the vLog file.
	Len    uint32 // Length in bytes of the value record in the vLog.
}

// ValuePtrSize defines the fixed serialized binary size of a ValuePtr.
const ValuePtrSize = 16

// Encode serializes the ValuePtr into a fixed 16-byte slice using BigEndian encoding.
func (vp ValuePtr) Encode() []byte {
	buf := make([]byte, ValuePtrSize)
	vp.EncodeTo(buf)
	return buf
}

// EncodeTo encodes the ValuePtr directly into dst, which must have at least ValuePtrSize (16) bytes.
func (vp ValuePtr) EncodeTo(dst []byte) {
	_ = dst[15] // eliminate bounds checks
	binary.BigEndian.PutUint32(dst[0:4], vp.FileID)
	binary.BigEndian.PutUint64(dst[4:12], vp.Offset)
	binary.BigEndian.PutUint32(dst[12:16], vp.Len)
}

// DecodeValuePtr deserializes a 16-byte slice into a ValuePtr.
func DecodeValuePtr(b []byte) (ValuePtr, error) {
	if len(b) < ValuePtrSize {
		return ValuePtr{}, ErrCorruptedRecord
	}
	return ValuePtr{
		FileID: binary.BigEndian.Uint32(b[0:4]),
		Offset: binary.BigEndian.Uint64(b[4:12]),
		Len:    binary.BigEndian.Uint32(b[12:16]),
	}, nil
}

// IsZero reports whether the ValuePtr is empty/uninitialized.
func (vp ValuePtr) IsZero() bool {
	return vp.FileID == 0 && vp.Offset == 0 && vp.Len == 0
}

// String returns a formatted representation of ValuePtr.
func (vp ValuePtr) String() string {
	return fmt.Sprintf("ValuePtr(FileID=%d, Offset=%d, Len=%d)", vp.FileID, vp.Offset, vp.Len)
}

// Core error types required by the storage engine specification.
var (
	// ErrKeyNotFound is returned when a requested key does not exist or has been deleted.
	ErrKeyNotFound = errors.New("lsm: key not found")

	// ErrCorruptedRecord is returned when checksum verification fails or payload is incomplete.
	ErrCorruptedRecord = errors.New("lsm: corrupted record")

	// ErrKeyEmpty is returned when an operation is attempted with an empty key.
	ErrKeyEmpty = errors.New("lsm: key cannot be empty")

	// ErrKeyTooLarge is returned when a key exceeds uint16 max (65,535 bytes).
	ErrKeyTooLarge = errors.New("lsm: key exceeds maximum length of 65535 bytes")

	// ErrWALClosed is returned when an operation is attempted on a closed WAL.
	ErrWALClosed = errors.New("lsm: WAL is closed")
)

// LogEntry represents an entry stored in the Write-Ahead Log (WAL).
type LogEntry struct {
	Type     RecordType
	Key      []byte
	ValuePtr ValuePtr
}
