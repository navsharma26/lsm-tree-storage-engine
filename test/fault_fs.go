package test

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"sync"
)

var (
	// ErrInjectedFault indicates a write failure injected by the test harness.
	ErrInjectedFault = errors.New("fault_fs: injected hardware I/O fault")
)

// FaultFile wraps an underlying *os.File to simulate abrupt power loss,
// kernel panics, and torn writes during critical write paths.
type FaultFile struct {
	mu             sync.Mutex
	file           *os.File
	path           string
	bytesRemaining int
	limitActive    bool
	panicOnLimit   bool
	corruptOnLimit bool
	lastOffset     int64
	lastWriteLen   int
	totalWritten   int64
	closed         bool
}

// NewFaultFile wraps an existing *os.File with fault injection capabilities.
func NewFaultFile(file *os.File) *FaultFile {
	stat, _ := file.Stat()
	var size int64
	if stat != nil {
		size = stat.Size()
	}
	return &FaultFile{
		file:         file,
		path:         file.Name(),
		lastOffset:   size,
		totalWritten: size,
	}
}

// PanicAfterNBytes instructs the FaultFile to panic abruptly after writing n bytes,
// simulating a hard machine crash or kernel panic mid-write.
func (f *FaultFile) PanicAfterNBytes(n int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bytesRemaining = n
	f.limitActive = true
	f.panicOnLimit = true
	f.corruptOnLimit = false
}

// FailAfterNBytes instructs the FaultFile to return ErrInjectedFault after writing n bytes.
func (f *FaultFile) FailAfterNBytes(n int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bytesRemaining = n
	f.limitActive = true
	f.panicOnLimit = false
	f.corruptOnLimit = false
}

// CorruptOnLimit instructs the FaultFile to write partial garbage bytes and halt
// when the limit is exhausted, simulating a torn write.
func (f *FaultFile) CorruptOnLimit(n int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bytesRemaining = n
	f.limitActive = true
	f.panicOnLimit = false
	f.corruptOnLimit = true
}

// Write intercepts writes to track byte counts and trigger failure when limits expire.
func (f *FaultFile) Write(p []byte) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.closed {
		return 0, os.ErrClosed
	}

	if f.limitActive {
		if f.bytesRemaining <= 0 {
			if f.panicOnLimit {
				panic("fault_fs: simulated kernel panic / sudden power loss")
			}
			return 0, ErrInjectedFault
		}

		if len(p) > f.bytesRemaining {
			// Write partial bytes up to remaining limit
			allowed := f.bytesRemaining
			partial := p[:allowed]

			if f.corruptOnLimit {
				// Corrupt the tail bytes to simulate torn write
				corrupted := make([]byte, allowed)
				copy(corrupted, partial)
				for i := len(corrupted) / 2; i < len(corrupted); i++ {
					corrupted[i] ^= 0xFF
				}
				partial = corrupted
			}

			n, err := f.file.Write(partial)
			f.lastOffset = f.totalWritten
			f.lastWriteLen = n
			f.totalWritten += int64(n)
			f.bytesRemaining = 0

			if f.panicOnLimit {
				panic("fault_fs: simulated kernel panic mid-write")
			}
			if err != nil {
				return n, err
			}
			return n, ErrInjectedFault
		}

		f.bytesRemaining -= len(p)
	}

	n, err := f.file.Write(p)
	f.lastOffset = f.totalWritten
	f.lastWriteLen = n
	f.totalWritten += int64(n)
	return n, err
}

// ReadAt delegates to underlying file.
func (f *FaultFile) ReadAt(p []byte, off int64) (int, error) {
	return f.file.ReadAt(p, off)
}

// Read delegates to underlying file.
func (f *FaultFile) Read(p []byte) (int, error) {
	return f.file.Read(p)
}

// Seek delegates to underlying file.
func (f *FaultFile) Seek(offset int64, whence int) (int64, error) {
	return f.file.Seek(offset, whence)
}

// Sync delegates to underlying file.
func (f *FaultFile) Sync() error {
	return f.file.Sync()
}

// Close closes the underlying file.
func (f *FaultFile) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = true
	return f.file.Close()
}

// Stat returns FileInfo.
func (f *FaultFile) Stat() (os.FileInfo, error) {
	return f.file.Stat()
}

// Truncate truncates file.
func (f *FaultFile) Truncate(size int64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.totalWritten = size
	return f.file.Truncate(size)
}

// Name returns file path.
func (f *FaultFile) Name() string {
	return f.path
}

// CorruptLastWrite flips bits in the last written block of the file to simulate
// storage media bit corruption or torn block writing.
func (f *FaultFile) CorruptLastWrite() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.lastWriteLen <= 0 {
		return errors.New("fault_fs: no previous write recorded")
	}

	corruptBuf := make([]byte, f.lastWriteLen)
	if _, err := f.file.ReadAt(corruptBuf, f.lastOffset); err != nil && err != io.EOF {
		return err
	}

	// Invert bytes to guarantee checksum failure
	for i := range corruptBuf {
		corruptBuf[i] ^= 0xAA
	}

	_, err := f.file.WriteAt(corruptBuf, f.lastOffset)
	return err
}

// -----------------------------------------------------------------------------
// Direct File System Fault Injection Helpers
// -----------------------------------------------------------------------------

// InjectTornWrite appends partial garbage bytes to path without valid record headers.
func InjectTornWrite(path string, garbage []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("fault_fs: failed to open file for torn write: %w", err)
	}
	defer f.Close()

	if _, err := f.Write(garbage); err != nil {
		return fmt.Errorf("fault_fs: failed appending garbage: %w", err)
	}
	return f.Sync()
}

// CorruptFileTailCRC flips the CRC32 checksum bytes of the last record in a WAL or VLog file.
func CorruptFileTailCRC(path string, recordHeaderOffset int64) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		return fmt.Errorf("fault_fs: failed opening file: %w", err)
	}
	defer f.Close()

	crcBuf := make([]byte, 4)
	if _, err := f.ReadAt(crcBuf, recordHeaderOffset); err != nil {
		return fmt.Errorf("fault_fs: failed reading CRC at offset %d: %w", recordHeaderOffset, err)
	}

	// Invert the CRC bytes
	crc := binary.BigEndian.Uint32(crcBuf)
	binary.BigEndian.PutUint32(crcBuf, crc^0xFFFFFFFF)

	if _, err := f.WriteAt(crcBuf, recordHeaderOffset); err != nil {
		return fmt.Errorf("fault_fs: failed writing corrupted CRC: %w", err)
	}
	return f.Sync()
}

// TruncateTail removes dropBytes from the end of a file to simulate an incomplete write.
func TruncateTail(path string, dropBytes int64) error {
	stat, err := os.Stat(path)
	if err != nil {
		return err
	}
	newSize := stat.Size() - dropBytes
	if newSize < 0 {
		newSize = 0
	}
	return os.Truncate(path, newSize)
}

// CorruptLastByte flips the single final byte of a file.
func CorruptLastByte(path string) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return err
	}
	if stat.Size() == 0 {
		return nil
	}

	lastByte := make([]byte, 1)
	if _, err := f.ReadAt(lastByte, stat.Size()-1); err != nil {
		return err
	}
	lastByte[0] ^= 0xFF

	if _, err := f.WriteAt(lastByte, stat.Size()-1); err != nil {
		return err
	}
	return f.Sync()
}

// ComputeCRC32 helper for test assertions.
func ComputeCRC32(data []byte) uint32 {
	return crc32.ChecksumIEEE(data)
}
