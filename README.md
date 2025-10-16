# MemBench - Memory Performance Benchmark Tool


A cross-platform memory performance benchmark tool for testing memory read, write, copy speeds, and latency.

## Features

- ✅ **Cross-Platform Support**: Linux, macOS, Windows
- 📊 **Multiple Test Modes**:
  - Sequential Read Test
  - Sequential Write Test
  - Memory Copy Test (memcpy)
  - Random Access Latency Test
  - SIMD Optimized Tests (AVX2)
  - Non-Temporal Streaming Writes
- 🚀 **High-Precision Timing**: Using high-resolution clocks
- ⚡ **Performance Optimizations**: Loop unrolling, SIMD instructions, memory barriers
- 📈 **Detailed Statistics**: Average, minimum, maximum values

## Build Requirements

- CMake 3.10 or higher
- C++17 compatible compiler:
  - GCC 7+ (Linux)
  - Clang 5+ (macOS/Linux)
  - MSVC 2017+ (Windows)

## Build Instructions

### Linux / macOS

```bash
# Create build directory
mkdir build
cd build

# Configure project (with native CPU optimizations)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build .

# Run
./membench
```

### Windows

```cmd
# Create build directory
mkdir build
cd build

# Configure project (using Visual Studio)
cmake ..

# Build
cmake --build . --config Release

# Run
Release\membench.exe
```

Or using MinGW:

```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
cmake --build .
membench.exe
```

## Usage

### Basic Usage

```bash
./membench
```

Default test uses a 64 MB buffer.

### Custom Buffer Size

```bash
# Use 128 MB buffer
./membench 128

# Use 256 MB buffer
./membench 256
```

Buffer size is specified in MB.

## Test Description

### 1. Sequential Read Test
Tests sequential memory read bandwidth with loop unrolling optimization, simulating streaming data processing scenarios.

### 2. Sequential Write Test
Tests sequential memory write bandwidth with loop unrolling optimization, evaluating memory write performance.

### 3. Memory Copy Test
Uses the `memcpy` function to test memory-to-memory copy performance, the most common operation in real applications.

### 4. SIMD Optimized Read Test (x86/x64 only)
Uses AVX2 vector instructions (256-bit) for highly optimized sequential reads, processing 128 bytes per iteration.

### 5. Streaming Write Test (x86/x64 only)
Uses non-temporal stores (bypasses CPU cache) for maximum write bandwidth, ideal for large data writes.

### 6. Random Access Latency Test
Tests average latency of random memory accesses in nanoseconds, reflecting memory response time.


## Performance Tips

1. **Build Optimization**: Always use Release mode (`-DCMAKE_BUILD_TYPE=Release`)
2. **Native Instructions**: Use `-march=native` for CPU-specific optimizations (enabled by default)
3. **Buffer Size**: Adjust test buffer size based on your system memory
4. **Background Processes**: Close other memory-intensive applications during testing
5. **Multiple Runs**: The program automatically runs multiple iterations and computes statistics
6. **CPU Frequency**: Lock CPU frequency to performance mode for consistent results

## Technical Details

### Optimizations Applied

- **Loop Unrolling**: 8-way unrolling with multiple accumulators for improved ILP (Instruction Level Parallelism)
- **SIMD Instructions**: AVX2 vector operations for x86/x64 platforms (256-bit registers)
- **Memory Barriers**: Prevent instruction reordering while avoiding performance overhead
- **Cache Warmup**: Pre-touch all pages before measurement to eliminate TLB misses
- **Non-Temporal Stores**: Stream writes that bypass cache for large data transfers
- **High-Resolution Timing**: Uses `std::chrono::high_resolution_clock` for nanosecond precision
- **Compiler Barriers**: Inline assembly to prevent code elimination without runtime cost
- **Random Initialization**: Prevents compression optimizations that could skew results


## Architecture Support

- **x86/x64**: Full support including AVX2 SIMD and streaming stores
- **ARM/Apple Silicon**: Basic optimizations (loop unrolling, barriers)
- **Other**: Portable C++17 implementation

