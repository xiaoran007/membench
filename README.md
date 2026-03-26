# MemBench

MemBench is a cross-platform memory bandwidth benchmark focused on reliable `read`, `write`, and `copy` measurements. The current implementation is tuned first for macOS, especially Apple Silicon, while keeping portable fallbacks for Linux and Windows.

## What It Measures

- Sequential read bandwidth
- Sequential write bandwidth
- Memory copy throughput (`memcpy`)

The benchmark now uses:

- Page-aligned buffers
- One global timer per measured iteration
- Reusable worker threads instead of recreating threads every round
- Separate warmup and measured phases
- Median and standard deviation in addition to average/min/max

On Apple Silicon, the default policy favors performance cores and uses QoS hints by default to reduce scheduler noise.

## Build Requirements

- CMake 3.10+
- A C++17 compiler
  - Apple Clang 17+ recommended on macOS
  - GCC 7+ or Clang 5+ on Linux
  - MSVC 2017+ on Windows

## Build

### macOS / Linux

```bash
./build.sh
```

Or manually:

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
./membench
```

### Windows

```cmd
build.bat
```

Or manually:

```cmd
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
Release\membench.exe
```

## Usage

Basic usage:

```bash
./membench
```

Compatible positional size argument:

```bash
./membench 1024
```

Explicit options:

```bash
./membench --size-mb 1024 --tests read,copy
./membench --threads 4 --warmup 2 --iterations 7
./membench --size-mb 1024 --thread-policy perf
```

### CLI Options

- `--size-mb <n>`: buffer size per buffer in MiB
- `--threads <n>`: override worker thread count
- `--tests read,write,copy`: choose a comma-separated subset
- `--iterations <n>`: measured iterations per test
- `--warmup <n>`: warmup iterations per test
- `--thread-policy perf|all`: use performance-core-biased or all-thread defaults
- `--no-qos`: disable macOS QoS hinting
- `--help`: print usage

## Defaults

- Default buffer size: `min(1 GiB, physical_memory / 8)`, with a floor of `256 MiB`
- Default measured iterations: `7`
- Default warmup iterations: `2`
- Default Apple Silicon thread policy: `perf`
- Default non-Apple thread policy: `all`

## Interpreting Results

Each test prints:

- `size_mb`
- `threads`
- `warmup`
- `iterations`
- `logical_bytes_per_iteration`
- `measured_elapsed_ms`
- `avg / median / min / max / stdev bandwidth`

For `copy`, the reported bandwidth is logical copied bytes per second. It is not doubled to estimate total DRAM bus traffic.

## Notes On Accuracy

- Larger buffers better reflect main-memory behavior; small buffers can be influenced by cache.
- Background activity, thermal throttling, and memory pressure can still affect results.
- On Apple Silicon, `perf` mode usually gives more stable numbers than using all cores.

## Out Of Scope In This Version

This version does not implement:

- Random latency measurement
- Hand-written SIMD kernels
- Non-temporal store benchmarks

Those were intentionally removed from the documentation because they are not present in the current codebase.
