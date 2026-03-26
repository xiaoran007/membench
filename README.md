# MemBench

MemBench is a cross-platform memory bandwidth benchmark for `read`, `write`, and `copy`. It now has two behaviors:

- `standard`: stable, conservative measurements
- `peak`: Apple Silicon-first peak seeking with short auto-calibration

On Apple Silicon, `./membench` now defaults to `peak` mode and tries to pick the best thread count and kernel per test.

## What It Measures

- Sequential read bandwidth
- Sequential write bandwidth
- Memory copy throughput

The benchmark uses:

- Page-aligned buffers
- One global timer per measured iteration
- Reusable worker threads inside each runner
- Separate warmup and measured phases
- Median and standard deviation in addition to average/min/max
- Apple Silicon NEON peak kernels for selected paths
- Per-test calibration in peak mode

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/membench
```

### Windows

```cmd
build.bat
```

Or manually:

```cmd
cmake -S . -B build
cmake --build build --config Release
.\build\Release\membench.exe
```

## Usage

Basic usage:

```bash
./build/membench
./build/membench 1024
```

Useful examples:

```bash
./build/membench --mode standard
./build/membench --mode peak --size-mb 1024 --tests read,copy
./build/membench --mode peak --no-calibrate
./build/membench --threads 4 --warmup 2 --iterations 7
```

### CLI Options

- `--size-mb <n>`: buffer size per buffer in MiB
- `--threads <n>`: override worker thread count
- `--tests read,write,copy`: choose a comma-separated subset
- `--iterations <n>`: measured iterations per test
- `--warmup <n>`: warmup iterations per test
- `--thread-policy perf|all`: constrain thread selection
- `--mode standard|peak`: choose stable or peak-seeking behavior
- `--no-calibrate`: disable peak-mode calibration
- `--no-qos`: disable macOS QoS hints
- `--help`: print usage

## Defaults

- Default buffer size: `min(1 GiB, physical_memory / 8)`, with a floor of `256 MiB`
- Default measured iterations: `7`
- Default warmup iterations: `2`
- Default Apple Silicon mode: `peak`
- Default Apple Silicon thread policy: `all`
- Default non-Apple mode: `standard`

On Apple Silicon, peak mode calibrates each test independently and may choose different kernels and thread counts for `read`, `write`, and `copy`.

## Interpreting Results

Each test prints:

- `mode`
- `kernel`
- `selected_threads`
- `calibrated`
- `size_mb`
- `warmup`
- `iterations`
- `logical_bytes_per_iteration`
- `measured_elapsed_ms`
- `avg / median / min / max / stdev bandwidth`

For `copy`, MemBench prints two bandwidth views:

- `logical ... bandwidth`: logical memcpy throughput
- `estimated traffic ... bandwidth`: logical throughput multiplied by 2

The estimated traffic number is only a convenience for comparing with vendor memory-bandwidth figures. It is not a hardware counter measurement.

## Notes On Accuracy

- Larger buffers better reflect main-memory behavior; small buffers can be influenced by cache.
- Background activity, thermal throttling, and memory pressure can still affect results.
- `standard` mode is better when you want stable, repeatable comparisons.
- `peak` mode is better when you want Apple Silicon to search for a higher-performing kernel/thread combination.

## Current Specialization

- Apple Silicon has dedicated NEON peak kernels and per-test calibration.
- Linux and Windows keep the portable fallback path.

## Out Of Scope In This Version

This version still does not implement:

- Random latency measurement
- Non-temporal store benchmarks
- Hardware-counter-based DRAM traffic validation
