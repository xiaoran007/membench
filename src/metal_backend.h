#ifndef MEMBENCH_METAL_BACKEND_H
#define MEMBENCH_METAL_BACKEND_H

#ifdef MEMBENCH_HAS_METAL

#include <cstddef>
#include <string>
#include <vector>

enum class MetalTestKind {
    Read,
    Write,
    Copy,
};

struct MetalBenchResult {
    double bandwidth_mb_per_sec = 0.0;
    double elapsed_ms = 0.0;
    std::size_t logical_bytes = 0;
};

struct MetalIterationResult {
    std::vector<double> bandwidth_samples;
    std::vector<double> elapsed_samples;
    std::size_t logical_bytes_per_iteration = 0;
};

// Returns true if a Metal GPU device is available.
bool metalIsAvailable();

// Returns a human-readable description of the Metal GPU device.
std::string metalDeviceName();

// Runs multiple iterations of a Metal bandwidth test and returns per-iteration
// samples suitable for statistics calculation by the caller.
MetalIterationResult metalRunBandwidthTest(
    MetalTestKind kind,
    std::size_t buffer_size_bytes,
    std::size_t warmup_iterations,
    std::size_t measured_iterations);

#endif // MEMBENCH_HAS_METAL
#endif // MEMBENCH_METAL_BACKEND_H
