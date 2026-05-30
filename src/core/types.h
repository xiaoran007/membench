#ifndef MEMBENCH_CORE_TYPES_H
#define MEMBENCH_CORE_TYPES_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace membench {

constexpr std::size_t KB = 1024;
constexpr std::size_t MB = 1024 * KB;
constexpr std::size_t GB = 1024 * MB;
constexpr std::size_t kCacheLineSize = 64;
constexpr std::size_t kDefaultWarmupIterations = 2;
constexpr std::size_t kDefaultMeasuredIterations = 7;
constexpr std::size_t kCalibrationWarmupIterations = 1;
constexpr std::size_t kCalibrationMeasuredIterations = 3;
constexpr std::size_t kCalibrationPassesPerIteration = 1;
constexpr std::size_t kMinDefaultBufferSize = 256 * MB;
constexpr std::size_t kMaxDefaultBufferSize = 1 * GB;
constexpr std::size_t kCalibrationBufferSize = 1 * GB;
constexpr std::size_t kMaxBufferSize = 16 * GB;

enum class TestKind {
    Read,
    Write,
    Copy,
};

enum class ThreadPolicy {
    Perf,
    All,
};

enum class RunMode {
    Standard,
    Peak,
};

enum class Backend {
    Cpu,
    Metal,
    Auto,
};

enum class KernelKind {
    ScalarAuto,
    NeonPeak,
    LibcMemset,
    NeonStore,
    LibcMemcpy,
    NeonCopy,
    Avx2Read,
    Avx2StreamStore,
    Avx2StreamCopy,
    MetalRead,
    MetalWrite,
    MetalCopy,
};

struct PlatformInfo {
    std::size_t page_size = 4096;
    std::uint64_t physical_memory_bytes = 0;
    unsigned int hardware_threads = 1;
    unsigned int performance_cores = 0;
    bool apple_silicon = false;
    bool x86_avx2 = false;
    std::vector<unsigned int> cpu_affinity_order;
};

struct BenchmarkOptions {
    std::size_t size_bytes = 0;
    std::size_t warmup_iterations = kDefaultWarmupIterations;
    std::size_t measured_iterations = kDefaultMeasuredIterations;
    unsigned int threads_override = 0;
    ThreadPolicy thread_policy = ThreadPolicy::All;
    RunMode mode = RunMode::Standard;
    Backend backend = Backend::Cpu;
    bool calibrate = false;
    bool use_qos = false;
    std::vector<TestKind> tests;
};

struct Statistics {
    double average = 0.0;
    double median = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double stdev = 0.0;
};

struct TestResult {
    Statistics bandwidth_mb_per_sec;
    Statistics elapsed_ms;
    std::size_t logical_bytes_per_iteration = 0;
};

struct ExecutionPlan {
    KernelKind kernel = KernelKind::ScalarAuto;
    unsigned int selected_threads = 1;
    bool calibrated = false;
};

struct CalibrationCandidate {
    KernelKind kernel = KernelKind::ScalarAuto;
    unsigned int requested_threads = 1;
    unsigned int actual_threads = 1;
    double score_mb_per_sec = 0.0;
};

}  // namespace membench

#endif  // MEMBENCH_CORE_TYPES_H
