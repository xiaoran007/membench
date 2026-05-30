#include "version.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#include <malloc.h>
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#if defined(__aarch64__) || defined(__arm64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef MEMBENCH_HAS_METAL
#include "metal_backend.h"
#endif

namespace {

constexpr std::size_t KB = 1024;
constexpr std::size_t MB = 1024 * KB;
constexpr std::size_t GB = 1024 * MB;
constexpr std::size_t kCacheLineSize = 64;
constexpr std::size_t kDefaultWarmupIterations = 2;
constexpr std::size_t kDefaultMeasuredIterations = 7;
constexpr std::size_t kCalibrationWarmupIterations = 1;
constexpr std::size_t kCalibrationMeasuredIterations = 2;
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

std::string formatBytes(std::uint64_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (bytes >= GB) {
        oss << (bytes / static_cast<double>(GB)) << " GiB";
    } else if (bytes >= MB) {
        oss << (bytes / static_cast<double>(MB)) << " MiB";
    } else if (bytes >= KB) {
        oss << (bytes / static_cast<double>(KB)) << " KiB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

std::string testKindToCliName(TestKind kind) {
    switch (kind) {
        case TestKind::Read:
            return "read";
        case TestKind::Write:
            return "write";
        case TestKind::Copy:
            return "copy";
    }
    return "unknown";
}

std::string testKindToTitle(TestKind kind) {
    switch (kind) {
        case TestKind::Read:
            return "Sequential Read";
        case TestKind::Write:
            return "Sequential Write";
        case TestKind::Copy:
            return "Memory Copy";
    }
    return "Unknown";
}

std::string threadPolicyToString(ThreadPolicy policy) {
    return policy == ThreadPolicy::Perf ? "perf" : "all";
}

std::string runModeToString(RunMode mode) {
    return mode == RunMode::Peak ? "peak" : "standard";
}

std::string kernelToString(KernelKind kernel) {
    switch (kernel) {
        case KernelKind::ScalarAuto:
            return "scalar_auto";
        case KernelKind::NeonPeak:
            return "neon_peak";
        case KernelKind::LibcMemset:
            return "libc_memset";
        case KernelKind::NeonStore:
            return "neon_store";
        case KernelKind::LibcMemcpy:
            return "libc_memcpy";
        case KernelKind::NeonCopy:
            return "neon_copy";
        case KernelKind::Avx2Read:
            return "avx2_read";
        case KernelKind::Avx2StreamStore:
            return "avx2_stream_store";
        case KernelKind::Avx2StreamCopy:
            return "avx2_stream_copy";
        case KernelKind::MetalRead:
            return "metal_read";
        case KernelKind::MetalWrite:
            return "metal_write";
        case KernelKind::MetalCopy:
            return "metal_copy";
    }
    return "unknown";
}

bool parseUnsigned64(const std::string& text, std::uint64_t* value) {
    if (value == nullptr || text.empty()) {
        return false;
    }

    std::size_t consumed = 0;
    try {
        const auto parsed = std::stoull(text, &consumed, 10);
        if (consumed != text.size()) {
            return false;
        }
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseThreadPolicy(const std::string& text, ThreadPolicy* policy) {
    if (policy == nullptr) {
        return false;
    }
    if (text == "perf") {
        *policy = ThreadPolicy::Perf;
        return true;
    }
    if (text == "all") {
        *policy = ThreadPolicy::All;
        return true;
    }
    return false;
}

bool parseRunMode(const std::string& text, RunMode* mode) {
    if (mode == nullptr) {
        return false;
    }
    if (text == "standard") {
        *mode = RunMode::Standard;
        return true;
    }
    if (text == "peak") {
        *mode = RunMode::Peak;
        return true;
    }
    return false;
}

std::string backendToString(Backend backend) {
    switch (backend) {
        case Backend::Cpu:   return "cpu";
        case Backend::Metal: return "metal";
        case Backend::Auto:  return "auto";
    }
    return "unknown";
}

bool parseBackend(const std::string& text, Backend* backend) {
    if (backend == nullptr) {
        return false;
    }
    if (text == "cpu") {
        *backend = Backend::Cpu;
        return true;
    }
    if (text == "metal") {
        *backend = Backend::Metal;
        return true;
    }
    if (text == "auto") {
        *backend = Backend::Auto;
        return true;
    }
    return false;
}

bool parseTestList(const std::string& text, std::vector<TestKind>* tests) {
    if (tests == nullptr || text.empty()) {
        return false;
    }

    std::vector<TestKind> parsed;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item == "read") {
            parsed.push_back(TestKind::Read);
        } else if (item == "write") {
            parsed.push_back(TestKind::Write);
        } else if (item == "copy") {
            parsed.push_back(TestKind::Copy);
        } else {
            return false;
        }
    }

    if (parsed.empty()) {
        return false;
    }

    std::sort(parsed.begin(), parsed.end(), [](TestKind lhs, TestKind rhs) {
        return static_cast<int>(lhs) < static_cast<int>(rhs);
    });
    parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());
    *tests = parsed;
    return true;
}

Statistics calculateStatistics(const std::vector<double>& values) {
    if (values.empty()) {
        return {};
    }

    Statistics stats;
    stats.average = std::accumulate(values.begin(), values.end(), 0.0) /
                    static_cast<double>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t midpoint = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        stats.median = (sorted[midpoint - 1] + sorted[midpoint]) / 2.0;
    } else {
        stats.median = sorted[midpoint];
    }
    stats.minimum = sorted.front();
    stats.maximum = sorted.back();

    double squared_sum = 0.0;
    for (double value : values) {
        const double delta = value - stats.average;
        squared_sum += delta * delta;
    }
    stats.stdev = std::sqrt(squared_sum / static_cast<double>(values.size()));
    return stats;
}

Statistics scaleStatistics(const Statistics& stats, double factor) {
    Statistics scaled = stats;
    scaled.average *= factor;
    scaled.median *= factor;
    scaled.minimum *= factor;
    scaled.maximum *= factor;
    scaled.stdev *= factor;
    return scaled;
}

std::uint64_t splitMix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

PlatformInfo detectPlatformInfo() {
    PlatformInfo info;
#ifdef _WIN32
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    info.page_size = system_info.dwPageSize;
    info.hardware_threads = system_info.dwNumberOfProcessors > 0
                                ? system_info.dwNumberOfProcessors
                                : 1;

    MEMORYSTATUSEX memory_status{};
    memory_status.dwLength = sizeof(memory_status);
    if (GlobalMemoryStatusEx(&memory_status)) {
        info.physical_memory_bytes = memory_status.ullTotalPhys;
    }
#else
    const long page_size = sysconf(_SC_PAGESIZE);
    info.page_size = page_size > 0 ? static_cast<std::size_t>(page_size) : 4096;

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    info.hardware_threads = hw_threads > 0 ? hw_threads : 1;

#if defined(__APPLE__)
    std::uint64_t memsize = 0;
    std::size_t memsize_len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &memsize_len, nullptr, 0) == 0) {
        info.physical_memory_bytes = memsize;
    }

    unsigned int perf_cores = 0;
    std::size_t perf_cores_len = sizeof(perf_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &perf_cores_len, nullptr, 0) ==
        0) {
        info.performance_cores = perf_cores;
    }
#if defined(__aarch64__) || defined(__arm64__)
    info.apple_silicon = true;
#endif
#else
    const long phys_pages = sysconf(_SC_PHYS_PAGES);
    if (phys_pages > 0 && page_size > 0) {
        info.physical_memory_bytes =
            static_cast<std::uint64_t>(phys_pages) * static_cast<std::uint64_t>(page_size);
    }

#if defined(__x86_64__) || defined(__i386__)
#if (defined(__GNUC__) || defined(__clang__)) && defined(__AVX2__)
    __builtin_cpu_init();
    info.x86_avx2 = __builtin_cpu_supports("avx2");
#endif
#endif
#endif
#endif

    if (info.physical_memory_bytes == 0) {
        info.physical_memory_bytes = 8ULL * GB;
    }
    return info;
}

std::size_t chooseDefaultBufferSize(const PlatformInfo& platform) {
    const std::uint64_t one_eighth_memory = platform.physical_memory_bytes / 8ULL;
    std::uint64_t chosen = std::min<std::uint64_t>(kMaxDefaultBufferSize, one_eighth_memory);
    chosen = std::max<std::uint64_t>(chosen, kMinDefaultBufferSize);
    return static_cast<std::size_t>(std::min<std::uint64_t>(chosen, kMaxBufferSize));
}

unsigned int chooseDefaultThreadCount(const PlatformInfo& platform, ThreadPolicy policy) {
    if (platform.apple_silicon && policy == ThreadPolicy::Perf) {
        if (platform.performance_cores > 0) {
            return platform.performance_cores;
        }
        return std::min(platform.hardware_threads, 4U);
    }
    return std::max(1U, platform.hardware_threads);
}

RunMode chooseDefaultMode(const PlatformInfo& platform) {
    return (platform.apple_silicon || platform.x86_avx2) ? RunMode::Peak : RunMode::Standard;
}

bool kernelSupported(const PlatformInfo& platform, KernelKind kernel) {
    switch (kernel) {
        case KernelKind::ScalarAuto:
        case KernelKind::LibcMemset:
        case KernelKind::LibcMemcpy:
            return true;
        case KernelKind::Avx2Read:
        case KernelKind::Avx2StreamStore:
        case KernelKind::Avx2StreamCopy:
#if defined(__AVX2__)
            return platform.x86_avx2;
#else
            (void)platform;
            return false;
#endif
        case KernelKind::NeonPeak:
        case KernelKind::NeonStore:
        case KernelKind::NeonCopy:
            return platform.apple_silicon;
        case KernelKind::MetalRead:
        case KernelKind::MetalWrite:
        case KernelKind::MetalCopy:
#ifdef MEMBENCH_HAS_METAL
            return platform.apple_silicon && metalIsAvailable();
#else
            (void)platform;
            return false;
#endif
    }
    return false;
}

double calibrationOverrideRatio(const PlatformInfo& platform, TestKind kind) {
    if (platform.x86_avx2) {
        return 1.03;
    }

    switch (kind) {
        case TestKind::Read:
            return 1.03;
        case TestKind::Write:
            return 1.05;
        case TestKind::Copy:
            return 1.10;
    }
    return 1.05;
}

std::size_t clampThreadCountForSize(std::size_t size_bytes, unsigned int requested_threads) {
    const std::size_t max_threads_by_size =
        std::max<std::size_t>(1, size_bytes / kCacheLineSize);
    return std::max<std::size_t>(1, std::min<std::size_t>(requested_threads, max_threads_by_size));
}

std::vector<unsigned int> buildThreadCandidates(const PlatformInfo& platform,
                                                const BenchmarkOptions& options) {
    std::vector<unsigned int> candidates;
    if (options.threads_override > 0) {
        candidates.push_back(options.threads_override);
        return candidates;
    }

    const unsigned int perf_threads = chooseDefaultThreadCount(platform, ThreadPolicy::Perf);
    const unsigned int all_threads = chooseDefaultThreadCount(platform, ThreadPolicy::All);

    if (platform.apple_silicon) {
        if (options.thread_policy == ThreadPolicy::Perf) {
            candidates.push_back(perf_threads);
        } else {
            candidates.push_back(perf_threads);
            candidates.push_back(all_threads);
        }
    } else if (options.mode == RunMode::Peak && platform.x86_avx2) {
        for (unsigned int value = 1; value < all_threads; value *= 2) {
            candidates.push_back(value);
        }
        candidates.push_back(std::max(1U, all_threads / 2U));
        candidates.push_back(all_threads);
    } else {
        candidates.push_back(chooseDefaultThreadCount(platform, options.thread_policy));
    }

    candidates.erase(std::remove(candidates.begin(), candidates.end(), 0), candidates.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    if (candidates.empty()) {
        candidates.push_back(1);
    }
    return candidates;
}

std::vector<KernelKind> buildKernelCandidates(const PlatformInfo& platform,
                                              TestKind kind,
                                              Backend backend) {
    std::vector<KernelKind> kernels;
    const bool want_cpu   = (backend == Backend::Cpu || backend == Backend::Auto);
    const bool want_metal = (backend == Backend::Metal || backend == Backend::Auto);

    switch (kind) {
        case TestKind::Read:
            if (want_cpu) {
                kernels.push_back(KernelKind::ScalarAuto);
                if (kernelSupported(platform, KernelKind::Avx2Read)) {
                    kernels.push_back(KernelKind::Avx2Read);
                }
                if (platform.apple_silicon) {
                    kernels.push_back(KernelKind::NeonPeak);
                }
            }
            if (want_metal && kernelSupported(platform, KernelKind::MetalRead)) {
                kernels.push_back(KernelKind::MetalRead);
            }
            break;
        case TestKind::Write:
            if (want_cpu) {
                kernels.push_back(KernelKind::LibcMemset);
                if (kernelSupported(platform, KernelKind::Avx2StreamStore)) {
                    kernels.push_back(KernelKind::Avx2StreamStore);
                }
                if (platform.apple_silicon) {
                    kernels.push_back(KernelKind::NeonStore);
                }
            }
            if (want_metal && kernelSupported(platform, KernelKind::MetalWrite)) {
                kernels.push_back(KernelKind::MetalWrite);
            }
            break;
        case TestKind::Copy:
            if (want_cpu) {
                kernels.push_back(KernelKind::LibcMemcpy);
                if (kernelSupported(platform, KernelKind::Avx2StreamCopy)) {
                    kernels.push_back(KernelKind::Avx2StreamCopy);
                }
                if (platform.apple_silicon) {
                    kernels.push_back(KernelKind::NeonCopy);
                }
            }
            if (want_metal && kernelSupported(platform, KernelKind::MetalCopy)) {
                kernels.push_back(KernelKind::MetalCopy);
            }
            break;
    }
    return kernels;
}

bool isMetalKernel(KernelKind kernel) {
    return kernel == KernelKind::MetalRead ||
           kernel == KernelKind::MetalWrite ||
           kernel == KernelKind::MetalCopy;
}

KernelKind chooseHeuristicKernel(const PlatformInfo& platform,
                                 const BenchmarkOptions& options,
                                 TestKind kind) {
    if (options.mode == RunMode::Peak && platform.apple_silicon) {
        if (kind == TestKind::Read) {
            return KernelKind::NeonPeak;
        }
    }

    switch (kind) {
        case TestKind::Read:
            return KernelKind::ScalarAuto;
        case TestKind::Write:
            if (options.mode == RunMode::Peak &&
                kernelSupported(platform, KernelKind::Avx2StreamStore)) {
                return KernelKind::Avx2StreamStore;
            }
            return KernelKind::LibcMemset;
        case TestKind::Copy:
            return KernelKind::LibcMemcpy;
    }
    return KernelKind::ScalarAuto;
}

void printUsage(const char* program_name, const PlatformInfo& platform) {
    const std::size_t default_size_mb = chooseDefaultBufferSize(platform) / MB;
    const std::string default_mode = runModeToString(chooseDefaultMode(platform));
    std::cout << "Usage: " << program_name << " [size_mb] [options]\n\n"
              << "Options:\n"
              << "  --size-mb <n>         Buffer size in MiB (default: " << default_size_mb
              << ")\n"
              << "  --threads <n>         Override worker thread count\n"
              << "  --tests <list>        Comma-separated tests: read,write,copy\n"
              << "  --iterations <n>      Measured iterations per test (default: "
              << kDefaultMeasuredIterations << ")\n"
              << "  --warmup <n>          Warmup iterations per test (default: "
              << kDefaultWarmupIterations << ")\n"
              << "  --thread-policy <p>   perf or all\n"
              << "  --mode <m>            standard or peak (default: " << default_mode << ")\n"
              << "  --backend <b>         cpu, metal, or auto (default: "
              << (platform.apple_silicon ? "auto" : "cpu") << ")\n"
              << "  --no-calibrate        Disable peak-mode kernel/thread calibration\n"
              << "  --no-qos              Disable macOS QoS hinting\n"
              << "  --help                Show this message\n\n"
              << "Examples:\n"
              << "  " << program_name << " 1024\n"
              << "  " << program_name << " --mode standard --thread-policy perf\n"
              << "  " << program_name << " --mode peak --size-mb 1024 --tests read,copy\n"
              << "  " << program_name
              << " --threads 4 --warmup 2 --iterations 7 --thread-policy all\n";
}

void printSystemInfo(const PlatformInfo& platform) {
    std::cout << "=== System Information ===\n";
#ifdef _WIN32
    std::cout << "Operating System: Windows\n";
#elif defined(__APPLE__)
    std::cout << "Operating System: macOS\n";
#elif defined(__linux__)
    std::cout << "Operating System: Linux\n";
#else
    std::cout << "Operating System: Unknown\n";
#endif
    std::cout << "Page size: " << platform.page_size << " bytes\n";
    std::cout << "Physical memory: " << formatBytes(platform.physical_memory_bytes) << '\n';
    std::cout << "Hardware threads: " << platform.hardware_threads << '\n';
    if (platform.apple_silicon) {
        if (platform.performance_cores > 0) {
            std::cout << "Performance cores: " << platform.performance_cores << '\n';
        } else {
            std::cout << "Performance cores: unavailable (falling back to conservative default)\n";
        }
#ifdef MEMBENCH_HAS_METAL
        if (metalIsAvailable()) {
            std::cout << "Metal GPU: " << metalDeviceName() << '\n';
        } else {
            std::cout << "Metal GPU: unavailable\n";
        }
#endif
    }
    if (platform.x86_avx2) {
        std::cout << "x86 AVX2: available\n";
    }
    std::cout << '\n';
}

class AlignedBuffer {
public:
    AlignedBuffer() = default;

    AlignedBuffer(std::size_t size, std::size_t alignment)
        : size_(size), alignment_(alignment) {
        if (size_ == 0) {
            throw std::runtime_error("buffer size must be greater than zero");
        }

#ifdef _WIN32
        data_ = static_cast<std::uint8_t*>(_aligned_malloc(size_, alignment_));
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
#else
        void* raw = nullptr;
        if (posix_memalign(&raw, alignment_, size_) != 0 || raw == nullptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<std::uint8_t*>(raw);
#endif
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.alignment_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        data_ = other.data_;
        size_ = other.size_;
        alignment_ = other.alignment_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.alignment_ = 0;
        return *this;
    }

    ~AlignedBuffer() {
        reset();
    }

    std::uint8_t* data() { return data_; }
    const std::uint8_t* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    void reset() {
        if (data_ == nullptr) {
            return;
        }
#ifdef _WIN32
        _aligned_free(data_);
#else
        free(data_);
#endif
        data_ = nullptr;
        size_ = 0;
        alignment_ = 0;
    }

    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t alignment_ = 0;
};

template <typename T>
inline void doNotOptimize(const T& value) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#elif defined(_WIN32)
    _ReadWriteBarrier();
    volatile T sink = value;
    _ReadWriteBarrier();
    (void)sink;
#else
    (void)value;
#endif
}

std::uint64_t monotonicNowNs() {
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
#endif
}

class BenchmarkRunner {
public:
    BenchmarkRunner(const PlatformInfo& platform,
                    RunMode mode,
                    bool use_qos,
                    std::uint8_t* buffer_a,
                    std::uint8_t* buffer_b,
                    std::size_t size_bytes,
                    unsigned int requested_threads)
        : platform_(platform),
          mode_(mode),
          use_qos_(use_qos),
          buffer_a_(buffer_a),
          buffer_b_(buffer_b),
          size_bytes_(size_bytes),
          thread_count_(resolveThreadCount(requested_threads)) {
        buildSlices();
        startWorkers();
    }

    ~BenchmarkRunner() {
        stopWorkers();
    }

    BenchmarkRunner(const BenchmarkRunner&) = delete;
    BenchmarkRunner& operator=(const BenchmarkRunner&) = delete;

    unsigned int threadCount() const {
        return thread_count_;
    }

    TestResult run(TestKind kind,
                   KernelKind kernel,
                   std::size_t warmup_iterations,
                   std::size_t measured_iterations,
                   std::size_t passes_per_iteration = 1) {
        std::vector<double> bandwidth_samples;
        std::vector<double> elapsed_samples;
        std::uint8_t write_pattern = 0x55U;

        for (std::size_t warmup_index = 0; warmup_index < warmup_iterations; ++warmup_index) {
            WorkerCommand command;
            command.kind = kind;
            command.kernel = kernel;
            command.passes = passes_per_iteration;
            command.write_pattern = write_pattern;
            (void)runTimedCommand(command);
            if (kind == TestKind::Write) {
                write_pattern = togglePattern(write_pattern);
            }
        }

        for (std::size_t iteration = 0; iteration < measured_iterations; ++iteration) {
            WorkerCommand command;
            command.kind = kind;
            command.kernel = kernel;
            command.passes = passes_per_iteration;
            command.write_pattern = write_pattern;

            const TimerResult timer = runTimedCommand(command);
            const double seconds = static_cast<double>(timer.elapsed_ns) / 1'000'000'000.0;
            const double bandwidth_mb_per_sec =
                ((size_bytes_ * passes_per_iteration) / static_cast<double>(MB)) / seconds;
            bandwidth_samples.push_back(bandwidth_mb_per_sec);
            elapsed_samples.push_back(static_cast<double>(timer.elapsed_ns) / 1'000'000.0);

            if (kind == TestKind::Write) {
                write_pattern = togglePattern(write_pattern);
            }
        }

        TestResult result;
        result.bandwidth_mb_per_sec = calculateStatistics(bandwidth_samples);
        result.elapsed_ms = calculateStatistics(elapsed_samples);
        result.logical_bytes_per_iteration = size_bytes_ * passes_per_iteration;
        return result;
    }

private:
    struct Slice {
        std::size_t offset = 0;
        std::size_t size = 0;
    };

    struct WorkerCommand {
        TestKind kind = TestKind::Read;
        KernelKind kernel = KernelKind::ScalarAuto;
        std::size_t passes = 1;
        std::uint8_t write_pattern = 0xAA;
    };

    struct TimerResult {
        std::uint64_t elapsed_ns = 0;
    };

    static std::uint8_t togglePattern(std::uint8_t pattern) {
        return pattern == 0x55U ? 0xAAU : 0x55U;
    }

    unsigned int resolveThreadCount(unsigned int requested_threads) const {
        const std::size_t clamped = clampThreadCountForSize(size_bytes_, requested_threads);
        return static_cast<unsigned int>(clamped);
    }

    static std::size_t alignDown(std::size_t value, std::size_t alignment) {
        return value - (value % alignment);
    }

    void buildSlices() {
        slices_.clear();
        slices_.reserve(thread_count_);

        const std::size_t aligned_chunk = alignDown(size_bytes_ / thread_count_, kCacheLineSize);
        std::size_t offset = 0;
        for (unsigned int index = 0; index < thread_count_; ++index) {
            Slice slice;
            slice.offset = offset;
            if (index == thread_count_ - 1) {
                slice.size = size_bytes_ - offset;
            } else {
                slice.size = aligned_chunk;
            }
            slices_.push_back(slice);
            offset += slice.size;
        }
    }

    void startWorkers() {
        workers_.reserve(thread_count_);
        for (unsigned int index = 0; index < thread_count_; ++index) {
            workers_.emplace_back([this, index]() { workerLoop(index, slices_[index]); });
        }
    }

    void stopWorkers() {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            stop_workers_ = true;
            ++command_generation_;
            ++release_generation_;
        }
        worker_command_cv_.notify_all();
        worker_release_cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void maybeApplyThreadPolicy() const {
#ifdef __APPLE__
        if (!use_qos_) {
            return;
        }
        const qos_class_t qos =
            mode_ == RunMode::Peak ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_USER_INITIATED;
        pthread_set_qos_class_self_np(qos, 0);
#else
        (void)mode_;
#endif
    }

    void workerLoop(unsigned int worker_index, const Slice& slice) {
        maybeApplyThreadPolicy();

        std::size_t observed_command_generation = 0;
        std::size_t observed_release_generation = 0;

        while (true) {
            WorkerCommand command;
            {
                std::unique_lock<std::mutex> lock(worker_mutex_);
                worker_command_cv_.wait(lock, [this, &observed_command_generation]() {
                    return stop_workers_ || command_generation_ != observed_command_generation;
                });
                if (stop_workers_) {
                    return;
                }

                observed_command_generation = command_generation_;
                command = current_command_;
                ++ready_workers_;
                if (ready_workers_ == thread_count_) {
                    worker_ready_cv_.notify_one();
                }

                worker_release_cv_.wait(lock, [this, &observed_release_generation]() {
                    return stop_workers_ || release_generation_ != observed_release_generation;
                });
                if (stop_workers_) {
                    return;
                }
                observed_release_generation = release_generation_;
            }

            executeCommand(command, slice, worker_index);

            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                ++completed_workers_;
                if (completed_workers_ == thread_count_) {
                    worker_done_cv_.notify_one();
                }
            }
        }
    }

    void executeCommand(const WorkerCommand& command, const Slice& slice, unsigned int worker_index) {
        switch (command.kind) {
            case TestKind::Read:
                if (command.kernel == KernelKind::Avx2Read && kernelSupported(platform_, command.kernel)) {
                    runReadAvx2(slice, command.passes, worker_index);
                } else if (command.kernel == KernelKind::NeonPeak && kernelSupported(platform_, command.kernel)) {
                    runReadNeonPeak(slice, command.passes, worker_index);
                } else {
                    runReadScalar(slice, command.passes, worker_index);
                }
                break;
            case TestKind::Write:
                if (command.kernel == KernelKind::Avx2StreamStore && kernelSupported(platform_, command.kernel)) {
                    runWriteAvx2Stream(slice, command.passes, command.write_pattern);
                } else if (command.kernel == KernelKind::NeonStore && kernelSupported(platform_, command.kernel)) {
                    runWriteNeonStore(slice, command.passes, command.write_pattern);
                } else {
                    runWriteMemset(slice, command.passes, command.write_pattern);
                }
                break;
            case TestKind::Copy:
                if (command.kernel == KernelKind::Avx2StreamCopy && kernelSupported(platform_, command.kernel)) {
                    runCopyAvx2Stream(slice, command.passes);
                } else if (command.kernel == KernelKind::NeonCopy && kernelSupported(platform_, command.kernel)) {
                    runCopyNeon(slice, command.passes);
                } else {
                    runCopyMemcpy(slice, command.passes);
                }
                break;
        }
    }

    void runReadScalar(const Slice& slice, std::size_t passes, unsigned int worker_index) {
        const auto* bytes = buffer_a_ + slice.offset;
        const auto* words = reinterpret_cast<const std::uint64_t*>(bytes);
        const std::size_t word_count = slice.size / sizeof(std::uint64_t);

        std::uint64_t accumulator0 = splitMix64(worker_index + 1U);
        std::uint64_t accumulator1 = splitMix64(worker_index + 11U);
        std::uint64_t accumulator2 = splitMix64(worker_index + 21U);
        std::uint64_t accumulator3 = splitMix64(worker_index + 31U);

        for (std::size_t pass = 0; pass < passes; ++pass) {
            std::size_t index = 0;
            for (; index + 4 <= word_count; index += 4) {
                accumulator0 += words[index];
                accumulator1 += words[index + 1];
                accumulator2 += words[index + 2];
                accumulator3 += words[index + 3];
            }
            for (; index < word_count; ++index) {
                accumulator0 += words[index];
            }
        }

        const std::size_t tail_offset = word_count * sizeof(std::uint64_t);
        std::uint64_t tail = 0;
        for (std::size_t index = tail_offset; index < slice.size; ++index) {
            tail += bytes[index];
        }

        const std::uint64_t local_sink =
            accumulator0 ^ accumulator1 ^ accumulator2 ^ accumulator3 ^ tail;
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
    }

    void runReadAvx2(const Slice& slice, std::size_t passes, unsigned int worker_index) {
#if defined(__AVX2__)
        const auto* bytes = buffer_a_ + slice.offset;
        __m256i acc0 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 1U)));
        __m256i acc1 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 11U)));
        __m256i acc2 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 21U)));
        __m256i acc3 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 31U)));
        __m256i acc4 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 41U)));
        __m256i acc5 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 51U)));
        __m256i acc6 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 61U)));
        __m256i acc7 = _mm256_set1_epi64x(static_cast<long long>(splitMix64(worker_index + 71U)));

        std::size_t index = 0;
        for (std::size_t pass = 0; pass < passes; ++pass) {
            index = 0;
            for (; index + 256 <= slice.size; index += 256) {
                _mm_prefetch(reinterpret_cast<const char*>(bytes + index + 1024), _MM_HINT_NTA);
                acc0 = _mm256_add_epi64(
                    acc0, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 0)));
                acc1 = _mm256_add_epi64(
                    acc1, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 32)));
                acc2 = _mm256_add_epi64(
                    acc2, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 64)));
                acc3 = _mm256_add_epi64(
                    acc3, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 96)));
                acc4 = _mm256_add_epi64(
                    acc4, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 128)));
                acc5 = _mm256_add_epi64(
                    acc5, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 160)));
                acc6 = _mm256_add_epi64(
                    acc6, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 192)));
                acc7 = _mm256_add_epi64(
                    acc7, _mm256_load_si256(reinterpret_cast<const __m256i*>(bytes + index + 224)));
            }
        }

        std::uint64_t tail = 0;
        for (; index < slice.size; ++index) {
            tail += bytes[index];
        }

        acc0 = _mm256_xor_si256(acc0, acc1);
        acc2 = _mm256_xor_si256(acc2, acc3);
        acc4 = _mm256_xor_si256(acc4, acc5);
        acc6 = _mm256_xor_si256(acc6, acc7);
        acc0 = _mm256_xor_si256(acc0, acc2);
        acc4 = _mm256_xor_si256(acc4, acc6);
        acc0 = _mm256_xor_si256(acc0, acc4);

        alignas(32) std::uint64_t lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc0);
        const std::uint64_t local_sink = lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3] ^ tail;
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
#else
        runReadScalar(slice, passes, worker_index);
#endif
    }

    void runReadNeonPeak(const Slice& slice, std::size_t passes, unsigned int worker_index) {
#if defined(__aarch64__) || defined(__arm64__) || defined(__ARM_NEON)
        const auto* bytes = buffer_a_ + slice.offset;
        std::size_t index = 0;
        uint64x2_t acc0 = vdupq_n_u64(splitMix64(worker_index + 1U));
        uint64x2_t acc1 = vdupq_n_u64(splitMix64(worker_index + 11U));
        uint64x2_t acc2 = vdupq_n_u64(splitMix64(worker_index + 21U));
        uint64x2_t acc3 = vdupq_n_u64(splitMix64(worker_index + 31U));
        uint64x2_t acc4 = vdupq_n_u64(splitMix64(worker_index + 41U));
        uint64x2_t acc5 = vdupq_n_u64(splitMix64(worker_index + 51U));
        uint64x2_t acc6 = vdupq_n_u64(splitMix64(worker_index + 61U));
        uint64x2_t acc7 = vdupq_n_u64(splitMix64(worker_index + 71U));

        for (std::size_t pass = 0; pass < passes; ++pass) {
            index = 0;
            for (; index + 128 <= slice.size; index += 128) {
                __builtin_prefetch(bytes + index + 512, 0, 0);
                acc0 = vaddq_u64(acc0, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 0)));
                acc1 = vaddq_u64(acc1, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 16)));
                acc2 = vaddq_u64(acc2, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 32)));
                acc3 = vaddq_u64(acc3, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 48)));
                acc4 = vaddq_u64(acc4, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 64)));
                acc5 = vaddq_u64(acc5, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 80)));
                acc6 = vaddq_u64(acc6, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 96)));
                acc7 = vaddq_u64(acc7, vreinterpretq_u64_u8(vld1q_u8(bytes + index + 112)));
            }

            std::uint64_t tail = 0;
            for (; index < slice.size; ++index) {
                tail += bytes[index];
            }
            sink_.fetch_xor(tail, std::memory_order_relaxed);
        }

        const uint64x2_t pair01 = veorq_u64(acc0, acc1);
        const uint64x2_t pair23 = veorq_u64(acc2, acc3);
        const uint64x2_t pair45 = veorq_u64(acc4, acc5);
        const uint64x2_t pair67 = veorq_u64(acc6, acc7);
        const uint64x2_t pair0123 = veorq_u64(pair01, pair23);
        const uint64x2_t pair4567 = veorq_u64(pair45, pair67);
        const uint64x2_t final_acc = veorq_u64(pair0123, pair4567);
        const std::uint64_t local_sink = vgetq_lane_u64(final_acc, 0) ^ vgetq_lane_u64(final_acc, 1);
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
#else
        runReadScalar(slice, passes, worker_index);
#endif
    }

    void runWriteMemset(const Slice& slice, std::size_t passes, std::uint8_t base_pattern) {
        auto* bytes = buffer_a_ + slice.offset;
        std::uint8_t pattern = base_pattern;

        for (std::size_t pass = 0; pass < passes; ++pass) {
            std::memset(bytes, pattern, slice.size);
            pattern = togglePattern(pattern);
        }

        const std::uint64_t local_sink =
            static_cast<std::uint64_t>(bytes[0]) ^
            (static_cast<std::uint64_t>(bytes[slice.size - 1]) << 8U);
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
    }

    void runWriteAvx2Stream(const Slice& slice,
                            std::size_t passes,
                            std::uint8_t base_pattern) {
#if defined(__AVX2__)
        auto* bytes = buffer_a_ + slice.offset;
        std::uint8_t pattern = base_pattern;

        for (std::size_t pass = 0; pass < passes; ++pass) {
            const __m256i fill = _mm256_set1_epi8(static_cast<char>(pattern));
            std::size_t index = 0;
            for (; index + 256 <= slice.size; index += 256) {
                auto* ptr = reinterpret_cast<__m256i*>(bytes + index);
                _mm256_stream_si256(ptr + 0, fill);
                _mm256_stream_si256(ptr + 1, fill);
                _mm256_stream_si256(ptr + 2, fill);
                _mm256_stream_si256(ptr + 3, fill);
                _mm256_stream_si256(ptr + 4, fill);
                _mm256_stream_si256(ptr + 5, fill);
                _mm256_stream_si256(ptr + 6, fill);
                _mm256_stream_si256(ptr + 7, fill);
            }
            if (index < slice.size) {
                std::memset(bytes + index, pattern, slice.size - index);
            }
            _mm_sfence();
            pattern = togglePattern(pattern);
        }

        const std::uint64_t local_sink =
            static_cast<std::uint64_t>(bytes[0]) ^
            (static_cast<std::uint64_t>(bytes[slice.size - 1]) << 8U);
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
#else
        runWriteMemset(slice, passes, base_pattern);
#endif
    }

    void runWriteNeonStore(const Slice& slice, std::size_t passes, std::uint8_t base_pattern) {
#if defined(__aarch64__) || defined(__arm64__) || defined(__ARM_NEON)
        auto* bytes = buffer_a_ + slice.offset;
        std::uint8_t pattern = base_pattern;

        for (std::size_t pass = 0; pass < passes; ++pass) {
            const uint8x16_t fill = vdupq_n_u8(pattern);
            std::size_t index = 0;
            for (; index + 64 <= slice.size; index += 64) {
                vst1q_u8(bytes + index + 0, fill);
                vst1q_u8(bytes + index + 16, fill);
                vst1q_u8(bytes + index + 32, fill);
                vst1q_u8(bytes + index + 48, fill);
            }
            if (index < slice.size) {
                std::memset(bytes + index, pattern, slice.size - index);
            }
            pattern = togglePattern(pattern);
        }

        const std::uint64_t local_sink =
            static_cast<std::uint64_t>(bytes[0]) ^
            (static_cast<std::uint64_t>(bytes[slice.size - 1]) << 8U);
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
#else
        runWriteMemset(slice, passes, base_pattern);
#endif
    }

    void runCopyMemcpy(const Slice& slice, std::size_t passes) {
        auto* a = buffer_a_ + slice.offset;
        auto* b = buffer_b_ + slice.offset;

        for (std::size_t pass = 0; pass < passes; ++pass) {
            if ((pass % 2U) == 0U) {
                std::memcpy(b, a, slice.size);
            } else {
                std::memcpy(a, b, slice.size);
            }
        }

        const std::uint64_t local_sink =
            static_cast<std::uint64_t>(a[0]) ^
            (static_cast<std::uint64_t>(b[slice.size - 1]) << 8U);
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
    }

    void runCopyAvx2Stream(const Slice& slice, std::size_t passes) {
#if defined(__AVX2__)
        auto copy_once = [&](std::uint8_t* dst, const std::uint8_t* src) {
            std::size_t index = 0;
            for (; index + 256 <= slice.size; index += 256) {
                _mm_prefetch(reinterpret_cast<const char*>(src + index + 1024), _MM_HINT_NTA);
                const __m256i v0 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 0));
                const __m256i v1 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 32));
                const __m256i v2 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 64));
                const __m256i v3 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 96));
                const __m256i v4 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 128));
                const __m256i v5 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 160));
                const __m256i v6 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 192));
                const __m256i v7 =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(src + index + 224));
                auto* ptr = reinterpret_cast<__m256i*>(dst + index);
                _mm256_stream_si256(ptr + 0, v0);
                _mm256_stream_si256(ptr + 1, v1);
                _mm256_stream_si256(ptr + 2, v2);
                _mm256_stream_si256(ptr + 3, v3);
                _mm256_stream_si256(ptr + 4, v4);
                _mm256_stream_si256(ptr + 5, v5);
                _mm256_stream_si256(ptr + 6, v6);
                _mm256_stream_si256(ptr + 7, v7);
            }
            if (index < slice.size) {
                std::memcpy(dst + index, src + index, slice.size - index);
            }
            _mm_sfence();
        };

        auto* a = buffer_a_ + slice.offset;
        auto* b = buffer_b_ + slice.offset;
        for (std::size_t pass = 0; pass < passes; ++pass) {
            if ((pass % 2U) == 0U) {
                copy_once(b, a);
            } else {
                copy_once(a, b);
            }
        }

        const std::uint64_t local_sink =
            static_cast<std::uint64_t>(a[0]) ^
            (static_cast<std::uint64_t>(b[slice.size - 1]) << 8U);
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
#else
        runCopyMemcpy(slice, passes);
#endif
    }

    void runCopyNeon(const Slice& slice, std::size_t passes) {
#if defined(__aarch64__) || defined(__arm64__) || defined(__ARM_NEON)
        auto copy_once = [&](std::uint8_t* dst, const std::uint8_t* src) {
            std::size_t index = 0;
            for (; index + 64 <= slice.size; index += 64) {
                __builtin_prefetch(src + index + 512, 0, 0);
                const uint8x16_t v0 = vld1q_u8(src + index + 0);
                const uint8x16_t v1 = vld1q_u8(src + index + 16);
                const uint8x16_t v2 = vld1q_u8(src + index + 32);
                const uint8x16_t v3 = vld1q_u8(src + index + 48);
                vst1q_u8(dst + index + 0, v0);
                vst1q_u8(dst + index + 16, v1);
                vst1q_u8(dst + index + 32, v2);
                vst1q_u8(dst + index + 48, v3);
            }
            if (index < slice.size) {
                std::memcpy(dst + index, src + index, slice.size - index);
            }
        };

        auto* a = buffer_a_ + slice.offset;
        auto* b = buffer_b_ + slice.offset;
        for (std::size_t pass = 0; pass < passes; ++pass) {
            if ((pass % 2U) == 0U) {
                copy_once(b, a);
            } else {
                copy_once(a, b);
            }
        }

        const std::uint64_t local_sink =
            static_cast<std::uint64_t>(a[0]) ^
            (static_cast<std::uint64_t>(b[slice.size - 1]) << 8U);
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
#else
        runCopyMemcpy(slice, passes);
#endif
    }

    TimerResult runTimedCommand(const WorkerCommand& command) {
        std::unique_lock<std::mutex> lock(worker_mutex_);
        current_command_ = command;
        ready_workers_ = 0;
        completed_workers_ = 0;
        ++command_generation_;
        worker_command_cv_.notify_all();

        worker_ready_cv_.wait(lock, [this]() { return ready_workers_ == thread_count_; });
        const std::uint64_t start_ns = monotonicNowNs();
        ++release_generation_;
        worker_release_cv_.notify_all();
        lock.unlock();

        std::unique_lock<std::mutex> done_lock(worker_mutex_);
        worker_done_cv_.wait(done_lock, [this]() { return completed_workers_ == thread_count_; });
        const std::uint64_t end_ns = monotonicNowNs();
        return {end_ns - start_ns};
    }

    const PlatformInfo& platform_;
    RunMode mode_;
    bool use_qos_ = false;
    std::uint8_t* buffer_a_ = nullptr;
    std::uint8_t* buffer_b_ = nullptr;
    std::size_t size_bytes_ = 0;
    unsigned int thread_count_ = 1;
    std::vector<Slice> slices_;
    std::vector<std::thread> workers_;

    std::atomic<std::uint64_t> sink_{0};

    std::mutex worker_mutex_;
    std::condition_variable worker_command_cv_;
    std::condition_variable worker_ready_cv_;
    std::condition_variable worker_release_cv_;
    std::condition_variable worker_done_cv_;
    WorkerCommand current_command_;
    std::size_t command_generation_ = 0;
    std::size_t release_generation_ = 0;
    std::size_t ready_workers_ = 0;
    std::size_t completed_workers_ = 0;
    bool stop_workers_ = false;
};

class MemoryBenchmark {
public:
    MemoryBenchmark(const PlatformInfo& platform, const BenchmarkOptions& options)
        : platform_(platform),
          options_(options),
          alignment_(std::max(platform.page_size, kCacheLineSize)),
          buffer_a_(options.size_bytes, alignment_),
          buffer_b_(options.size_bytes, alignment_) {
        initializeBuffers();
    }

    void printConfiguration() const {
        std::cout << "=== Benchmark Configuration ===\n";
        std::cout << "Size: " << options_.size_bytes / MB << " MiB\n";
        std::cout << "Alignment: " << alignment_ << " bytes\n";
        std::cout << "Mode: " << runModeToString(options_.mode) << '\n';
        std::cout << "Backend: " << backendToString(options_.backend) << '\n';
        std::cout << "Calibration: " << (isCalibrationEnabled() ? "enabled" : "disabled") << '\n';
        std::cout << "Warmup iterations: " << options_.warmup_iterations << '\n';
        std::cout << "Measured iterations: " << options_.measured_iterations << '\n';
        std::cout << "Thread policy: " << threadPolicyToString(options_.thread_policy) << '\n';
        if (options_.threads_override > 0) {
            std::cout << "Thread override: " << options_.threads_override << '\n';
        }
        std::cout << "macOS QoS hint: " << (options_.use_qos ? "enabled" : "disabled") << '\n';
        std::cout << "Tests: ";
        for (std::size_t index = 0; index < options_.tests.size(); ++index) {
            if (index > 0) {
                std::cout << ',';
            }
            std::cout << testKindToCliName(options_.tests[index]);
        }
        std::cout << "\n\n";
    }

    struct SummaryEntry {
        TestKind kind;
        ExecutionPlan plan;
        unsigned int actual_threads;
        double avg_bandwidth_gb_per_sec;
        double avg_traffic_gb_per_sec;  // copy only: 2x logical
    };

    void run() {
        std::vector<SummaryEntry> summary;

        for (TestKind kind : options_.tests) {
            const ExecutionPlan plan = selectExecutionPlan(kind);

            if (isMetalKernel(plan.kernel)) {
#ifdef MEMBENCH_HAS_METAL
                TestResult result = runMetalTest(kind, plan);
                summary.push_back(makeSummaryEntry(kind, plan, 0, result));
#endif
            } else {
                BenchmarkRunner runner(platform_,
                                       options_.mode,
                                       options_.use_qos,
                                       buffer_a_.data(),
                                       buffer_b_.data(),
                                       options_.size_bytes,
                                       plan.selected_threads);
                TestResult result = runner.run(
                    kind, plan.kernel, options_.warmup_iterations, options_.measured_iterations);
                printTestResult(kind, plan, runner.threadCount(), result);
                summary.push_back(makeSummaryEntry(kind, plan, runner.threadCount(), result));
            }
        }

        if (summary.size() > 1) {
            printSummary(summary);
        }
    }

private:
    bool isCalibrationEnabled() const {
        return options_.mode == RunMode::Peak && options_.calibrate &&
               (platform_.apple_silicon || platform_.x86_avx2);
    }

    void initializeBuffers() {
        std::cout << "Allocating " << options_.size_bytes / MB << " MiB per buffer..." << std::endl;
        auto* words_a = reinterpret_cast<std::uint64_t*>(buffer_a_.data());
        auto* words_b = reinterpret_cast<std::uint64_t*>(buffer_b_.data());
        const std::size_t word_count = options_.size_bytes / sizeof(std::uint64_t);

        for (std::size_t index = 0; index < word_count; ++index) {
            words_a[index] = splitMix64(0x123456789ABCDEF0ULL + static_cast<std::uint64_t>(index));
            words_b[index] = splitMix64(0x0FEDCBA987654321ULL + static_cast<std::uint64_t>(index));
        }

        const std::size_t tail_offset = word_count * sizeof(std::uint64_t);
        for (std::size_t index = tail_offset; index < options_.size_bytes; ++index) {
            buffer_a_.data()[index] = static_cast<std::uint8_t>(index & 0xFFU);
            buffer_b_.data()[index] = static_cast<std::uint8_t>((255U - index) & 0xFFU);
        }

        doNotOptimize(words_a[0]);
        doNotOptimize(words_b[0]);
        std::cout << "Buffers initialized with deterministic non-zero data.\n\n";
    }

    ExecutionPlan selectExecutionPlan(TestKind kind) {
        if (!isCalibrationEnabled()) {
            return buildHeuristicPlan(kind);
        }

        const std::size_t calibration_size = std::min(options_.size_bytes, kCalibrationBufferSize);
        const std::vector<unsigned int> thread_candidates =
            buildThreadCandidates(platform_, options_);
        const std::vector<KernelKind> kernel_candidates =
            buildKernelCandidates(platform_, kind, options_.backend);

        const ExecutionPlan heuristic_plan = buildHeuristicPlan(kind);
        CalibrationCandidate best_candidate;

        if (!isMetalKernel(heuristic_plan.kernel)) {
            BenchmarkRunner heuristic_runner(platform_,
                                             options_.mode,
                                             options_.use_qos,
                                             buffer_a_.data(),
                                             buffer_b_.data(),
                                             calibration_size,
                                             heuristic_plan.selected_threads);
            const TestResult heuristic_result =
                heuristic_runner.run(kind,
                                     heuristic_plan.kernel,
                                     kCalibrationWarmupIterations,
                                     kCalibrationMeasuredIterations,
                                     kCalibrationPassesPerIteration);
            best_candidate.kernel = heuristic_plan.kernel;
            best_candidate.requested_threads = heuristic_plan.selected_threads;
            best_candidate.actual_threads = heuristic_runner.threadCount();
            best_candidate.score_mb_per_sec = heuristic_result.bandwidth_mb_per_sec.average;
        }

        std::size_t total_candidates = 0;
        for (KernelKind kernel : kernel_candidates) {
            if (!kernelSupported(platform_, kernel)) {
                continue;
            }
            total_candidates += isMetalKernel(kernel) ? 1 : thread_candidates.size();
        }

        std::size_t completed_candidates = 0;
        if (total_candidates > 0) {
            std::cout << "Calibrating " << testKindToTitle(kind) << " ("
                      << total_candidates << " candidates)...\n";
        }

        const double override_ratio = calibrationOverrideRatio(platform_, kind);
        for (KernelKind kernel : kernel_candidates) {
            if (!kernelSupported(platform_, kernel)) {
                continue;
            }

            if (isMetalKernel(kernel)) {
#ifdef MEMBENCH_HAS_METAL
                ++completed_candidates;
                std::cout << "\r  candidate " << completed_candidates << '/'
                          << total_candidates << ": " << kernelToString(kernel) << std::flush;
                MetalTestKind mk = metalTestKindFor(kind);
                MetalIterationResult mr = metalRunBandwidthTest(
                    mk, calibration_size,
                    kCalibrationWarmupIterations,
                    kCalibrationMeasuredIterations);
                double score = 0.0;
                if (!mr.bandwidth_samples.empty()) {
                    for (double s : mr.bandwidth_samples) score += s;
                    score /= static_cast<double>(mr.bandwidth_samples.size());
                }
                if (score > best_candidate.score_mb_per_sec * override_ratio) {
                    best_candidate.kernel = kernel;
                    best_candidate.requested_threads = 0;
                    best_candidate.actual_threads = 0;
                    best_candidate.score_mb_per_sec = score;
                }
#endif
                continue;
            }

            for (unsigned int requested_threads : thread_candidates) {
                BenchmarkRunner runner(platform_,
                                       options_.mode,
                                       options_.use_qos,
                                       buffer_a_.data(),
                                       buffer_b_.data(),
                                       calibration_size,
                                       requested_threads);
                ++completed_candidates;
                std::cout << "\r  candidate " << completed_candidates << '/'
                          << total_candidates << ": " << kernelToString(kernel)
                          << ", " << runner.threadCount() << " threads" << std::flush;
                const TestResult result = runner.run(kind,
                                                     kernel,
                                                     kCalibrationWarmupIterations,
                                                     kCalibrationMeasuredIterations,
                                                     kCalibrationPassesPerIteration);
                const double score = result.bandwidth_mb_per_sec.average;
                if (kernel == best_candidate.kernel &&
                    runner.threadCount() == best_candidate.actual_threads) {
                    continue;
                }
                if (score > best_candidate.score_mb_per_sec * override_ratio) {
                    best_candidate.kernel = kernel;
                    best_candidate.requested_threads = requested_threads;
                    best_candidate.actual_threads = runner.threadCount();
                    best_candidate.score_mb_per_sec = score;
                }
            }
        }
        if (total_candidates > 0) {
            std::cout << "\r  selected " << kernelToString(best_candidate.kernel);
            if (!isMetalKernel(best_candidate.kernel)) {
                std::cout << ", " << best_candidate.actual_threads << " threads";
            }
            std::cout << " at " << std::fixed << std::setprecision(2)
                      << (best_candidate.score_mb_per_sec / 1024.0) << " GB/s"
                      << "                      \n";
        }

        ExecutionPlan plan;
        plan.kernel = best_candidate.kernel;
        if (isMetalKernel(best_candidate.kernel)) {
            plan.selected_threads = 0;
        } else {
            plan.selected_threads =
                best_candidate.actual_threads > 0 ? best_candidate.actual_threads : buildHeuristicPlan(kind).selected_threads;
        }
        plan.calibrated = true;
        return plan;
    }

    ExecutionPlan buildHeuristicPlan(TestKind kind) const {
        ExecutionPlan plan;

        // If the user explicitly selected the Metal backend or if auto and
        // Metal is available, prefer the Metal kernel.
        if (options_.backend == Backend::Metal ||
            (options_.backend == Backend::Auto && kernelSupported(platform_, metalKernelFor(kind)))) {
            plan.kernel = metalKernelFor(kind);
            plan.selected_threads = 0;  // GPU manages its own parallelism
            plan.calibrated = false;
            return plan;
        }

        plan.kernel = chooseHeuristicKernel(platform_, options_, kind);
        if (options_.threads_override > 0) {
            plan.selected_threads = options_.threads_override;
        } else if (options_.mode == RunMode::Peak && platform_.apple_silicon) {
            if (kind == TestKind::Read) {
                plan.selected_threads = chooseDefaultThreadCount(
                    platform_,
                    options_.thread_policy == ThreadPolicy::Perf ? ThreadPolicy::Perf
                                                                 : ThreadPolicy::All);
            } else {
                plan.selected_threads = chooseDefaultThreadCount(platform_, ThreadPolicy::Perf);
            }
        } else {
            plan.selected_threads = chooseDefaultThreadCount(platform_, options_.thread_policy);
        }

        if (options_.mode == RunMode::Standard && platform_.apple_silicon &&
            options_.threads_override == 0 && options_.thread_policy == ThreadPolicy::All) {
            plan.selected_threads = chooseDefaultThreadCount(platform_, ThreadPolicy::Perf);
        }
        plan.calibrated = false;
        return plan;
    }

    static KernelKind metalKernelFor(TestKind kind) {
        switch (kind) {
            case TestKind::Read:  return KernelKind::MetalRead;
            case TestKind::Write: return KernelKind::MetalWrite;
            case TestKind::Copy:  return KernelKind::MetalCopy;
        }
        return KernelKind::MetalRead;
    }

#ifdef MEMBENCH_HAS_METAL
    static MetalTestKind metalTestKindFor(TestKind kind) {
        switch (kind) {
            case TestKind::Read:  return MetalTestKind::Read;
            case TestKind::Write: return MetalTestKind::Write;
            case TestKind::Copy:  return MetalTestKind::Copy;
        }
        return MetalTestKind::Read;
    }

    TestResult runMetalTest(TestKind kind, const ExecutionPlan& plan) {
        MetalTestKind mk = metalTestKindFor(kind);
        MetalIterationResult mr = metalRunBandwidthTest(
            mk,
            options_.size_bytes,
            options_.warmup_iterations,
            options_.measured_iterations);

        TestResult result;
        result.bandwidth_mb_per_sec = calculateStatistics(mr.bandwidth_samples);
        result.elapsed_ms = calculateStatistics(mr.elapsed_samples);
        result.logical_bytes_per_iteration = mr.logical_bytes_per_iteration;
        printTestResult(kind, plan, 0, result);
        return result;
    }
#endif

    SummaryEntry makeSummaryEntry(TestKind kind,
                                  const ExecutionPlan& plan,
                                  unsigned int actual_threads,
                                  const TestResult& result) const {
        SummaryEntry entry;
        entry.kind = kind;
        entry.plan = plan;
        entry.actual_threads = actual_threads;
        entry.avg_bandwidth_gb_per_sec = result.bandwidth_mb_per_sec.average / 1024.0;
        entry.avg_traffic_gb_per_sec =
            (kind == TestKind::Copy) ? entry.avg_bandwidth_gb_per_sec * 2.0 : 0.0;
        return entry;
    }

    void printSummary(const std::vector<SummaryEntry>& entries) const {
        std::cout << "=== Summary ===\n";
        for (const auto& e : entries) {
            std::ostringstream line;
            line << std::fixed << std::setprecision(2);

            const std::string label = testKindToTitle(e.kind);
            line << "  " << label << ":  ";

            if (e.kind == TestKind::Copy) {
                line << e.avg_bandwidth_gb_per_sec << " GB/s logical, "
                     << e.avg_traffic_gb_per_sec << " GB/s traffic";
            } else {
                line << e.avg_bandwidth_gb_per_sec << " GB/s";
            }

            line << "  (" << kernelToString(e.plan.kernel);
            if (isMetalKernel(e.plan.kernel)) {
                line << ", gpu";
            } else {
                line << ", " << e.actual_threads << " threads";
            }
            if (e.plan.calibrated) {
                line << ", calibrated";
            }
            line << ")";
            std::cout << line.str() << '\n';
        }
        std::cout << '\n';
    }

    void printBandwidthStats(const std::string& prefix, const Statistics& stats) const {
        std::cout << std::setprecision(2);
        std::cout << prefix << "avg bandwidth: " << (stats.average / 1024.0) << " GB/s ("
                  << stats.average << " MB/s)\n";
        std::cout << prefix << "median bandwidth: " << (stats.median / 1024.0) << " GB/s ("
                  << stats.median << " MB/s)\n";
        std::cout << prefix << "min bandwidth: " << (stats.minimum / 1024.0) << " GB/s ("
                  << stats.minimum << " MB/s)\n";
        std::cout << prefix << "max bandwidth: " << (stats.maximum / 1024.0) << " GB/s ("
                  << stats.maximum << " MB/s)\n";
        std::cout << prefix << "stdev bandwidth: " << (stats.stdev / 1024.0) << " GB/s ("
                  << stats.stdev << " MB/s)\n";
    }

    void printTestResult(TestKind kind,
                         const ExecutionPlan& plan,
                         unsigned int actual_threads,
                         const TestResult& result) const {
        std::cout << "=== " << testKindToTitle(kind) << " Test ===\n";
        std::cout << "mode: " << runModeToString(options_.mode) << '\n';
        std::cout << "kernel: " << kernelToString(plan.kernel) << '\n';
        if (isMetalKernel(plan.kernel)) {
            std::cout << "selected_threads: gpu\n";
        } else {
            std::cout << "selected_threads: " << actual_threads << '\n';
        }
        std::cout << "calibrated: " << (plan.calibrated ? "yes" : "no") << '\n';
        std::cout << "size_mb: " << options_.size_bytes / MB << '\n';
        std::cout << "warmup: " << options_.warmup_iterations << '\n';
        std::cout << "iterations: " << options_.measured_iterations << '\n';
        std::cout << "logical_bytes_per_iteration: " << result.logical_bytes_per_iteration << " ("
                  << formatBytes(result.logical_bytes_per_iteration) << ")\n";
        std::cout << "measured_elapsed_ms: avg " << std::fixed << std::setprecision(3)
                  << result.elapsed_ms.average << ", median " << result.elapsed_ms.median << '\n';

        if (kind == TestKind::Copy) {
            printBandwidthStats("logical ", result.bandwidth_mb_per_sec);
            const Statistics traffic = scaleStatistics(result.bandwidth_mb_per_sec, 2.0);
            printBandwidthStats("estimated traffic ", traffic);
            std::cout << "Estimated traffic bandwidth is logical memcpy throughput multiplied by 2.\n";
        } else {
            printBandwidthStats("", result.bandwidth_mb_per_sec);
        }
        std::cout << '\n';
    }

    PlatformInfo platform_;
    BenchmarkOptions options_;
    std::size_t alignment_ = 0;
    AlignedBuffer buffer_a_;
    AlignedBuffer buffer_b_;
};

}  // namespace

int main(int argc, char* argv[]) {
    const PlatformInfo platform = detectPlatformInfo();

    BenchmarkOptions options;
    options.size_bytes = chooseDefaultBufferSize(platform);
    options.warmup_iterations = kDefaultWarmupIterations;
    options.measured_iterations = kDefaultMeasuredIterations;
    options.mode = chooseDefaultMode(platform);
    options.thread_policy = platform.apple_silicon ? ThreadPolicy::All : ThreadPolicy::All;
    options.calibrate = platform.apple_silicon || platform.x86_avx2;
    options.use_qos = platform.apple_silicon;
#ifdef MEMBENCH_HAS_METAL
    options.backend = platform.apple_silicon ? Backend::Auto : Backend::Cpu;
#else
    options.backend = Backend::Cpu;
#endif
    options.tests = {TestKind::Read, TestKind::Write, TestKind::Copy};

    bool positional_size_consumed = false;
    bool mode_explicit = false;
    bool thread_policy_explicit = false;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            auto requireValue = [&](const std::string& option_name) -> std::string {
                if (index + 1 >= argc) {
                    throw std::runtime_error("missing value for " + option_name);
                }
                ++index;
                return argv[index];
            };

            if (arg == "--help") {
                printUsage(argv[0], platform);
                return 0;
            }
            if (arg == "--size-mb") {
                std::uint64_t size_mb = 0;
                if (!parseUnsigned64(requireValue(arg), &size_mb) || size_mb == 0 ||
                    size_mb > (kMaxBufferSize / MB)) {
                    throw std::runtime_error("invalid --size-mb value");
                }
                options.size_bytes = static_cast<std::size_t>(size_mb) * MB;
                continue;
            }
            if (arg == "--threads") {
                std::uint64_t threads = 0;
                if (!parseUnsigned64(requireValue(arg), &threads) || threads == 0 ||
                    threads > std::numeric_limits<unsigned int>::max()) {
                    throw std::runtime_error("invalid --threads value");
                }
                options.threads_override = static_cast<unsigned int>(threads);
                continue;
            }
            if (arg == "--iterations") {
                std::uint64_t iterations = 0;
                if (!parseUnsigned64(requireValue(arg), &iterations) || iterations == 0) {
                    throw std::runtime_error("invalid --iterations value");
                }
                options.measured_iterations = static_cast<std::size_t>(iterations);
                continue;
            }
            if (arg == "--warmup") {
                std::uint64_t warmup = 0;
                if (!parseUnsigned64(requireValue(arg), &warmup)) {
                    throw std::runtime_error("invalid --warmup value");
                }
                options.warmup_iterations = static_cast<std::size_t>(warmup);
                continue;
            }
            if (arg == "--tests") {
                if (!parseTestList(requireValue(arg), &options.tests)) {
                    throw std::runtime_error("invalid --tests list");
                }
                continue;
            }
            if (arg == "--thread-policy") {
                if (!parseThreadPolicy(requireValue(arg), &options.thread_policy)) {
                    throw std::runtime_error("invalid --thread-policy value");
                }
                thread_policy_explicit = true;
                continue;
            }
            if (arg == "--mode") {
                if (!parseRunMode(requireValue(arg), &options.mode)) {
                    throw std::runtime_error("invalid --mode value");
                }
                mode_explicit = true;
                continue;
            }
            if (arg == "--backend") {
                if (!parseBackend(requireValue(arg), &options.backend)) {
                    throw std::runtime_error("invalid --backend value (use cpu, metal, or auto)");
                }
                continue;
            }
            if (arg == "--no-calibrate") {
                options.calibrate = false;
                continue;
            }
            if (arg == "--no-qos") {
                options.use_qos = false;
                continue;
            }
            if (!arg.empty() && arg.front() == '-') {
                throw std::runtime_error("unknown option: " + arg);
            }
            if (positional_size_consumed) {
                throw std::runtime_error("unexpected positional argument: " + arg);
            }

            std::uint64_t size_mb = 0;
            if (!parseUnsigned64(arg, &size_mb) || size_mb == 0 ||
                size_mb > (kMaxBufferSize / MB)) {
                throw std::runtime_error("invalid buffer size argument");
            }
            options.size_bytes = static_cast<std::size_t>(size_mb) * MB;
            positional_size_consumed = true;
        }

        if (platform.apple_silicon && mode_explicit && options.mode == RunMode::Standard &&
            !thread_policy_explicit) {
            options.thread_policy = ThreadPolicy::Perf;
        }
        if (options.mode == RunMode::Standard) {
            if (!mode_explicit && platform.apple_silicon) {
                options.thread_policy = ThreadPolicy::All;
            }
            if (!platform.apple_silicon) {
                options.calibrate = false;
            }
        } else if (!platform.apple_silicon && !platform.x86_avx2) {
            options.calibrate = false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "MemBench v" << MEMBENCH_VERSION << '\n';
    std::cout << "Memory Read/Write/Copy Benchmark\n";
    std::cout << "========================================\n\n";

    printSystemInfo(platform);

    try {
        MemoryBenchmark benchmark(platform, options);
        benchmark.printConfiguration();
        benchmark.run();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
