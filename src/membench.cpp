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
#include <malloc.h>
#include <windows.h>
#include <intrin.h>
#else
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace {

constexpr std::size_t KB = 1024;
constexpr std::size_t MB = 1024 * KB;
constexpr std::size_t GB = 1024 * MB;
constexpr std::size_t kCacheLineSize = 64;
constexpr std::size_t kDefaultWarmupIterations = 2;
constexpr std::size_t kDefaultMeasuredIterations = 7;
constexpr std::size_t kMinDefaultBufferSize = 256 * MB;
constexpr std::size_t kMaxDefaultBufferSize = 1 * GB;
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

struct PlatformInfo {
    std::size_t page_size = 4096;
    std::uint64_t physical_memory_bytes = 0;
    unsigned int hardware_threads = 1;
    unsigned int performance_cores = 0;
    bool apple_silicon = false;
};

struct BenchmarkOptions {
    std::size_t size_bytes = 0;
    std::size_t warmup_iterations = kDefaultWarmupIterations;
    std::size_t measured_iterations = kDefaultMeasuredIterations;
    unsigned int threads_override = 0;
    ThreadPolicy thread_policy = ThreadPolicy::All;
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
    std::vector<double> bandwidth_samples_mb_per_sec;
    std::vector<double> elapsed_samples_ms;
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
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &perf_cores_len,
                     nullptr, 0) == 0) {
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

void printUsage(const char* program_name, const PlatformInfo& platform) {
    const std::size_t default_size_mb = chooseDefaultBufferSize(platform) / MB;
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
              << "  --no-qos              Disable macOS QoS hinting\n"
              << "  --help                Show this message\n\n"
              << "Examples:\n"
              << "  " << program_name << " 1024\n"
              << "  " << program_name << " --size-mb 1024 --tests read,copy\n"
              << "  " << program_name
              << " --threads 4 --warmup 2 --iterations 7 --thread-policy perf\n";
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
    std::size_t alignment() const { return alignment_; }

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

}  // namespace

class MemoryBenchmark {
public:
    MemoryBenchmark(const PlatformInfo& platform, const BenchmarkOptions& options)
        : platform_(platform),
          options_(options),
          alignment_(std::max(platform.page_size, kCacheLineSize)),
          thread_count_(resolveThreadCount()),
          buffer_a_(options.size_bytes, alignment_),
          buffer_b_(options.size_bytes, alignment_) {
        initializeBuffers();
        buildSlices();
        startWorkers();
    }

    ~MemoryBenchmark() {
        stopWorkers();
    }

    void printConfiguration() const {
        std::cout << "=== Benchmark Configuration ===\n";
        std::cout << "Size: " << options_.size_bytes / MB << " MiB\n";
        std::cout << "Alignment: " << alignment_ << " bytes\n";
        std::cout << "Threads: " << thread_count_ << '\n';
        std::cout << "Warmup iterations: " << options_.warmup_iterations << '\n';
        std::cout << "Measured iterations: " << options_.measured_iterations << '\n';
        std::cout << "Thread policy: " << threadPolicyToString(options_.thread_policy) << '\n';
        std::cout << "macOS QoS hint: " << (options_.use_qos ? "enabled" : "disabled") << '\n';
        std::cout << "Tests: ";
        for (std::size_t i = 0; i < options_.tests.size(); ++i) {
            if (i > 0) {
                std::cout << ',';
            }
            std::cout << testKindToCliName(options_.tests[i]);
        }
        std::cout << "\n\n";
    }

    void run() {
        for (TestKind kind : options_.tests) {
            runBandwidthTest(kind);
        }
    }

private:
    struct Slice {
        std::size_t offset = 0;
        std::size_t size = 0;
    };

    struct WorkerCommand {
        TestKind kind = TestKind::Read;
        std::size_t passes = 1;
        std::uint8_t write_pattern = 0xAA;
    };

    struct TimerResult {
        std::uint64_t elapsed_ns = 0;
    };

    template <typename T>
    static inline void doNotOptimize(const T& value) {
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

    static std::uint64_t monotonicNowNs() {
#ifdef __APPLE__
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
#endif
    }

    unsigned int resolveThreadCount() const {
        unsigned int requested = options_.threads_override > 0
                                     ? options_.threads_override
                                     : chooseDefaultThreadCount(platform_, options_.thread_policy);
        const std::size_t max_threads_by_size =
            std::max<std::size_t>(1, options_.size_bytes / kCacheLineSize);
        requested = static_cast<unsigned int>(
            std::min<std::size_t>(requested, max_threads_by_size));
        return std::max(1U, requested);
    }

    void initializeBuffers() {
        std::cout << "Allocating " << options_.size_bytes / MB << " MiB per buffer..." << std::endl;
        auto* words_a = reinterpret_cast<std::uint64_t*>(buffer_a_.data());
        auto* words_b = reinterpret_cast<std::uint64_t*>(buffer_b_.data());
        const std::size_t word_count = options_.size_bytes / sizeof(std::uint64_t);

        for (std::size_t i = 0; i < word_count; ++i) {
            words_a[i] = splitMix64(0x123456789ABCDEF0ULL + static_cast<std::uint64_t>(i));
            words_b[i] = splitMix64(0x0FEDCBA987654321ULL + static_cast<std::uint64_t>(i));
        }

        const std::size_t tail_offset = word_count * sizeof(std::uint64_t);
        for (std::size_t i = tail_offset; i < options_.size_bytes; ++i) {
            buffer_a_.data()[i] = static_cast<std::uint8_t>(i & 0xFFU);
            buffer_b_.data()[i] = static_cast<std::uint8_t>((255U - i) & 0xFFU);
        }

        doNotOptimize(words_a[0]);
        doNotOptimize(words_b[0]);
        std::cout << "Buffers initialized with deterministic non-zero data.\n\n";
    }

    void buildSlices() {
        slices_.clear();
        slices_.reserve(thread_count_);

        const std::size_t aligned_chunk =
            alignDown(options_.size_bytes / thread_count_, kCacheLineSize);
        std::size_t offset = 0;
        for (unsigned int index = 0; index < thread_count_; ++index) {
            Slice slice;
            slice.offset = offset;
            if (index == thread_count_ - 1) {
                slice.size = options_.size_bytes - offset;
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
        workers_.clear();
    }

    static std::size_t alignDown(std::size_t value, std::size_t alignment) {
        return value - (value % alignment);
    }

    void maybeApplyThreadPolicy() const {
#ifdef __APPLE__
        if (options_.use_qos && options_.thread_policy == ThreadPolicy::Perf) {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        }
#else
        (void)options_;
#endif
    }

    void workerLoop(unsigned int index, const Slice& slice) {
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

            executeCommand(command, slice, index);

            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                ++completed_workers_;
                if (completed_workers_ == thread_count_) {
                    worker_done_cv_.notify_one();
                }
            }
        }
    }

    void executeCommand(const WorkerCommand& command, const Slice& slice, unsigned int index) {
        switch (command.kind) {
            case TestKind::Read:
                runReadPasses(slice, command.passes, index);
                break;
            case TestKind::Write:
                runWritePasses(slice, command.passes, command.write_pattern);
                break;
            case TestKind::Copy:
                runCopyPasses(slice, command.passes);
                break;
        }
    }

    void runReadPasses(const Slice& slice, std::size_t passes, unsigned int worker_index) {
        const auto* bytes = buffer_a_.data() + slice.offset;
        const auto* words = reinterpret_cast<const std::uint64_t*>(bytes);
        const std::size_t word_count = slice.size / sizeof(std::uint64_t);

        std::uint64_t accumulator0 = splitMix64(worker_index + 1U);
        std::uint64_t accumulator1 = splitMix64(worker_index + 11U);
        std::uint64_t accumulator2 = splitMix64(worker_index + 21U);
        std::uint64_t accumulator3 = splitMix64(worker_index + 31U);

        for (std::size_t pass = 0; pass < passes; ++pass) {
            std::size_t i = 0;
            for (; i + 4 <= word_count; i += 4) {
                accumulator0 += words[i];
                accumulator1 += words[i + 1];
                accumulator2 += words[i + 2];
                accumulator3 += words[i + 3];
            }
            for (; i < word_count; ++i) {
                accumulator0 += words[i];
            }
        }

        const std::size_t tail_offset = word_count * sizeof(std::uint64_t);
        std::uint64_t tail = 0;
        for (std::size_t i = tail_offset; i < slice.size; ++i) {
            tail += bytes[i];
        }

        const std::uint64_t local_sink =
            accumulator0 ^ accumulator1 ^ accumulator2 ^ accumulator3 ^ tail;
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
    }

    void runWritePasses(const Slice& slice, std::size_t passes, std::uint8_t base_pattern) {
        auto* bytes = buffer_a_.data() + slice.offset;
        std::uint8_t pattern = base_pattern;

        for (std::size_t pass = 0; pass < passes; ++pass) {
            std::memset(bytes, pattern, slice.size);
            pattern = (pattern == 0x55U) ? 0xAAU : 0x55U;
        }

        const std::uint64_t local_sink =
            static_cast<std::uint64_t>(bytes[0]) ^
            static_cast<std::uint64_t>(bytes[slice.size - 1]) << 8U;
        sink_.fetch_xor(local_sink, std::memory_order_relaxed);
        doNotOptimize(local_sink);
    }

    void runCopyPasses(const Slice& slice, std::size_t passes) {
        auto* a = buffer_a_.data() + slice.offset;
        auto* b = buffer_b_.data() + slice.offset;

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

    TimerResult runTimedCommand(const WorkerCommand& command) {
        {
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
            worker_done_cv_.wait(done_lock,
                                 [this]() { return completed_workers_ == thread_count_; });
            const std::uint64_t end_ns = monotonicNowNs();
            return {end_ns - start_ns};
        }
    }

    void printTestResult(TestKind kind, const TestResult& result) const {
        const auto& bw = result.bandwidth_mb_per_sec;
        const auto& elapsed = result.elapsed_ms;

        std::cout << "=== " << testKindToTitle(kind) << " Test ===\n";
        std::cout << "size_mb: " << options_.size_bytes / MB << '\n';
        std::cout << "threads: " << thread_count_ << '\n';
        std::cout << "warmup: " << options_.warmup_iterations << '\n';
        std::cout << "iterations: " << options_.measured_iterations << '\n';
        std::cout << "logical_bytes_per_iteration: " << options_.size_bytes << " ("
                  << formatBytes(options_.size_bytes) << ")\n";
        std::cout << "measured_elapsed_ms: avg " << std::fixed << std::setprecision(3)
                  << elapsed.average << ", median " << elapsed.median << '\n';
        std::cout << std::setprecision(2);
        std::cout << "avg bandwidth: " << (bw.average / 1024.0) << " GB/s (" << bw.average
                  << " MB/s)\n";
        std::cout << "median bandwidth: " << (bw.median / 1024.0) << " GB/s (" << bw.median
                  << " MB/s)\n";
        std::cout << "min bandwidth: " << (bw.minimum / 1024.0) << " GB/s (" << bw.minimum
                  << " MB/s)\n";
        std::cout << "max bandwidth: " << (bw.maximum / 1024.0) << " GB/s (" << bw.maximum
                  << " MB/s)\n";
        std::cout << "stdev bandwidth: " << (bw.stdev / 1024.0) << " GB/s (" << bw.stdev
                  << " MB/s)\n";
        if (kind == TestKind::Copy) {
            std::cout << "Copy throughput reports logical copied bytes, not doubled DRAM traffic.\n";
        }
        std::cout << '\n';
    }

    void runBandwidthTest(TestKind kind) {
        std::vector<double> bandwidth_samples;
        std::vector<double> elapsed_samples;

        std::uint8_t write_pattern = 0x55U;
        for (std::size_t warmup_index = 0; warmup_index < options_.warmup_iterations;
             ++warmup_index) {
            WorkerCommand command;
            command.kind = kind;
            command.passes = 1;
            command.write_pattern = write_pattern;
            (void)runTimedCommand(command);
            if (kind == TestKind::Write) {
                write_pattern = (write_pattern == 0x55U) ? 0xAAU : 0x55U;
            }
        }

        for (std::size_t iteration = 0; iteration < options_.measured_iterations; ++iteration) {
            WorkerCommand command;
            command.kind = kind;
            command.passes = 1;
            command.write_pattern = write_pattern;

            const TimerResult timer = runTimedCommand(command);
            const double seconds = static_cast<double>(timer.elapsed_ns) / 1'000'000'000.0;
            const double bandwidth_mb_per_sec =
                (options_.size_bytes / static_cast<double>(MB)) / seconds;
            bandwidth_samples.push_back(bandwidth_mb_per_sec);
            elapsed_samples.push_back(static_cast<double>(timer.elapsed_ns) / 1'000'000.0);

            if (kind == TestKind::Write) {
                write_pattern = (write_pattern == 0x55U) ? 0xAAU : 0x55U;
            }
        }

        TestResult result;
        result.bandwidth_samples_mb_per_sec = bandwidth_samples;
        result.elapsed_samples_ms = elapsed_samples;
        result.bandwidth_mb_per_sec = calculateStatistics(bandwidth_samples);
        result.elapsed_ms = calculateStatistics(elapsed_samples);
        printTestResult(kind, result);
    }

    PlatformInfo platform_;
    BenchmarkOptions options_;
    std::size_t alignment_ = 0;
    unsigned int thread_count_ = 1;
    AlignedBuffer buffer_a_;
    AlignedBuffer buffer_b_;
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

int main(int argc, char* argv[]) {
    const PlatformInfo platform = detectPlatformInfo();

    BenchmarkOptions options;
    options.size_bytes = chooseDefaultBufferSize(platform);
    options.warmup_iterations = kDefaultWarmupIterations;
    options.measured_iterations = kDefaultMeasuredIterations;
    options.thread_policy = platform.apple_silicon ? ThreadPolicy::Perf : ThreadPolicy::All;
    options.use_qos = platform.apple_silicon;
    options.tests = {TestKind::Read, TestKind::Write, TestKind::Copy};

    bool positional_size_consumed = false;
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
        if (!parseUnsigned64(arg, &size_mb) || size_mb == 0 || size_mb > (kMaxBufferSize / MB)) {
            throw std::runtime_error("invalid buffer size argument");
        }
        options.size_bytes = static_cast<std::size_t>(size_mb) * MB;
        positional_size_consumed = true;
    }

    std::cout << "========================================\n";
    std::cout << "MemBench v" << MEMBENCH_VERSION << '\n';
    std::cout << "Reliable Memory Read/Write/Copy Benchmark\n";
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
