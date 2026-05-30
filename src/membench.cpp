#include "version.h"

#include "core/format.h"
#include "core/statistics.h"
#include "core/types.h"
#include "kernels/kernel_registry.h"
#include "memory/aligned_buffer.h"
#include "platform/platform.h"
#include "planner/execution_planner.h"
#include "runner/benchmark_runner.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__arm64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef MEMBENCH_HAS_METAL
#include "metal_backend.h"
#endif

namespace {

using namespace membench;

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
    if (!platform.cpu_affinity_order.empty()) {
        std::cout << "CPU affinity order: " << platform.cpu_affinity_order.size()
                  << " logical CPUs, physical cores first\n";
    }
    std::cout << '\n';
}


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
        std::cout << "Worker affinity: "
                  << (!platform_.cpu_affinity_order.empty() ? "physical-first" : "disabled")
                  << '\n';
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
        if (platform_.x86_avx2 && kind == TestKind::Read) {
            return buildHeuristicPlan(kind);
        }

        const std::size_t calibration_size = std::min(options_.size_bytes, kCalibrationBufferSize);
        std::vector<unsigned int> thread_candidates = buildThreadCandidates(platform_, options_);
        if (platform_.x86_avx2 && kind == TestKind::Write && options_.threads_override == 0) {
            thread_candidates.erase(
                std::remove_if(thread_candidates.begin(),
                               thread_candidates.end(),
                               [](unsigned int threads) { return threads > 4; }),
                thread_candidates.end());
            if (thread_candidates.empty()) {
                thread_candidates.push_back(1);
            }
        }
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
            best_candidate.score_mb_per_sec = heuristic_result.bandwidth_mb_per_sec.median;
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
                    score = calculateStatistics(mr.bandwidth_samples).median;
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
                const double score = result.bandwidth_mb_per_sec.median;
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
        } else if (options_.mode == RunMode::Peak && platform_.x86_avx2 &&
                   kind == TestKind::Write) {
            plan.selected_threads = std::min(4U, chooseDefaultThreadCount(platform_, ThreadPolicy::All));
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
