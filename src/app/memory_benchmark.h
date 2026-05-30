#ifndef MEMBENCH_APP_MEMORY_BENCHMARK_H
#define MEMBENCH_APP_MEMORY_BENCHMARK_H

#include "core/format.h"
#include "core/random.h"
#include "core/statistics.h"
#include "core/types.h"
#include "kernels/kernel_registry.h"
#include "memory/aligned_buffer.h"
#include "planner/execution_planner.h"
#include "runner/benchmark_runner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef MEMBENCH_HAS_METAL
#include "metal_backend.h"
#endif

namespace membench {

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

}  // namespace membench

#endif  // MEMBENCH_APP_MEMORY_BENCHMARK_H
