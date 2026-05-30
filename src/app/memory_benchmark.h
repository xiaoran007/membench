#ifndef MEMBENCH_APP_MEMORY_BENCHMARK_H
#define MEMBENCH_APP_MEMORY_BENCHMARK_H

#include "backends/cpu_backend.h"
#include "core/format.h"
#include "core/random.h"
#include "core/statistics.h"
#include "core/types.h"
#include "kernels/kernel_registry.h"
#include "memory/aligned_buffer.h"
#include "planner/execution_planner.h"
#include "reporting/benchmark_reporter.h"

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
    MemoryBenchmark(const PlatformInfo& platform,
                    const BenchmarkOptions& options,
                    BenchmarkReporter& reporter)
        : platform_(platform),
          options_(options),
          reporter_(reporter),
          alignment_(std::max(platform.page_size, kCacheLineSize)),
          buffer_a_(),
          buffer_b_() {
        reporter_.beginAllocation(options_.size_bytes, alignment_);
        buffer_a_ = AlignedBuffer(options_.size_bytes, alignment_);
        buffer_b_ = AlignedBuffer(options_.size_bytes, alignment_);
        initializeBuffers();
    }

    void printConfiguration() const {
        reporter_.configuration(platform_, options_, alignment_, isCalibrationEnabled());
    }

    void run() {
        std::vector<BenchmarkSummaryEntry> summary;

        for (TestKind kind : options_.tests) {
            const ExecutionPlan plan = selectExecutionPlan(kind);
            reporter_.beginTest(kind, plan);

            if (isMetalKernel(plan.kernel)) {
#ifdef MEMBENCH_HAS_METAL
                TestResult result = runMetalTest(kind, plan);
                summary.push_back(makeSummaryEntry(kind, plan, 0, result));
#endif
            } else {
                CpuBackend backend(platform_,
                                   options_.mode,
                                   options_.use_qos,
                                   buffer_a_.data(),
                                   buffer_b_.data(),
                                   options_.size_bytes);
                const CpuBackendResult output = backend.run(kind,
                                                            plan.kernel,
                                                            options_.warmup_iterations,
                                                            options_.measured_iterations,
                                                            plan.selected_threads);
                reporter_.testCompleted(kind, plan, output.actual_threads, output.result);
                summary.push_back(makeSummaryEntry(kind, plan, output.actual_threads, output.result));
            }
        }

        reporter_.summary(summary);
        reporter_.finishRun();
    }

private:
    bool isCalibrationEnabled() const {
        return options_.mode == RunMode::Peak && options_.calibrate &&
               (platform_.apple_silicon || platform_.x86_avx2);
    }

    void initializeBuffers() {
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
        reporter_.finishInitialization();
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
            CpuBackend heuristic_backend(platform_,
                                         options_.mode,
                                         options_.use_qos,
                                         buffer_a_.data(),
                                         buffer_b_.data(),
                                         calibration_size);
            const CpuBackendResult heuristic_output =
                heuristic_backend.run(kind,
                                      heuristic_plan.kernel,
                                      kCalibrationWarmupIterations,
                                      kCalibrationMeasuredIterations,
                                      heuristic_plan.selected_threads,
                                      kCalibrationPassesPerIteration);
            best_candidate.kernel = heuristic_plan.kernel;
            best_candidate.requested_threads = heuristic_plan.selected_threads;
            best_candidate.actual_threads = heuristic_output.actual_threads;
            best_candidate.score_mb_per_sec = heuristic_output.result.bandwidth_mb_per_sec.median;
        }

        std::size_t total_candidates = 0;
        for (KernelKind kernel : kernel_candidates) {
            if (!kernelSupported(platform_, kernel)) {
                continue;
            }
            total_candidates += isMetalKernel(kernel) ? 1 : thread_candidates.size();
        }

        std::size_t completed_candidates = 0;
        reporter_.beginCalibration(kind, total_candidates);

        const double override_ratio = calibrationOverrideRatio(platform_, kind);
        for (KernelKind kernel : kernel_candidates) {
            if (!kernelSupported(platform_, kernel)) {
                continue;
            }

            if (isMetalKernel(kernel)) {
#ifdef MEMBENCH_HAS_METAL
                ++completed_candidates;
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
                reporter_.calibrationCandidate(
                    {kind,
                     kernel,
                     0,
                     completed_candidates,
                     total_candidates,
                     best_candidate.score_mb_per_sec,
                     best_candidate.kernel,
                     best_candidate.actual_threads});
#endif
                continue;
            }

            for (unsigned int requested_threads : thread_candidates) {
                CpuBackend backend(platform_,
                                   options_.mode,
                                   options_.use_qos,
                                   buffer_a_.data(),
                                   buffer_b_.data(),
                                   calibration_size);
                ++completed_candidates;
                const CpuBackendResult output = backend.run(kind,
                                                            kernel,
                                                            kCalibrationWarmupIterations,
                                                            kCalibrationMeasuredIterations,
                                                            requested_threads,
                                                            kCalibrationPassesPerIteration);
                const double score = output.result.bandwidth_mb_per_sec.median;
                if (kernel == best_candidate.kernel &&
                    output.actual_threads == best_candidate.actual_threads) {
                    reporter_.calibrationCandidate(
                        {kind,
                         kernel,
                         output.actual_threads,
                         completed_candidates,
                         total_candidates,
                         best_candidate.score_mb_per_sec,
                         best_candidate.kernel,
                         best_candidate.actual_threads});
                    continue;
                }
                if (score > best_candidate.score_mb_per_sec * override_ratio) {
                    best_candidate.kernel = kernel;
                    best_candidate.requested_threads = requested_threads;
                    best_candidate.actual_threads = output.actual_threads;
                    best_candidate.score_mb_per_sec = score;
                }
                reporter_.calibrationCandidate(
                    {kind,
                     kernel,
                     output.actual_threads,
                     completed_candidates,
                     total_candidates,
                     best_candidate.score_mb_per_sec,
                     best_candidate.kernel,
                     best_candidate.actual_threads});
            }
        }
        reporter_.calibrationSelected(kind, best_candidate);

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
        reporter_.testCompleted(kind, plan, 0, result);
        return result;
    }
#endif

    BenchmarkSummaryEntry makeSummaryEntry(TestKind kind,
                                           const ExecutionPlan& plan,
                                           unsigned int actual_threads,
                                           const TestResult& result) const {
        BenchmarkSummaryEntry entry;
        entry.kind = kind;
        entry.plan = plan;
        entry.actual_threads = actual_threads;
        entry.avg_bandwidth_gb_per_sec = result.bandwidth_mb_per_sec.average / 1024.0;
        entry.avg_traffic_gb_per_sec =
            (kind == TestKind::Copy) ? entry.avg_bandwidth_gb_per_sec * 2.0 : 0.0;
        return entry;
    }

    PlatformInfo platform_;
    BenchmarkOptions options_;
    BenchmarkReporter& reporter_;
    std::size_t alignment_ = 0;
    AlignedBuffer buffer_a_;
    AlignedBuffer buffer_b_;
};

}  // namespace membench

#endif  // MEMBENCH_APP_MEMORY_BENCHMARK_H
