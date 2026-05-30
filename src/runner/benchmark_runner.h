#ifndef MEMBENCH_RUNNER_BENCHMARK_RUNNER_H
#define MEMBENCH_RUNNER_BENCHMARK_RUNNER_H

#include "core/random.h"
#include "core/statistics.h"
#include "core/types.h"
#include "kernels/kernel_registry.h"
#include "platform/platform.h"
#include "planner/execution_planner.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
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

namespace membench {

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
          thread_count_(resolveThreadCount(requested_threads)),
          cpu_affinity_order_(platform.cpu_affinity_order) {
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
        applyCurrentThreadPolicy(mode_, use_qos_);
    }

    void maybeApplyThreadAffinity(unsigned int worker_index) const {
        applyCurrentThreadAffinity(cpu_affinity_order_, worker_index);
    }

    void workerLoop(unsigned int worker_index, const Slice& slice) {
        maybeApplyThreadAffinity(worker_index);
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
    std::vector<unsigned int> cpu_affinity_order_;
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

}  // namespace membench

#endif  // MEMBENCH_RUNNER_BENCHMARK_RUNNER_H
