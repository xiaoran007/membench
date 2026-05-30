#ifndef MEMBENCH_BACKENDS_CPU_BACKEND_H
#define MEMBENCH_BACKENDS_CPU_BACKEND_H

#include "core/types.h"
#include "runner/benchmark_runner.h"

#include <cstddef>
#include <cstdint>

namespace membench {

struct CpuBackendResult {
    TestResult result;
    unsigned int actual_threads = 0;
};

class CpuBackend {
public:
    CpuBackend(const PlatformInfo& platform,
               RunMode mode,
               bool use_qos,
               std::uint8_t* buffer_a,
               std::uint8_t* buffer_b,
               std::size_t size_bytes)
        : platform_(platform),
          mode_(mode),
          use_qos_(use_qos),
          buffer_a_(buffer_a),
          buffer_b_(buffer_b),
          size_bytes_(size_bytes) {}

    CpuBackendResult run(TestKind kind,
                         KernelKind kernel,
                         std::size_t warmup_iterations,
                         std::size_t measured_iterations,
                         unsigned int requested_threads,
                         std::size_t passes_per_iteration = 1) const {
        BenchmarkRunner runner(platform_,
                               mode_,
                               use_qos_,
                               buffer_a_,
                               buffer_b_,
                               size_bytes_,
                               requested_threads);
        CpuBackendResult output;
        output.result = runner.run(kind,
                                   kernel,
                                   warmup_iterations,
                                   measured_iterations,
                                   passes_per_iteration);
        output.actual_threads = runner.threadCount();
        return output;
    }

private:
    const PlatformInfo& platform_;
    RunMode mode_;
    bool use_qos_ = false;
    std::uint8_t* buffer_a_ = nullptr;
    std::uint8_t* buffer_b_ = nullptr;
    std::size_t size_bytes_ = 0;
};

}  // namespace membench

#endif  // MEMBENCH_BACKENDS_CPU_BACKEND_H
