#include "planner/execution_planner.h"

#include "kernels/kernel_registry.h"

#include <algorithm>

namespace membench {

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

double calibrationOverrideRatio(const PlatformInfo& platform, TestKind kind) {
    if (platform.x86_avx2) {
        if (kind == TestKind::Copy) {
            return 1.0;
        }
        return 1.03;
    }

    switch (kind) {
        case TestKind::Read:  return 1.03;
        case TestKind::Write: return 1.05;
        case TestKind::Copy:  return 1.10;
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
    const bool want_cpu = (backend == Backend::Cpu || backend == Backend::Auto);
    const bool want_metal = (backend == Backend::Metal || backend == Backend::Auto);

    switch (kind) {
        case TestKind::Read:
            if (want_cpu) {
                kernels.push_back(KernelKind::ScalarAuto);
                if (kernelSupported(platform, KernelKind::IspcRead)) {
                    kernels.push_back(KernelKind::IspcRead);
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
                if (kernelSupported(platform, KernelKind::Avx2StreamStore)) {
                    kernels.push_back(KernelKind::Avx2StreamStore);
                } else {
                    kernels.push_back(KernelKind::LibcMemset);
                }
                if (kernelSupported(platform, KernelKind::IspcWrite)) {
                    kernels.push_back(KernelKind::IspcWrite);
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
                if (kernelSupported(platform, KernelKind::IspcCopy)) {
                    kernels.push_back(KernelKind::IspcCopy);
                }
                if (!platform.x86_avx2 && kernelSupported(platform, KernelKind::Avx2StreamCopy)) {
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
            if (platform.x86_avx2 && kernelSupported(platform, KernelKind::IspcCopy)) {
                return KernelKind::IspcCopy;
            }
            return KernelKind::LibcMemcpy;
    }
    return KernelKind::ScalarAuto;
}

}  // namespace membench
