#ifndef MEMBENCH_PLANNER_EXECUTION_PLANNER_H
#define MEMBENCH_PLANNER_EXECUTION_PLANNER_H

#include "core/types.h"

#include <cstddef>
#include <vector>

namespace membench {

std::size_t chooseDefaultBufferSize(const PlatformInfo& platform);
unsigned int chooseDefaultThreadCount(const PlatformInfo& platform, ThreadPolicy policy);
RunMode chooseDefaultMode(const PlatformInfo& platform);
double calibrationOverrideRatio(const PlatformInfo& platform, TestKind kind);
std::size_t clampThreadCountForSize(std::size_t size_bytes, unsigned int requested_threads);
std::vector<unsigned int> buildThreadCandidates(const PlatformInfo& platform,
                                                const BenchmarkOptions& options);
std::vector<KernelKind> buildKernelCandidates(const PlatformInfo& platform,
                                              TestKind kind,
                                              Backend backend);
KernelKind chooseHeuristicKernel(const PlatformInfo& platform,
                                 const BenchmarkOptions& options,
                                 TestKind kind);

}  // namespace membench

#endif  // MEMBENCH_PLANNER_EXECUTION_PLANNER_H
