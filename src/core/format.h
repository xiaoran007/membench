#ifndef MEMBENCH_CORE_FORMAT_H
#define MEMBENCH_CORE_FORMAT_H

#include "core/types.h"

#include <cstdint>
#include <string>

namespace membench {

std::string formatBytes(std::uint64_t bytes);
std::string testKindToCliName(TestKind kind);
std::string testKindToTitle(TestKind kind);
std::string threadPolicyToString(ThreadPolicy policy);
std::string runModeToString(RunMode mode);
std::string backendToString(Backend backend);
std::string kernelToString(KernelKind kernel);

}  // namespace membench

#endif  // MEMBENCH_CORE_FORMAT_H
