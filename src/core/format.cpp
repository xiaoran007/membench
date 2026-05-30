#include "core/format.h"

#include <iomanip>
#include <sstream>

namespace membench {

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
        case TestKind::Read:  return "read";
        case TestKind::Write: return "write";
        case TestKind::Copy:  return "copy";
    }
    return "unknown";
}

std::string testKindToTitle(TestKind kind) {
    switch (kind) {
        case TestKind::Read:  return "Sequential Read";
        case TestKind::Write: return "Sequential Write";
        case TestKind::Copy:  return "Memory Copy";
    }
    return "Unknown";
}

std::string threadPolicyToString(ThreadPolicy policy) {
    return policy == ThreadPolicy::Perf ? "perf" : "all";
}

std::string runModeToString(RunMode mode) {
    return mode == RunMode::Peak ? "peak" : "standard";
}

std::string backendToString(Backend backend) {
    switch (backend) {
        case Backend::Cpu:   return "cpu";
        case Backend::Metal: return "metal";
        case Backend::Auto:  return "auto";
    }
    return "unknown";
}

std::string kernelToString(KernelKind kernel) {
    switch (kernel) {
        case KernelKind::ScalarAuto:      return "scalar_auto";
        case KernelKind::NeonPeak:        return "neon_peak";
        case KernelKind::LibcMemset:      return "libc_memset";
        case KernelKind::NeonStore:       return "neon_store";
        case KernelKind::LibcMemcpy:      return "libc_memcpy";
        case KernelKind::NeonCopy:        return "neon_copy";
        case KernelKind::Avx2Read:        return "avx2_read";
        case KernelKind::Avx2StreamStore: return "avx2_stream_store";
        case KernelKind::Avx2StreamCopy:  return "avx2_stream_copy";
        case KernelKind::IspcRead:        return "ispc_read";
        case KernelKind::IspcWrite:       return "ispc_write";
        case KernelKind::IspcCopy:        return "ispc_copy";
        case KernelKind::MetalRead:       return "metal_read";
        case KernelKind::MetalWrite:      return "metal_write";
        case KernelKind::MetalCopy:       return "metal_copy";
    }
    return "unknown";
}

}  // namespace membench
