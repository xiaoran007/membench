#include "kernels/kernel_registry.h"

#ifdef MEMBENCH_HAS_METAL
#include "metal_backend.h"
#endif

namespace membench {

bool kernelSupported(const PlatformInfo& platform, KernelKind kernel) {
    switch (kernel) {
        case KernelKind::ScalarAuto:
        case KernelKind::LibcMemset:
        case KernelKind::LibcMemcpy:
            return true;
        case KernelKind::Avx2Read:
        case KernelKind::Avx2StreamStore:
        case KernelKind::Avx2StreamCopy:
#if defined(__AVX2__)
            return platform.x86_avx2;
#else
            (void)platform;
            return false;
#endif
        case KernelKind::NeonPeak:
        case KernelKind::NeonStore:
        case KernelKind::NeonCopy:
            return platform.apple_silicon;
        case KernelKind::MetalRead:
        case KernelKind::MetalWrite:
        case KernelKind::MetalCopy:
#ifdef MEMBENCH_HAS_METAL
            return platform.apple_silicon && metalIsAvailable();
#else
            (void)platform;
            return false;
#endif
    }
    return false;
}

bool isMetalKernel(KernelKind kernel) {
    return kernel == KernelKind::MetalRead ||
           kernel == KernelKind::MetalWrite ||
           kernel == KernelKind::MetalCopy;
}

}  // namespace membench
