#ifndef MEMBENCH_KERNELS_KERNEL_REGISTRY_H
#define MEMBENCH_KERNELS_KERNEL_REGISTRY_H

#include "core/types.h"

namespace membench {

bool kernelSupported(const PlatformInfo& platform, KernelKind kernel);
bool isMetalKernel(KernelKind kernel);
bool hasIspcKernels();

}  // namespace membench

#endif  // MEMBENCH_KERNELS_KERNEL_REGISTRY_H
