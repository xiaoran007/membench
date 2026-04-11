#ifdef MEMBENCH_HAS_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_backend.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFloat4Size = sizeof(float) * 4;  // 16 bytes
constexpr std::size_t kElementsPerThread = 256;          // must match shader
constexpr NSUInteger kPreferredThreadgroupSize = 256;

constexpr std::size_t MB = 1024ULL * 1024ULL;

// Singleton-ish holder for the Metal device + shader library so we don't
// recreate them on every call.
struct MetalContext {
    id<MTLDevice>              device        = nil;
    id<MTLCommandQueue>        queue         = nil;
    id<MTLLibrary>             library       = nil;
    id<MTLComputePipelineState> readPipeline  = nil;
    id<MTLComputePipelineState> writePipeline = nil;
    id<MTLComputePipelineState> copyPipeline  = nil;
    bool                       initialised   = false;
    bool                       available     = false;
    std::string                deviceName;

    bool init() {
        if (initialised) return available;
        initialised = true;

        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) return false;

            deviceName = [[device name] UTF8String];
            queue = [device newCommandQueue];
            if (!queue) return false;

            // Compile shader source at runtime.
            NSError* error = nil;
            NSString* shaderSource = @
                "#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "constant uint kEPT = 256;\n"
                "\n"
                "kernel void bw_read(\n"
                "    device const float4* src [[buffer(0)]],\n"
                "    device atomic_uint* sink [[buffer(1)]],\n"
                "    constant uint& count [[buffer(2)]],\n"
                "    uint tid [[thread_position_in_grid]],\n"
                "    uint simd_lane [[thread_index_in_simdgroup]]) {\n"
                "  uint base = tid * kEPT;\n"
                "  if (base >= count) return;\n"
                "  float4 a0 = float4(0.0), a1 = float4(0.0);\n"
                "  float4 a2 = float4(0.0), a3 = float4(0.0);\n"
                "  uint end = min(base + kEPT, count);\n"
                "  uint i = base;\n"
                "  for (; i + 4 <= end; i += 4) {\n"
                "    a0 += src[i];   a1 += src[i+1];\n"
                "    a2 += src[i+2]; a3 += src[i+3];\n"
                "  }\n"
                "  for (; i < end; ++i) a0 += src[i];\n"
                "  float4 s = a0 + a1 + a2 + a3;\n"
                "  uint h = as_type<uint>(s.x + s.y + s.z + s.w);\n"
                "  h = simd_xor(h);\n"
                "  if (simd_lane == 0) {\n"
                "    atomic_fetch_xor_explicit(sink, h, memory_order_relaxed);\n"
                "  }\n"
                "}\n"
                "\n"
                "kernel void bw_write(\n"
                "    device float4* dst [[buffer(0)]],\n"
                "    constant uint& count [[buffer(1)]],\n"
                "    constant uint& pattern [[buffer(2)]],\n"
                "    uint tid [[thread_position_in_grid]]) {\n"
                "  uint base = tid * kEPT;\n"
                "  if (base >= count) return;\n"
                "  float val = as_type<float>(pattern);\n"
                "  float4 fill = float4(val);\n"
                "  uint end = min(base + kEPT, count);\n"
                "  uint i = base;\n"
                "  for (; i + 4 <= end; i += 4) {\n"
                "    dst[i] = fill; dst[i+1] = fill;\n"
                "    dst[i+2] = fill; dst[i+3] = fill;\n"
                "  }\n"
                "  for (; i < end; ++i) dst[i] = fill;\n"
                "}\n"
                "\n"
                "kernel void bw_copy(\n"
                "    device const float4* src [[buffer(0)]],\n"
                "    device float4* dst [[buffer(1)]],\n"
                "    constant uint& count [[buffer(2)]],\n"
                "    uint tid [[thread_position_in_grid]]) {\n"
                "  uint base = tid * kEPT;\n"
                "  if (base >= count) return;\n"
                "  uint end = min(base + kEPT, count);\n"
                "  uint i = base;\n"
                "  for (; i + 4 <= end; i += 4) {\n"
                "    float4 v0 = src[i];   float4 v1 = src[i+1];\n"
                "    float4 v2 = src[i+2]; float4 v3 = src[i+3];\n"
                "    dst[i] = v0; dst[i+1] = v1;\n"
                "    dst[i+2] = v2; dst[i+3] = v3;\n"
                "  }\n"
                "  for (; i < end; ++i) dst[i] = src[i];\n"
                "}\n";

            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeFast;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }
            library = [device newLibraryWithSource:shaderSource options:opts error:&error];
            if (!library) {
                std::cerr << "[metal] failed to load shader library";
                if (error) {
                    std::cerr << ": " << [[error localizedDescription] UTF8String];
                }
                std::cerr << '\n';
                return false;
            }

            auto makePipeline = [&](const char* name) -> id<MTLComputePipelineState> {
                NSString* nsName = [NSString stringWithUTF8String:name];
                id<MTLFunction> fn = [library newFunctionWithName:nsName];
                if (!fn) {
                    std::cerr << "[metal] shader function '" << name << "' not found\n";
                    return nil;
                }
                NSError* pipeErr = nil;
                id<MTLComputePipelineState> pso =
                    [device newComputePipelineStateWithFunction:fn error:&pipeErr];
                if (!pso) {
                    std::cerr << "[metal] pipeline creation failed for '" << name << "'";
                    if (pipeErr) {
                        std::cerr << ": " << [[pipeErr localizedDescription] UTF8String];
                    }
                    std::cerr << '\n';
                }
                return pso;
            };

            readPipeline  = makePipeline("bw_read");
            writePipeline = makePipeline("bw_write");
            copyPipeline  = makePipeline("bw_copy");

            available = (readPipeline && writePipeline && copyPipeline);
        }
        return available;
    }
};

MetalContext& ctx() {
    static MetalContext instance;
    return instance;
}

id<MTLComputePipelineState> pipelineForKind(MetalTestKind kind) {
    auto& c = ctx();
    switch (kind) {
        case MetalTestKind::Read:  return c.readPipeline;
        case MetalTestKind::Write: return c.writePipeline;
        case MetalTestKind::Copy:  return c.copyPipeline;
    }
    return nil;
}

double runOnce(MetalTestKind kind,
               id<MTLBuffer> bufA,
               id<MTLBuffer> bufB,
               id<MTLBuffer> sinkBuf,
               uint32_t float4Count,
               uint32_t pattern) {
    auto& c = ctx();
    id<MTLComputePipelineState> pso = pipelineForKind(kind);

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [c.queue commandBuffer];

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];

        switch (kind) {
            case MetalTestKind::Read:
                [enc setBuffer:bufA  offset:0 atIndex:0];
                [enc setBuffer:sinkBuf offset:0 atIndex:1];
                [enc setBytes:&float4Count length:sizeof(float4Count) atIndex:2];
                break;
            case MetalTestKind::Write:
                [enc setBuffer:bufA  offset:0 atIndex:0];
                [enc setBytes:&float4Count length:sizeof(float4Count) atIndex:1];
                [enc setBytes:&pattern     length:sizeof(pattern)     atIndex:2];
                break;
            case MetalTestKind::Copy:
                [enc setBuffer:bufA  offset:0 atIndex:0];
                [enc setBuffer:bufB  offset:0 atIndex:1];
                [enc setBytes:&float4Count length:sizeof(float4Count) atIndex:2];
                break;
        }

        NSUInteger threadsNeeded = (float4Count + kElementsPerThread - 1) / kElementsPerThread;
        NSUInteger threadgroupSize = std::min(pso.maxTotalThreadsPerThreadgroup,
                                              kPreferredThreadgroupSize);
        if (threadgroupSize > threadsNeeded) {
            threadgroupSize = threadsNeeded;
        }

        MTLSize gridSize = MTLSizeMake(threadsNeeded, 1, 1);
        MTLSize tgSize   = MTLSizeMake(threadgroupSize, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Use GPU timestamps when available; fall back to host wall-clock.
        double elapsed_sec = 0.0;
        if (@available(macOS 10.15, *)) {
            CFTimeInterval gpuStart = cmdBuf.GPUStartTime;
            CFTimeInterval gpuEnd   = cmdBuf.GPUEndTime;
            if (gpuEnd > gpuStart) {
                elapsed_sec = gpuEnd - gpuStart;
            }
        }
        if (elapsed_sec <= 0.0) {
            // Should not normally happen but guard against it.
            elapsed_sec = 1e-9;
        }

        return elapsed_sec;
    }
}

}  // namespace

bool metalIsAvailable() {
    return ctx().init();
}

std::string metalDeviceName() {
    if (!ctx().init()) return "unavailable";
    return ctx().deviceName;
}

MetalIterationResult metalRunBandwidthTest(
    MetalTestKind kind,
    std::size_t buffer_size_bytes,
    std::size_t warmup_iterations,
    std::size_t measured_iterations)
{
    MetalIterationResult result;
    if (!ctx().init()) return result;

    auto& c = ctx();

    // Round down to a multiple of kFloat4Size so indexing is clean.
    const std::size_t aligned_size = (buffer_size_bytes / kFloat4Size) * kFloat4Size;
    const uint32_t float4Count = static_cast<uint32_t>(aligned_size / kFloat4Size);
    if (float4Count == 0) return result;

    @autoreleasepool {
        // Allocate shared-mode buffers (UMA zero-copy).
        id<MTLBuffer> bufA = [c.device newBufferWithLength:aligned_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = nil;
        if (kind == MetalTestKind::Copy) {
            bufB = [c.device newBufferWithLength:aligned_size
                                         options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> sinkBuf = [c.device newBufferWithLength:sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];

        if (!bufA || (kind == MetalTestKind::Copy && !bufB) || !sinkBuf) {
            std::cerr << "[metal] buffer allocation failed\n";
            return result;
        }

        // Initialize buffers with non-zero data to avoid zero-page optimisation.
        {
            auto* ptr = static_cast<uint8_t*>(bufA.contents);
            for (std::size_t i = 0; i < aligned_size; ++i) {
                ptr[i] = static_cast<uint8_t>((i * 7 + 0x55) & 0xFF);
            }
        }
        if (bufB) {
            auto* ptr = static_cast<uint8_t*>(bufB.contents);
            for (std::size_t i = 0; i < aligned_size; ++i) {
                ptr[i] = static_cast<uint8_t>((i * 13 + 0xAA) & 0xFF);
            }
        }

        uint32_t pattern = 0x55555555U;

        // Warmup
        for (std::size_t i = 0; i < warmup_iterations; ++i) {
            (void)runOnce(kind, bufA, bufB, sinkBuf, float4Count, pattern);
            pattern ^= 0xFFFFFFFFU;
        }

        // Measured iterations
        std::size_t logical_bytes = aligned_size;
        if (kind == MetalTestKind::Copy) {
            // Copy reads + writes, but logical throughput = one buffer size
            // (consistent with CPU copy reporting).
        }

        result.logical_bytes_per_iteration = logical_bytes;

        for (std::size_t i = 0; i < measured_iterations; ++i) {
            double elapsed_sec = runOnce(kind, bufA, bufB, sinkBuf, float4Count, pattern);
            double elapsed_ms = elapsed_sec * 1000.0;
            double bw = (static_cast<double>(logical_bytes) / static_cast<double>(MB)) / elapsed_sec;
            result.bandwidth_samples.push_back(bw);
            result.elapsed_samples.push_back(elapsed_ms);
            pattern ^= 0xFFFFFFFFU;
        }
    }

    return result;
}

#endif // MEMBENCH_HAS_METAL
