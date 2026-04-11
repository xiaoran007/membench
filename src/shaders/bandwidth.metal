#include <metal_stdlib>
using namespace metal;

// Each thread processes kElementsPerThread float4 vectors (4096 bytes).
// The inner loops are manually unrolled 4-wide for better GPU instruction
// scheduling and memory-level parallelism.
constant uint kElementsPerThread = 256;

// ---------------------------------------------------------------------------
// Read bandwidth kernel
//
// Every thread reads kElementsPerThread × float4 from the source buffer
// into four independent accumulators (ILP).  SIMD-group reduction
// (simd_xor) collapses all lane values so that only one thread per SIMD
// group touches the atomic sink, drastically reducing contention.
// ---------------------------------------------------------------------------
kernel void bw_read(
    device const float4* src     [[buffer(0)]],
    device atomic_uint*  sink    [[buffer(1)]],
    constant uint&       count   [[buffer(2)]],
    uint                 tid     [[thread_position_in_grid]],
    uint                 simd_lane [[thread_index_in_simdgroup]])
{
    uint base = tid * kElementsPerThread;
    if (base >= count) return;

    float4 a0 = float4(0.0), a1 = float4(0.0);
    float4 a2 = float4(0.0), a3 = float4(0.0);
    uint end = min(base + kElementsPerThread, count);
    uint i = base;
    for (; i + 4 <= end; i += 4) {
        a0 += src[i];   a1 += src[i+1];
        a2 += src[i+2]; a3 += src[i+3];
    }
    for (; i < end; ++i) a0 += src[i];

    float4 s = a0 + a1 + a2 + a3;
    uint h = as_type<uint>(s.x + s.y + s.z + s.w);
    h = simd_xor(h);
    if (simd_lane == 0) {
        atomic_fetch_xor_explicit(sink, h, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Write bandwidth kernel
// ---------------------------------------------------------------------------
kernel void bw_write(
    device float4*  dst       [[buffer(0)]],
    constant uint&  count     [[buffer(1)]],
    constant uint&  pattern   [[buffer(2)]],
    uint            tid       [[thread_position_in_grid]])
{
    uint base = tid * kElementsPerThread;
    if (base >= count) return;

    float val = as_type<float>(pattern);
    float4 fill = float4(val, val, val, val);
    uint end = min(base + kElementsPerThread, count);
    uint i = base;
    for (; i + 4 <= end; i += 4) {
        dst[i] = fill; dst[i+1] = fill;
        dst[i+2] = fill; dst[i+3] = fill;
    }
    for (; i < end; ++i) dst[i] = fill;
}

// ---------------------------------------------------------------------------
// Copy bandwidth kernel
// ---------------------------------------------------------------------------
kernel void bw_copy(
    device const float4* src   [[buffer(0)]],
    device float4*       dst   [[buffer(1)]],
    constant uint&       count [[buffer(2)]],
    uint                 tid   [[thread_position_in_grid]])
{
    uint base = tid * kElementsPerThread;
    if (base >= count) return;

    uint end = min(base + kElementsPerThread, count);
    uint i = base;
    for (; i + 4 <= end; i += 4) {
        float4 v0 = src[i];   float4 v1 = src[i+1];
        float4 v2 = src[i+2]; float4 v3 = src[i+3];
        dst[i] = v0; dst[i+1] = v1;
        dst[i+2] = v2; dst[i+3] = v3;
    }
    for (; i < end; ++i) dst[i] = src[i];
}
