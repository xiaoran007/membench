#include <metal_stdlib>
using namespace metal;

// Each thread processes up to kElementsPerThread float4 vectors. Accesses are
// grid-strided so adjacent SIMD lanes touch adjacent float4 values on every
// load/store instruction.
constant uint kElementsPerThread = 256;

// ---------------------------------------------------------------------------
// Read bandwidth kernel
//
// Every thread reads up to kElementsPerThread × float4 from the source buffer
// into four independent accumulators (ILP). SIMD-group reduction (simd_xor)
// collapses all lane values so that only one thread per SIMD group touches the
// atomic sink, drastically reducing contention.
// ---------------------------------------------------------------------------
kernel void bw_read(
    device const float4* src     [[buffer(0)]],
    device atomic_uint*  sink    [[buffer(1)]],
    constant uint&       count   [[buffer(2)]],
    constant uint&       total_threads [[buffer(3)]],
    uint                 tid     [[thread_position_in_grid]],
    uint                 simd_lane [[thread_index_in_simdgroup]])
{
    if (tid >= count) return;

    float4 a0 = float4(0.0), a1 = float4(0.0);
    float4 a2 = float4(0.0), a3 = float4(0.0);
    for (uint step = 0; step < kElementsPerThread; step += 4) {
        uint i0 = tid + step * total_threads;
        if (i0 >= count) break;
        a0 += src[i0];

        uint i1 = i0 + total_threads;
        if (i1 < count) a1 += src[i1];

        uint i2 = i1 + total_threads;
        if (i2 < count) a2 += src[i2];

        uint i3 = i2 + total_threads;
        if (i3 < count) a3 += src[i3];
    }

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
    constant uint&  total_threads [[buffer(3)]],
    uint            tid       [[thread_position_in_grid]])
{
    if (tid >= count) return;

    float val = as_type<float>(pattern);
    float4 fill = float4(val, val, val, val);
    for (uint step = 0; step < kElementsPerThread; step += 4) {
        uint i0 = tid + step * total_threads;
        if (i0 >= count) break;
        dst[i0] = fill;

        uint i1 = i0 + total_threads;
        if (i1 < count) dst[i1] = fill;

        uint i2 = i1 + total_threads;
        if (i2 < count) dst[i2] = fill;

        uint i3 = i2 + total_threads;
        if (i3 < count) dst[i3] = fill;
    }
}

// ---------------------------------------------------------------------------
// Copy bandwidth kernel
// ---------------------------------------------------------------------------
kernel void bw_copy(
    device const float4* src   [[buffer(0)]],
    device float4*       dst   [[buffer(1)]],
    constant uint&       count [[buffer(2)]],
    constant uint&       total_threads [[buffer(3)]],
    uint                 tid   [[thread_position_in_grid]])
{
    if (tid >= count) return;

    for (uint step = 0; step < kElementsPerThread; step += 4) {
        uint i0 = tid + step * total_threads;
        if (i0 >= count) break;
        dst[i0] = src[i0];

        uint i1 = i0 + total_threads;
        if (i1 < count) dst[i1] = src[i1];

        uint i2 = i1 + total_threads;
        if (i2 < count) dst[i2] = src[i2];

        uint i3 = i2 + total_threads;
        if (i3 < count) dst[i3] = src[i3];
    }
}
