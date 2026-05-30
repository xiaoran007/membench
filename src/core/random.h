#ifndef MEMBENCH_CORE_RANDOM_H
#define MEMBENCH_CORE_RANDOM_H

#include <cstdint>

namespace membench {

inline std::uint64_t splitMix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

}  // namespace membench

#endif  // MEMBENCH_CORE_RANDOM_H
