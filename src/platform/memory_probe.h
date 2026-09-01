#ifndef MEMBENCH_PLATFORM_MEMORY_PROBE_H
#define MEMBENCH_PLATFORM_MEMORY_PROBE_H

#include "core/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace membench {

enum class MemoryProbeReadStatus {
    NotAttempted,
    Success,
    NotFound,
    PermissionDenied,
    IoError,
    Empty,
};

enum class MemoryProbeParseStatus {
    NotAttempted,
    Success,
    Malformed,
    NoMemoryDevices,
};

struct MemoryDeviceProbe {
    std::size_t structure_index = 0;
    std::size_t structure_offset = 0;
    std::size_t structure_length = 0;
    bool populated = false;
    std::uint8_t raw_memory_type = 0;
    std::string technology;
    unsigned int data_width_bits = 0;
    std::uint64_t speed_mt_s = 0;
    std::uint64_t configured_speed_mt_s = 0;
    std::uint64_t effective_rate_mt_s = 0;
};

struct MemoryProbeResult {
    std::string source;
    MemoryProbeReadStatus read_status = MemoryProbeReadStatus::NotAttempted;
    std::string read_error;
    std::size_t bytes_read = 0;
    MemoryProbeParseStatus parse_status = MemoryProbeParseStatus::NotAttempted;
    std::string parse_error;
    std::size_t structures_seen = 0;
    bool end_marker_seen = false;
    std::vector<MemoryDeviceProbe> devices;
    PlatformInfo::MemoryInfo memory;
    std::string rate_reason;
    std::string aggregate_width_reason;
};

MemoryProbeResult probeMemoryFromSmbiosBytes(const std::vector<std::uint8_t>& table,
                                             const std::string& source);
MemoryProbeResult probeMemoryFromSmbiosFile(const std::string& path);
MemoryProbeResult probeSystemSmbiosMemory();

const char* memoryProbeReadStatusName(MemoryProbeReadStatus status);
const char* memoryProbeParseStatusName(MemoryProbeParseStatus status);

}  // namespace membench

#endif  // MEMBENCH_PLATFORM_MEMORY_PROBE_H
