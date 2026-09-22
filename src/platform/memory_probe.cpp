#include "platform/memory_probe.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#endif

namespace membench {
namespace {

std::uint16_t readLe16(const std::vector<std::uint8_t>& data, std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset]) |
           (static_cast<std::uint16_t>(data[offset + 1]) << 8U);
}

std::uint32_t readLe32(const std::vector<std::uint8_t>& data, std::size_t offset) {
    return static_cast<std::uint32_t>(data[offset]) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 8U) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16U) |
           (static_cast<std::uint32_t>(data[offset + 3]) << 24U);
}

std::string smbiosMemoryType(std::uint8_t type) {
    switch (type) {
        case 0x12: return "DDR";
        case 0x13: return "DDR2";
        case 0x18: return "DDR3";
        case 0x1A: return "DDR4";
        case 0x1B: return "LPDDR";
        case 0x1C: return "LPDDR2";
        case 0x1D: return "LPDDR3";
        case 0x1E: return "LPDDR4";
        case 0x1F: return "Logical non-volatile device";
        case 0x22: return "DDR5";
        case 0x23: return "LPDDR5";
        default: return {};
    }
}

std::uint64_t speedField(const std::vector<std::uint8_t>& table,
                         std::size_t offset,
                         std::size_t length,
                         std::size_t legacy_offset,
                         std::size_t extended_offset) {
    if (length < legacy_offset + sizeof(std::uint16_t)) {
        return 0;
    }
    std::uint64_t speed = readLe16(table, offset + legacy_offset);
    if (speed == 0xFFFFU && length >= extended_offset + sizeof(std::uint32_t)) {
        speed = readLe32(table, offset + extended_offset);
    }
    return speed == 0xFFFFU ? 0 : speed;
}

void summarizeDevices(MemoryProbeResult* result) {
    std::size_t populated_devices = 0;
    unsigned int single_device_width = 0;
    std::uint64_t common_rate = 0;
    bool rates_match = true;

    for (const MemoryDeviceProbe& device : result->devices) {
        if (!device.populated) {
            continue;
        }
        ++populated_devices;
        single_device_width = device.data_width_bits;

        if (result->memory.technology.empty()) {
            result->memory.technology = device.technology;
        } else if (!device.technology.empty() &&
                   result->memory.technology != device.technology) {
            result->memory.technology = "mixed";
        }

        if (device.effective_rate_mt_s > 0) {
            if (common_rate == 0) {
                common_rate = device.effective_rate_mt_s;
            } else if (common_rate != device.effective_rate_mt_s) {
                rates_match = false;
            }
        }
    }

    if (populated_devices == 0) {
        result->rate_reason = "no populated SMBIOS Type 17 memory devices";
        result->aggregate_width_reason = result->rate_reason;
    } else if (!rates_match) {
        result->rate_reason = "populated devices report different effective transfer rates";
    } else if (common_rate == 0) {
        result->rate_reason = "no populated device reports a usable configured or nominal speed";
    } else {
        result->memory.transfer_rate_mt_s = common_rate;
    }

    if (populated_devices == 1 && single_device_width > 0) {
        result->memory.aggregate_bus_width_bits = single_device_width;
    } else if (populated_devices == 1) {
        result->aggregate_width_reason = "the populated device does not report its data width";
    } else if (populated_devices > 1) {
        result->aggregate_width_reason =
            "SMBIOS reports per-device widths but not the active memory-channel topology";
    }

    result->memory.source = populated_devices > 0 ? "SMBIOS" : "";
    if (result->memory.transfer_rate_mt_s > 0 &&
        result->memory.aggregate_bus_width_bits > 0) {
        result->memory.theoretical_bandwidth_gb_per_sec =
            static_cast<double>(result->memory.transfer_rate_mt_s) *
            static_cast<double>(result->memory.aggregate_bus_width_bits) / 8000.0;
    }
}

MemoryProbeReadStatus readStatusFromErrno(int error) {
    if (error == ENOENT) {
        return MemoryProbeReadStatus::NotFound;
    }
    if (error == EACCES || error == EPERM) {
        return MemoryProbeReadStatus::PermissionDenied;
    }
    return MemoryProbeReadStatus::IoError;
}

#ifdef _WIN32
MemoryProbeResult probeWindowsFirmwareTable() {
    MemoryProbeResult result;
    result.source = "Windows raw SMBIOS firmware table";
    const DWORD size = GetSystemFirmwareTable('RSMB', 0, nullptr, 0);
    if (size <= 8) {
        result.read_status = MemoryProbeReadStatus::IoError;
        result.read_error = "GetSystemFirmwareTable did not return a raw SMBIOS table";
        return result;
    }

    std::vector<std::uint8_t> raw(size);
    if (GetSystemFirmwareTable('RSMB', 0, raw.data(), size) != size) {
        result.read_status = MemoryProbeReadStatus::IoError;
        result.read_error = "GetSystemFirmwareTable failed while reading the table";
        return result;
    }
    const std::uint32_t table_size = readLe32(raw, 4);
    if (table_size == 0 || table_size > size - 8) {
        result.read_status = MemoryProbeReadStatus::IoError;
        result.read_error = "raw SMBIOS header contains an invalid table length";
        return result;
    }
    return probeMemoryFromSmbiosBytes(
        {raw.begin() + 8, raw.begin() + 8 + table_size}, result.source);
}
#endif

}  // namespace

MemoryProbeResult probeMemoryFromSmbiosBytes(const std::vector<std::uint8_t>& table,
                                             const std::string& source) {
    MemoryProbeResult result;
    result.source = source;
    result.bytes_read = table.size();
    result.read_status = table.empty() ? MemoryProbeReadStatus::Empty
                                       : MemoryProbeReadStatus::Success;
    if (table.empty()) {
        result.read_error = "SMBIOS input is empty";
        return result;
    }

    std::size_t offset = 0;
    while (offset + 4 <= table.size()) {
        const std::uint8_t type = table[offset];
        const std::size_t length = table[offset + 1];
        if (length < 4 || offset + length > table.size()) {
            result.parse_status = MemoryProbeParseStatus::Malformed;
            result.parse_error = "invalid SMBIOS structure length at byte offset " +
                                 std::to_string(offset);
            break;
        }

        ++result.structures_seen;
        if (type == 17) {
            MemoryDeviceProbe device;
            device.structure_index = result.devices.size();
            device.structure_offset = offset;
            device.structure_length = length;
            if (length >= 14) {
                device.populated = readLe16(table, offset + 12) != 0;
            }
            if (length >= 19) {
                device.raw_memory_type = table[offset + 18];
                device.technology = smbiosMemoryType(device.raw_memory_type);
            }
            if (length >= 12) {
                const std::uint16_t width = readLe16(table, offset + 10);
                device.data_width_bits = width == 0xFFFFU ? 0 : width;
            }
            device.speed_mt_s = speedField(table, offset, length, 21, 84);
            device.configured_speed_mt_s = speedField(table, offset, length, 32, 88);
            device.effective_rate_mt_s = device.configured_speed_mt_s > 0
                                             ? device.configured_speed_mt_s
                                             : device.speed_mt_s;
            result.devices.push_back(device);
        }

        std::size_t next = offset + length;
        while (next + 1 < table.size() && (table[next] != 0 || table[next + 1] != 0)) {
            ++next;
        }
        if (next + 1 >= table.size()) {
            result.parse_status = MemoryProbeParseStatus::Malformed;
            result.parse_error = "unterminated SMBIOS string area after byte offset " +
                                 std::to_string(offset);
            break;
        }
        offset = next + 2;
        if (type == 127) {
            result.end_marker_seen = true;
            result.parse_status = MemoryProbeParseStatus::Success;
            break;
        }
    }

    if (result.parse_status == MemoryProbeParseStatus::NotAttempted) {
        result.parse_status = result.devices.empty()
                                  ? MemoryProbeParseStatus::NoMemoryDevices
                                  : MemoryProbeParseStatus::Success;
    } else if (result.parse_status == MemoryProbeParseStatus::Success &&
               result.devices.empty()) {
        result.parse_status = MemoryProbeParseStatus::NoMemoryDevices;
    }
    summarizeDevices(&result);
    return result;
}

MemoryProbeResult probeMemoryFromSmbiosFile(const std::string& path) {
    errno = 0;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        const int error = errno;
        MemoryProbeResult result;
        result.source = path;
        result.read_status = readStatusFromErrno(error);
        result.read_error = error == 0 ? "failed to open SMBIOS input"
                                       : std::strerror(error);
        return result;
    }

    std::vector<std::uint8_t> table{std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>()};
    if (file.bad()) {
        MemoryProbeResult result;
        result.source = path;
        result.read_status = MemoryProbeReadStatus::IoError;
        result.read_error = "I/O error while reading SMBIOS input";
        result.bytes_read = table.size();
        return result;
    }
    return probeMemoryFromSmbiosBytes(table, path);
}

MemoryProbeResult probeSystemSmbiosMemory() {
#ifdef _WIN32
    return probeWindowsFirmwareTable();
#elif defined(__linux__)
    return probeMemoryFromSmbiosFile("/sys/firmware/dmi/tables/DMI");
#else
    MemoryProbeResult result;
    result.source = "system SMBIOS";
    result.read_status = MemoryProbeReadStatus::NotAttempted;
    result.read_error = "system SMBIOS probing is only available on Linux and Windows";
    return result;
#endif
}

const char* memoryProbeReadStatusName(MemoryProbeReadStatus status) {
    switch (status) {
        case MemoryProbeReadStatus::NotAttempted: return "not attempted";
        case MemoryProbeReadStatus::Success: return "success";
        case MemoryProbeReadStatus::NotFound: return "not found";
        case MemoryProbeReadStatus::PermissionDenied: return "permission denied";
        case MemoryProbeReadStatus::IoError: return "I/O error";
        case MemoryProbeReadStatus::Empty: return "empty input";
    }
    return "unknown";
}

const char* memoryProbeParseStatusName(MemoryProbeParseStatus status) {
    switch (status) {
        case MemoryProbeParseStatus::NotAttempted: return "not attempted";
        case MemoryProbeParseStatus::Success: return "success";
        case MemoryProbeParseStatus::Malformed: return "malformed input";
        case MemoryProbeParseStatus::NoMemoryDevices: return "no Type 17 memory devices";
    }
    return "unknown";
}

}  // namespace membench
