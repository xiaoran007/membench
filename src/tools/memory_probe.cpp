#include "platform/memory_probe.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using namespace membench;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [--smbios-file <path>]\n\n"
              << "Without --smbios-file, read the system SMBIOS source on Linux or Windows.\n"
              << "With --smbios-file, parse a captured raw DMI table on any platform.\n";
}

std::string valueOrUnavailable(const std::string& value) {
    return value.empty() ? "unavailable" : value;
}

void printDevice(const MemoryDeviceProbe& device) {
    std::cout << "\nDevice " << device.structure_index << ":\n"
              << "  structure offset: " << device.structure_offset << " bytes\n"
              << "  structure length: " << device.structure_length << " bytes\n"
              << "  populated: " << (device.populated ? "yes" : "no") << '\n'
              << "  raw memory type: 0x" << std::hex << std::uppercase
              << static_cast<unsigned int>(device.raw_memory_type) << std::dec
              << std::nouppercase << '\n'
              << "  memory type: " << valueOrUnavailable(device.technology) << '\n';
    if (device.data_width_bits > 0) {
        std::cout << "  data width: " << device.data_width_bits << " bits\n";
    } else {
        std::cout << "  data width: unavailable\n";
    }
    if (device.speed_mt_s > 0) {
        std::cout << "  nominal speed: " << device.speed_mt_s << " MT/s\n";
    } else {
        std::cout << "  nominal speed: unavailable\n";
    }
    if (device.configured_speed_mt_s > 0) {
        std::cout << "  configured speed: " << device.configured_speed_mt_s << " MT/s\n";
    } else {
        std::cout << "  configured speed: unavailable\n";
    }
    if (device.effective_rate_mt_s > 0) {
        std::cout << "  selected effective rate: " << device.effective_rate_mt_s << " MT/s\n";
    } else {
        std::cout << "  selected effective rate: unavailable\n";
    }
}

void printSummary(const MemoryProbeResult& result) {
    std::size_t populated_devices = 0;
    for (const MemoryDeviceProbe& device : result.devices) {
        if (device.populated) {
            ++populated_devices;
        }
    }

    std::cout << "\nSummary:\n"
              << "  populated devices: " << populated_devices << '\n'
              << "  memory type: " << valueOrUnavailable(result.memory.technology) << '\n';
    if (result.memory.transfer_rate_mt_s > 0) {
        std::cout << "  common effective rate: " << result.memory.transfer_rate_mt_s << " MT/s\n";
    } else {
        std::cout << "  common effective rate: unavailable";
        if (!result.rate_reason.empty()) {
            std::cout << " (" << result.rate_reason << ')';
        }
        std::cout << '\n';
    }
    if (result.memory.aggregate_bus_width_bits > 0) {
        std::cout << "  aggregate bus width: " << result.memory.aggregate_bus_width_bits
                  << " bits\n";
    } else {
        std::cout << "  aggregate bus width: unavailable";
        if (!result.aggregate_width_reason.empty()) {
            std::cout << " (" << result.aggregate_width_reason << ')';
        }
        std::cout << '\n';
    }
    if (result.memory.theoretical_bandwidth_gb_per_sec > 0.0) {
        std::cout << "  theoretical peak: " << std::fixed << std::setprecision(2)
                  << result.memory.theoretical_bandwidth_gb_per_sec << " GB/s\n";
    } else {
        std::cout << "  theoretical peak: unavailable\n";
    }
}

int resultExitCode(const MemoryProbeResult& result) {
    if (result.read_status != MemoryProbeReadStatus::Success ||
        result.parse_status == MemoryProbeParseStatus::Malformed ||
        result.parse_status == MemoryProbeParseStatus::NotAttempted) {
        return 1;
    }
    return result.memory.theoretical_bandwidth_gb_per_sec > 0.0 ? 0 : 2;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string input_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (argument == "--smbios-file") {
            if (index + 1 >= argc) {
                std::cerr << "Error: missing path for --smbios-file\n";
                return 1;
            }
            input_path = argv[++index];
            continue;
        }
        std::cerr << "Error: unknown argument: " << argument << '\n';
        return 1;
    }

    const MemoryProbeResult result = input_path.empty()
                                         ? probeSystemSmbiosMemory()
                                         : probeMemoryFromSmbiosFile(input_path);

    std::cout << "MemBench SMBIOS Memory Probe\n"
              << "Source: " << result.source << '\n'
              << "Read status: " << memoryProbeReadStatusName(result.read_status) << '\n';
    if (!result.read_error.empty()) {
        std::cout << "Read detail: " << result.read_error << '\n';
    }
    std::cout << "Bytes read: " << result.bytes_read << '\n'
              << "Parse status: " << memoryProbeParseStatusName(result.parse_status) << '\n';
    if (!result.parse_error.empty()) {
        std::cout << "Parse detail: " << result.parse_error << '\n';
    }
    std::cout << "Structures seen: " << result.structures_seen << '\n'
              << "End marker seen: " << (result.end_marker_seen ? "yes" : "no") << '\n'
              << "Type 17 records: " << result.devices.size() << '\n';

    for (const MemoryDeviceProbe& device : result.devices) {
        printDevice(device);
    }
    printSummary(result);
    return resultExitCode(result);
}
