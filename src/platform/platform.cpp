#include "platform/platform.h"

#include "core/types.h"
#include "platform/memory_probe.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#ifdef __linux__
#include <sched.h>
#endif
#include <time.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#if (defined(__GNUC__) || defined(__clang__)) && defined(__AVX2__)
#include <cpuid.h>
#endif
#endif

namespace membench {
namespace {

#ifdef __APPLE__
std::string commandOutput(const char* command) {
    std::string output;
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) {
        return output;
    }
    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    pclose(pipe);
    return output;
}

std::string labeledValue(const std::string& text,
                         const std::string& label,
                         std::size_t begin = 0,
                         std::size_t end = std::string::npos) {
    const std::size_t position = text.find(label, begin);
    if (position == std::string::npos || position >= end) {
        return {};
    }
    const std::size_t value_begin = position + label.size();
    const std::size_t line_end = text.find('\n', value_begin);
    const std::size_t value_end = std::min(line_end, end);
    const std::size_t first = text.find_first_not_of(" \t", value_begin);
    if (first == std::string::npos || first >= value_end) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(" \t\r", value_end - 1);
    return text.substr(first, last - first + 1);
}

unsigned int parseLeadingUnsigned(const std::string& text) {
    std::uint64_t value = 0;
    std::size_t length = 0;
    while (length < text.size() && text[length] >= '0' && text[length] <= '9') {
        value = value * 10 + static_cast<unsigned int>(text[length] - '0');
        ++length;
    }
    return value <= std::numeric_limits<unsigned int>::max()
               ? static_cast<unsigned int>(value)
               : 0;
}

double applePublishedBandwidth(const std::string& chip, unsigned int gpu_cores) {
    if (chip == "Apple M1") return 68.25;
    if (chip == "Apple M1 Pro") return 200.0;
    if (chip == "Apple M1 Max") return 400.0;
    if (chip == "Apple M1 Ultra") return 800.0;
    if (chip == "Apple M2") return 100.0;
    if (chip == "Apple M2 Pro") return 200.0;
    if (chip == "Apple M2 Max") return 400.0;
    if (chip == "Apple M2 Ultra") return 800.0;
    if (chip == "Apple M3") return 100.0;
    if (chip == "Apple M3 Pro") return 150.0;
    if (chip == "Apple M3 Max") {
        if (gpu_cores == 0) return 0.0;
        return gpu_cores <= 30 ? 300.0 : 400.0;
    }
    if (chip == "Apple M3 Ultra") return 819.0;
    if (chip == "Apple M4") return 120.0;
    if (chip == "Apple M4 Pro") return 273.0;
    if (chip == "Apple M4 Max") {
        if (gpu_cores == 0) return 0.0;
        return gpu_cores <= 32 ? 410.0 : 546.0;
    }
    if (chip == "Apple M5") return 153.0;
    if (chip == "Apple M5 Pro") return 307.0;
    if (chip == "Apple M5 Max") {
        if (gpu_cores == 0) return 0.0;
        return gpu_cores <= 32 ? 460.0 : 614.0;
    }
    return 0.0;
}

void detectAppleMemory(PlatformInfo::MemoryInfo* memory) {
    const std::string profile = commandOutput(
        "LC_ALL=C /usr/sbin/system_profiler SPHardwareDataType SPMemoryDataType "
        "SPDisplaysDataType 2>/dev/null");
    const std::size_t memory_section = profile.find("Memory:");
    const std::size_t graphics_section = profile.find("Graphics/Displays:");
    memory->device_name = labeledValue(profile, "Chip:");
    if (memory_section != std::string::npos) {
        memory->technology = labeledValue(profile, "Type:", memory_section, graphics_section);
        const std::string speed = labeledValue(profile, "Speed:", memory_section, graphics_section);
        memory->transfer_rate_mt_s = parseLeadingUnsigned(speed);
    }
    const unsigned int gpu_cores = graphics_section == std::string::npos
                                       ? 0
                                       : parseLeadingUnsigned(labeledValue(
                                             profile, "Total Number of Cores:", graphics_section));
    memory->theoretical_bandwidth_gb_per_sec =
        applePublishedBandwidth(memory->device_name, gpu_cores);
    if (memory->theoretical_bandwidth_gb_per_sec > 0.0) {
        memory->source = "Apple chip specification";
        memory->bandwidth_is_published = true;
    } else if (!memory->technology.empty() || memory->transfer_rate_mt_s > 0) {
        memory->source = "system_profiler";
    }
}
#endif

bool parseUnsigned64ForPlatform(const std::string& text, std::uint64_t* value) {
    if (value == nullptr || text.empty()) {
        return false;
    }

    std::size_t consumed = 0;
    try {
        const auto parsed = std::stoull(text, &consumed, 10);
        if (consumed != text.size()) {
            return false;
        }
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

#ifdef __linux__
bool containsCpu(const std::vector<unsigned int>& cpus, unsigned int cpu) {
    return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
}

std::vector<unsigned int> readThreadSiblings(unsigned int cpu) {
    const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                             "/topology/thread_siblings_list";
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line)) {
        return {cpu};
    }
    return parseCpuList(line);
}

std::vector<unsigned int> detectPhysicalFirstCpuOrder() {
    cpu_set_t allowed_set;
    CPU_ZERO(&allowed_set);
    if (sched_getaffinity(0, sizeof(allowed_set), &allowed_set) != 0) {
        return {};
    }

    std::vector<unsigned int> allowed;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed_set)) {
            allowed.push_back(static_cast<unsigned int>(cpu));
        }
    }

    std::vector<unsigned int> primary;
    std::vector<unsigned int> secondary;
    for (unsigned int cpu : allowed) {
        const std::vector<unsigned int> siblings = readThreadSiblings(cpu);
        unsigned int first_allowed_sibling = cpu;
        for (unsigned int sibling : siblings) {
            if (containsCpu(allowed, sibling)) {
                first_allowed_sibling = std::min(first_allowed_sibling, sibling);
            }
        }
        if (cpu == first_allowed_sibling) {
            primary.push_back(cpu);
        } else {
            secondary.push_back(cpu);
        }
    }

    primary.insert(primary.end(), secondary.begin(), secondary.end());
    return primary;
}
#endif

}  // namespace

void finalizeMemoryInfo(PlatformInfo::MemoryInfo* memory) {
    if (memory == nullptr || memory->bandwidth_is_published) {
        return;
    }
    if (memory->transfer_rate_mt_s == 0 || memory->aggregate_bus_width_bits == 0) {
        memory->theoretical_bandwidth_gb_per_sec = 0.0;
        return;
    }
    memory->theoretical_bandwidth_gb_per_sec =
        static_cast<double>(memory->transfer_rate_mt_s) *
        static_cast<double>(memory->aggregate_bus_width_bits) / 8000.0;
}

std::vector<unsigned int> parseCpuList(const std::string& text) {
    std::vector<unsigned int> cpus;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto dash = item.find('-');
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
        if (dash == std::string::npos) {
            if (!parseUnsigned64ForPlatform(item, &begin)) {
                continue;
            }
            end = begin;
        } else {
            if (!parseUnsigned64ForPlatform(item.substr(0, dash), &begin) ||
                !parseUnsigned64ForPlatform(item.substr(dash + 1), &end) || end < begin) {
                continue;
            }
        }
        for (std::uint64_t cpu = begin; cpu <= end; ++cpu) {
            if (cpu <= std::numeric_limits<unsigned int>::max()) {
                cpus.push_back(static_cast<unsigned int>(cpu));
            }
        }
    }
    return cpus;
}

PlatformInfo detectPlatformInfo() {
    PlatformInfo info;
#ifdef _WIN32
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    info.page_size = system_info.dwPageSize;
    info.hardware_threads = system_info.dwNumberOfProcessors > 0
                                ? system_info.dwNumberOfProcessors
                                : 1;

    MEMORYSTATUSEX memory_status{};
    memory_status.dwLength = sizeof(memory_status);
    if (GlobalMemoryStatusEx(&memory_status)) {
        info.physical_memory_bytes = memory_status.ullTotalPhys;
    }
    info.memory = probeSystemSmbiosMemory().memory;
#else
    const long page_size = sysconf(_SC_PAGESIZE);
    info.page_size = page_size > 0 ? static_cast<std::size_t>(page_size) : 4096;

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    info.hardware_threads = hw_threads > 0 ? hw_threads : 1;

#if defined(__APPLE__)
    std::uint64_t memsize = 0;
    std::size_t memsize_len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &memsize_len, nullptr, 0) == 0) {
        info.physical_memory_bytes = memsize;
    }

    unsigned int perf_cores = 0;
    std::size_t perf_cores_len = sizeof(perf_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &perf_cores_len, nullptr, 0) ==
        0) {
        info.performance_cores = perf_cores;
    }
#if defined(__aarch64__) || defined(__arm64__)
    info.apple_silicon = true;
#endif
    detectAppleMemory(&info.memory);
#else
    const long phys_pages = sysconf(_SC_PHYS_PAGES);
    if (phys_pages > 0 && page_size > 0) {
        info.physical_memory_bytes =
            static_cast<std::uint64_t>(phys_pages) * static_cast<std::uint64_t>(page_size);
    }

#if defined(__x86_64__) || defined(__i386__)
#if (defined(__GNUC__) || defined(__clang__)) && defined(__AVX2__)
    __builtin_cpu_init();
    info.x86_avx2 = __builtin_cpu_supports("avx2");
#endif
#endif

#ifdef __linux__
    info.memory = probeSystemSmbiosMemory().memory;
#endif
#endif
#endif

#ifdef __linux__
    info.cpu_affinity_order = detectPhysicalFirstCpuOrder();
#endif

    if (info.physical_memory_bytes == 0) {
        info.physical_memory_bytes = 8ULL * GB;
    }
    return info;
}

std::uint64_t monotonicNowNs() {
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
#endif
}

void applyCurrentThreadPolicy(RunMode mode, bool use_qos) {
#ifdef __APPLE__
    if (!use_qos) {
        return;
    }
    const qos_class_t qos =
        mode == RunMode::Peak ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_USER_INITIATED;
    pthread_set_qos_class_self_np(qos, 0);
#else
    (void)mode;
    (void)use_qos;
#endif
}

void applyCurrentThreadAffinity(const std::vector<unsigned int>& cpu_affinity_order,
                                unsigned int worker_index) {
#ifdef __linux__
    if (worker_index >= cpu_affinity_order.size()) {
        return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_affinity_order[worker_index], &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
    (void)cpu_affinity_order;
    (void)worker_index;
#endif
}

}  // namespace membench
