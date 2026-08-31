#include "platform/platform.h"

#include "core/types.h"

#include <algorithm>
#include <chrono>
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
