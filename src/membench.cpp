#include "version.h"

#include "app/memory_benchmark.h"
#include "core/format.h"
#include "core/types.h"
#include "platform/platform.h"
#include "planner/execution_planner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MEMBENCH_HAS_METAL
#include "metal_backend.h"
#endif

namespace {

using namespace membench;

bool parseUnsigned64(const std::string& text, std::uint64_t* value) {
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

bool parseThreadPolicy(const std::string& text, ThreadPolicy* policy) {
    if (policy == nullptr) {
        return false;
    }
    if (text == "perf") {
        *policy = ThreadPolicy::Perf;
        return true;
    }
    if (text == "all") {
        *policy = ThreadPolicy::All;
        return true;
    }
    return false;
}

bool parseRunMode(const std::string& text, RunMode* mode) {
    if (mode == nullptr) {
        return false;
    }
    if (text == "standard") {
        *mode = RunMode::Standard;
        return true;
    }
    if (text == "peak") {
        *mode = RunMode::Peak;
        return true;
    }
    return false;
}

bool parseBackend(const std::string& text, Backend* backend) {
    if (backend == nullptr) {
        return false;
    }
    if (text == "cpu") {
        *backend = Backend::Cpu;
        return true;
    }
    if (text == "metal") {
        *backend = Backend::Metal;
        return true;
    }
    if (text == "auto") {
        *backend = Backend::Auto;
        return true;
    }
    return false;
}

bool parseTestList(const std::string& text, std::vector<TestKind>* tests) {
    if (tests == nullptr || text.empty()) {
        return false;
    }

    std::vector<TestKind> parsed;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item == "read") {
            parsed.push_back(TestKind::Read);
        } else if (item == "write") {
            parsed.push_back(TestKind::Write);
        } else if (item == "copy") {
            parsed.push_back(TestKind::Copy);
        } else {
            return false;
        }
    }

    if (parsed.empty()) {
        return false;
    }

    std::sort(parsed.begin(), parsed.end(), [](TestKind lhs, TestKind rhs) {
        return static_cast<int>(lhs) < static_cast<int>(rhs);
    });
    parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());
    *tests = parsed;
    return true;
}

void printUsage(const char* program_name, const PlatformInfo& platform) {
    const std::size_t default_size_mb = chooseDefaultBufferSize(platform) / MB;
    const std::string default_mode = runModeToString(chooseDefaultMode(platform));
    std::cout << "Usage: " << program_name << " [size_mb] [options]\n\n"
              << "Options:\n"
              << "  --size-mb <n>         Buffer size in MiB (default: " << default_size_mb
              << ")\n"
              << "  --threads <n>         Override worker thread count\n"
              << "  --tests <list>        Comma-separated tests: read,write,copy\n"
              << "  --iterations <n>      Measured iterations per test (default: "
              << kDefaultMeasuredIterations << ")\n"
              << "  --warmup <n>          Warmup iterations per test (default: "
              << kDefaultWarmupIterations << ")\n"
              << "  --thread-policy <p>   perf or all\n"
              << "  --mode <m>            standard or peak (default: " << default_mode << ")\n"
              << "  --backend <b>         cpu, metal, or auto (default: "
              << (platform.apple_silicon ? "auto" : "cpu") << ")\n"
              << "  --no-calibrate        Disable peak-mode kernel/thread calibration\n"
              << "  --no-qos              Disable macOS QoS hinting\n"
              << "  --help                Show this message\n\n"
              << "Examples:\n"
              << "  " << program_name << " 1024\n"
              << "  " << program_name << " --mode standard --thread-policy perf\n"
              << "  " << program_name << " --mode peak --size-mb 1024 --tests read,copy\n"
              << "  " << program_name
              << " --threads 4 --warmup 2 --iterations 7 --thread-policy all\n";
}

void printSystemInfo(const PlatformInfo& platform) {
    std::cout << "=== System Information ===\n";
#ifdef _WIN32
    std::cout << "Operating System: Windows\n";
#elif defined(__APPLE__)
    std::cout << "Operating System: macOS\n";
#elif defined(__linux__)
    std::cout << "Operating System: Linux\n";
#else
    std::cout << "Operating System: Unknown\n";
#endif
    std::cout << "Page size: " << platform.page_size << " bytes\n";
    std::cout << "Physical memory: " << formatBytes(platform.physical_memory_bytes) << '\n';
    std::cout << "Hardware threads: " << platform.hardware_threads << '\n';
    if (platform.apple_silicon) {
        if (platform.performance_cores > 0) {
            std::cout << "Performance cores: " << platform.performance_cores << '\n';
        } else {
            std::cout << "Performance cores: unavailable (falling back to conservative default)\n";
        }
#ifdef MEMBENCH_HAS_METAL
        if (metalIsAvailable()) {
            std::cout << "Metal GPU: " << metalDeviceName() << '\n';
        } else {
            std::cout << "Metal GPU: unavailable\n";
        }
#endif
    }
    if (platform.x86_avx2) {
        std::cout << "x86 AVX2: available\n";
    }
    if (!platform.cpu_affinity_order.empty()) {
        std::cout << "CPU affinity order: " << platform.cpu_affinity_order.size()
                  << " logical CPUs, physical cores first\n";
    }
    std::cout << '\n';
}



}  // namespace

int main(int argc, char* argv[]) {
    const PlatformInfo platform = detectPlatformInfo();

    BenchmarkOptions options;
    options.size_bytes = chooseDefaultBufferSize(platform);
    options.warmup_iterations = kDefaultWarmupIterations;
    options.measured_iterations = kDefaultMeasuredIterations;
    options.mode = chooseDefaultMode(platform);
    options.thread_policy = platform.apple_silicon ? ThreadPolicy::All : ThreadPolicy::All;
    options.calibrate = platform.apple_silicon || platform.x86_avx2;
    options.use_qos = platform.apple_silicon;
#ifdef MEMBENCH_HAS_METAL
    options.backend = platform.apple_silicon ? Backend::Auto : Backend::Cpu;
#else
    options.backend = Backend::Cpu;
#endif
    options.tests = {TestKind::Read, TestKind::Write, TestKind::Copy};

    bool positional_size_consumed = false;
    bool mode_explicit = false;
    bool thread_policy_explicit = false;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            auto requireValue = [&](const std::string& option_name) -> std::string {
                if (index + 1 >= argc) {
                    throw std::runtime_error("missing value for " + option_name);
                }
                ++index;
                return argv[index];
            };

            if (arg == "--help") {
                printUsage(argv[0], platform);
                return 0;
            }
            if (arg == "--size-mb") {
                std::uint64_t size_mb = 0;
                if (!parseUnsigned64(requireValue(arg), &size_mb) || size_mb == 0 ||
                    size_mb > (kMaxBufferSize / MB)) {
                    throw std::runtime_error("invalid --size-mb value");
                }
                options.size_bytes = static_cast<std::size_t>(size_mb) * MB;
                continue;
            }
            if (arg == "--threads") {
                std::uint64_t threads = 0;
                if (!parseUnsigned64(requireValue(arg), &threads) || threads == 0 ||
                    threads > std::numeric_limits<unsigned int>::max()) {
                    throw std::runtime_error("invalid --threads value");
                }
                options.threads_override = static_cast<unsigned int>(threads);
                continue;
            }
            if (arg == "--iterations") {
                std::uint64_t iterations = 0;
                if (!parseUnsigned64(requireValue(arg), &iterations) || iterations == 0) {
                    throw std::runtime_error("invalid --iterations value");
                }
                options.measured_iterations = static_cast<std::size_t>(iterations);
                continue;
            }
            if (arg == "--warmup") {
                std::uint64_t warmup = 0;
                if (!parseUnsigned64(requireValue(arg), &warmup)) {
                    throw std::runtime_error("invalid --warmup value");
                }
                options.warmup_iterations = static_cast<std::size_t>(warmup);
                continue;
            }
            if (arg == "--tests") {
                if (!parseTestList(requireValue(arg), &options.tests)) {
                    throw std::runtime_error("invalid --tests list");
                }
                continue;
            }
            if (arg == "--thread-policy") {
                if (!parseThreadPolicy(requireValue(arg), &options.thread_policy)) {
                    throw std::runtime_error("invalid --thread-policy value");
                }
                thread_policy_explicit = true;
                continue;
            }
            if (arg == "--mode") {
                if (!parseRunMode(requireValue(arg), &options.mode)) {
                    throw std::runtime_error("invalid --mode value");
                }
                mode_explicit = true;
                continue;
            }
            if (arg == "--backend") {
                if (!parseBackend(requireValue(arg), &options.backend)) {
                    throw std::runtime_error("invalid --backend value (use cpu, metal, or auto)");
                }
                continue;
            }
            if (arg == "--no-calibrate") {
                options.calibrate = false;
                continue;
            }
            if (arg == "--no-qos") {
                options.use_qos = false;
                continue;
            }
            if (!arg.empty() && arg.front() == '-') {
                throw std::runtime_error("unknown option: " + arg);
            }
            if (positional_size_consumed) {
                throw std::runtime_error("unexpected positional argument: " + arg);
            }

            std::uint64_t size_mb = 0;
            if (!parseUnsigned64(arg, &size_mb) || size_mb == 0 ||
                size_mb > (kMaxBufferSize / MB)) {
                throw std::runtime_error("invalid buffer size argument");
            }
            options.size_bytes = static_cast<std::size_t>(size_mb) * MB;
            positional_size_consumed = true;
        }

        if (platform.apple_silicon && mode_explicit && options.mode == RunMode::Standard &&
            !thread_policy_explicit) {
            options.thread_policy = ThreadPolicy::Perf;
        }
        if (options.mode == RunMode::Standard) {
            if (!mode_explicit && platform.apple_silicon) {
                options.thread_policy = ThreadPolicy::All;
            }
            if (!platform.apple_silicon) {
                options.calibrate = false;
            }
        } else if (!platform.apple_silicon && !platform.x86_avx2) {
            options.calibrate = false;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "MemBench v" << MEMBENCH_VERSION << '\n';
    std::cout << "Memory Read/Write/Copy Benchmark\n";
    std::cout << "========================================\n\n";

    printSystemInfo(platform);

    try {
        MemoryBenchmark benchmark(platform, options);
        benchmark.printConfiguration();
        benchmark.run();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
