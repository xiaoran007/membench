#include "core/format.h"
#include "core/statistics.h"
#include "core/types.h"
#include "platform/platform.h"
#include "planner/execution_planner.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void expectNear(double actual, double expected, double tolerance, const std::string& message) {
    expect(std::fabs(actual - expected) <= tolerance,
           message + " (actual " + std::to_string(actual) +
               ", expected " + std::to_string(expected) + ")");
}

void testStatistics() {
    const membench::Statistics stats = membench::calculateStatistics({1.0, 4.0, 2.0, 3.0});
    expectNear(stats.average, 2.5, 1e-12, "average");
    expectNear(stats.median, 2.5, 1e-12, "median");
    expectNear(stats.minimum, 1.0, 1e-12, "minimum");
    expectNear(stats.maximum, 4.0, 1e-12, "maximum");

    const membench::Statistics scaled = membench::scaleStatistics(stats, 2.0);
    expectNear(scaled.average, 5.0, 1e-12, "scaled average");
    expectNear(scaled.median, 5.0, 1e-12, "scaled median");
}

void testFormatting() {
    expect(membench::formatBytes(membench::MB) == "1.00 MiB", "MiB formatting");
    expect(membench::testKindToCliName(membench::TestKind::Copy) == "copy", "test CLI name");
    expect(membench::kernelToString(membench::KernelKind::LibcMemcpy) == "libc_memcpy",
           "kernel string");
}

void testCpuListParsing() {
    const std::vector<unsigned int> cpus = membench::parseCpuList("0-2,4,7-8");
    const std::vector<unsigned int> expected{0, 1, 2, 4, 7, 8};
    expect(cpus == expected, "CPU range parsing");

    const std::vector<unsigned int> partial = membench::parseCpuList("bad,3,5-4,6");
    const std::vector<unsigned int> partial_expected{3, 6};
    expect(partial == partial_expected, "invalid CPU list items are ignored");
}

void testPlannerDefaults() {
    membench::PlatformInfo platform;
    platform.physical_memory_bytes = 16ULL * membench::GB;
    platform.hardware_threads = 8;

    expect(membench::chooseDefaultBufferSize(platform) == membench::GB,
           "default buffer is capped at 1 GiB");
    expect(membench::chooseDefaultThreadCount(platform, membench::ThreadPolicy::All) == 8,
           "default all-thread count");
    expect(membench::chooseDefaultMode(platform) == membench::RunMode::Standard,
           "portable default mode");

    platform.apple_silicon = true;
    platform.performance_cores = 4;
    expect(membench::chooseDefaultMode(platform) == membench::RunMode::Peak,
           "Apple Silicon default mode");
    expect(membench::chooseDefaultThreadCount(platform, membench::ThreadPolicy::Perf) == 4,
           "Apple Silicon perf-thread count");
}

}  // namespace

int main() {
    testStatistics();
    testFormatting();
    testCpuListParsing();
    testPlannerDefaults();
    std::cout << "unit tests passed\n";
    return 0;
}
