#ifndef MEMBENCH_REPORTING_BENCHMARK_REPORTER_H
#define MEMBENCH_REPORTING_BENCHMARK_REPORTER_H

#include "core/format.h"
#include "core/statistics.h"
#include "core/types.h"
#include "kernels/kernel_registry.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef MEMBENCH_HAS_METAL
#include "metal_backend.h"
#endif

namespace membench {

enum class OutputMode {
    Auto,
    Plain,
    Tui,
};

enum class TuiStyle {
    Auto,
    Unicode,
    Ascii,
};

enum class TestPhase {
    Pending,
    Calibrating,
    Measuring,
    Done,
};

struct BenchmarkSummaryEntry {
    TestKind kind;
    ExecutionPlan plan;
    unsigned int actual_threads = 0;
    double avg_bandwidth_gb_per_sec = 0.0;
    double avg_traffic_gb_per_sec = 0.0;
};

struct CalibrationProgress {
    TestKind kind = TestKind::Read;
    KernelKind kernel = KernelKind::ScalarAuto;
    unsigned int threads = 0;
    std::size_t completed = 0;
    std::size_t total = 0;
    double best_score_mb_per_sec = 0.0;
    KernelKind best_kernel = KernelKind::ScalarAuto;
    unsigned int best_threads = 0;
};

inline bool stdoutIsTerminal() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

inline bool enableAnsiTerminal() {
#ifdef _WIN32
    HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    if (output == INVALID_HANDLE_VALUE) {
        return false;
    }
    DWORD mode = 0;
    if (!GetConsoleMode(output, &mode)) {
        return false;
    }
    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    return SetConsoleMode(output, mode) != 0;
#else
    return true;
#endif
}

inline bool environmentLooksUtf8() {
#ifdef _WIN32
    return true;
#else
    const char* vars[] = {std::getenv("LC_ALL"), std::getenv("LC_CTYPE"), std::getenv("LANG")};
    for (const char* value : vars) {
        if (value == nullptr) {
            continue;
        }
        std::string text(value);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (text.find("utf-8") != std::string::npos || text.find("utf8") != std::string::npos) {
            return true;
        }
    }
    return false;
#endif
}

inline std::string operatingSystemName() {
#ifdef _WIN32
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "Unknown";
#endif
}

class BenchmarkReporter {
public:
    virtual ~BenchmarkReporter() = default;

    virtual void beginRun(const std::string& version,
                          const PlatformInfo& platform,
                          const BenchmarkOptions& options) = 0;
    virtual void beginAllocation(std::size_t size_bytes, std::size_t alignment) = 0;
    virtual void finishInitialization() = 0;
    virtual void configuration(const PlatformInfo& platform,
                               const BenchmarkOptions& options,
                               std::size_t alignment,
                               bool calibration_enabled) = 0;
    virtual void beginCalibration(TestKind kind, std::size_t total_candidates) = 0;
    virtual void calibrationCandidate(const CalibrationProgress& progress) = 0;
    virtual void calibrationSelected(TestKind kind, const CalibrationCandidate& selected) = 0;
    virtual void beginTest(TestKind kind, const ExecutionPlan& plan) = 0;
    virtual void testCompleted(TestKind kind,
                               const ExecutionPlan& plan,
                               unsigned int actual_threads,
                               const TestResult& result) = 0;
    virtual void summary(const std::vector<BenchmarkSummaryEntry>& entries) = 0;
    virtual void finishRun() = 0;
};

class PlainReporter final : public BenchmarkReporter {
public:
    void beginRun(const std::string& version,
                  const PlatformInfo& platform,
                  const BenchmarkOptions& options) override {
        (void)options;
        setOptions(options);
        std::cout << "========================================\n";
        std::cout << "MemBench v" << version << '\n';
        std::cout << "Memory Read/Write/Copy Benchmark\n";
        std::cout << "========================================\n\n";

        std::cout << "=== System Information ===\n";
        std::cout << "Operating System: " << operatingSystemName() << '\n';
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

    void beginAllocation(std::size_t size_bytes, std::size_t alignment) override {
        (void)alignment;
        std::cout << "Allocating " << size_bytes / MB << " MiB per buffer..." << std::endl;
    }

    void finishInitialization() override {
        std::cout << "Buffers initialized with deterministic non-zero data.\n\n";
    }

    void configuration(const PlatformInfo& platform,
                       const BenchmarkOptions& options,
                       std::size_t alignment,
                       bool calibration_enabled) override {
        std::cout << "=== Benchmark Configuration ===\n";
        std::cout << "Size: " << options.size_bytes / MB << " MiB\n";
        std::cout << "Alignment: " << alignment << " bytes\n";
        std::cout << "Mode: " << runModeToString(options.mode) << '\n';
        std::cout << "Backend: " << backendToString(options.backend) << '\n';
        std::cout << "Calibration: " << (calibration_enabled ? "enabled" : "disabled") << '\n';
        std::cout << "Warmup iterations: " << options.warmup_iterations << '\n';
        std::cout << "Measured iterations: " << options.measured_iterations << '\n';
        std::cout << "Thread policy: " << threadPolicyToString(options.thread_policy) << '\n';
        if (options.threads_override > 0) {
            std::cout << "Thread override: " << options.threads_override << '\n';
        }
        std::cout << "Worker affinity: "
                  << (!platform.cpu_affinity_order.empty() ? "physical-first" : "disabled")
                  << '\n';
        std::cout << "macOS QoS hint: " << (options.use_qos ? "enabled" : "disabled") << '\n';
        std::cout << "Tests: ";
        for (std::size_t index = 0; index < options.tests.size(); ++index) {
            if (index > 0) {
                std::cout << ',';
            }
            std::cout << testKindToCliName(options.tests[index]);
        }
        std::cout << "\n\n";
    }

    void beginCalibration(TestKind kind, std::size_t total_candidates) override {
        if (total_candidates > 0) {
            std::cout << "Calibrating " << testKindToTitle(kind) << " ("
                      << total_candidates << " candidates)...\n";
        }
    }

    void calibrationCandidate(const CalibrationProgress& progress) override {
        std::cout << "\r  candidate " << progress.completed << '/'
                  << progress.total << ": " << kernelToString(progress.kernel);
        if (!isMetalKernel(progress.kernel)) {
            std::cout << ", " << progress.threads << " threads";
        }
        std::cout << std::flush;
    }

    void calibrationSelected(TestKind, const CalibrationCandidate& selected) override {
        std::cout << "\r  selected " << kernelToString(selected.kernel);
        if (!isMetalKernel(selected.kernel)) {
            std::cout << ", " << selected.actual_threads << " threads";
        }
        std::cout << " at " << std::fixed << std::setprecision(2)
                  << (selected.score_mb_per_sec / 1024.0) << " GB/s"
                  << "                      \n";
    }

    void beginTest(TestKind, const ExecutionPlan&) override {}

    void testCompleted(TestKind kind,
                       const ExecutionPlan& plan,
                       unsigned int actual_threads,
                       const TestResult& result) override {
        std::cout << "=== " << testKindToTitle(kind) << " Test ===\n";
        std::cout << "mode: " << runModeToString(options_mode_) << '\n';
        std::cout << "kernel: " << kernelToString(plan.kernel) << '\n';
        if (isMetalKernel(plan.kernel)) {
            std::cout << "selected_threads: gpu\n";
        } else {
            std::cout << "selected_threads: " << actual_threads << '\n';
        }
        std::cout << "calibrated: " << (plan.calibrated ? "yes" : "no") << '\n';
        std::cout << "size_mb: " << options_size_bytes_ / MB << '\n';
        std::cout << "warmup: " << options_warmup_iterations_ << '\n';
        std::cout << "iterations: " << options_measured_iterations_ << '\n';
        std::cout << "logical_bytes_per_iteration: " << result.logical_bytes_per_iteration << " ("
                  << formatBytes(result.logical_bytes_per_iteration) << ")\n";
        std::cout << "measured_elapsed_ms: avg " << std::fixed << std::setprecision(3)
                  << result.elapsed_ms.average << ", median " << result.elapsed_ms.median << '\n';

        if (kind == TestKind::Copy) {
            printBandwidthStats("logical ", result.bandwidth_mb_per_sec);
            const Statistics traffic = scaleStatistics(result.bandwidth_mb_per_sec, 2.0);
            printBandwidthStats("estimated traffic ", traffic);
            std::cout << "Estimated traffic bandwidth is logical memcpy throughput multiplied by 2.\n";
        } else {
            printBandwidthStats("", result.bandwidth_mb_per_sec);
        }
        std::cout << '\n';
    }

    void summary(const std::vector<BenchmarkSummaryEntry>& entries) override {
        if (entries.size() <= 1) {
            return;
        }
        std::cout << "=== Summary ===\n";
        for (const auto& e : entries) {
            std::ostringstream line;
            line << std::fixed << std::setprecision(2);
            line << "  " << testKindToTitle(e.kind) << ":  ";
            if (e.kind == TestKind::Copy) {
                line << e.avg_bandwidth_gb_per_sec << " GB/s logical, "
                     << e.avg_traffic_gb_per_sec << " GB/s traffic";
            } else {
                line << e.avg_bandwidth_gb_per_sec << " GB/s";
            }
            line << "  (" << kernelToString(e.plan.kernel);
            if (isMetalKernel(e.plan.kernel)) {
                line << ", gpu";
            } else {
                line << ", " << e.actual_threads << " threads";
            }
            if (e.plan.calibrated) {
                line << ", calibrated";
            }
            line << ")";
            std::cout << line.str() << '\n';
        }
        std::cout << '\n';
    }

    void finishRun() override {}

    void setOptions(const BenchmarkOptions& options) {
        options_mode_ = options.mode;
        options_size_bytes_ = options.size_bytes;
        options_warmup_iterations_ = options.warmup_iterations;
        options_measured_iterations_ = options.measured_iterations;
    }

private:
    void printBandwidthStats(const std::string& prefix, const Statistics& stats) const {
        std::cout << std::setprecision(2);
        std::cout << prefix << "avg bandwidth: " << (stats.average / 1024.0) << " GB/s ("
                  << stats.average << " MB/s)\n";
        std::cout << prefix << "median bandwidth: " << (stats.median / 1024.0) << " GB/s ("
                  << stats.median << " MB/s)\n";
        std::cout << prefix << "min bandwidth: " << (stats.minimum / 1024.0) << " GB/s ("
                  << stats.minimum << " MB/s)\n";
        std::cout << prefix << "max bandwidth: " << (stats.maximum / 1024.0) << " GB/s ("
                  << stats.maximum << " MB/s)\n";
        std::cout << prefix << "stdev bandwidth: " << (stats.stdev / 1024.0) << " GB/s ("
                  << stats.stdev << " MB/s)\n";
    }

    RunMode options_mode_ = RunMode::Standard;
    std::size_t options_size_bytes_ = 0;
    std::size_t options_warmup_iterations_ = 0;
    std::size_t options_measured_iterations_ = 0;
};

class TuiReporter final : public BenchmarkReporter {
public:
    explicit TuiReporter(bool unicode) : unicode_(unicode) {}

    ~TuiReporter() override {
        leaveAlternateScreen();
        showCursor();
    }

    void beginRun(const std::string& version,
                  const PlatformInfo& platform,
                  const BenchmarkOptions& options) override {
        version_ = version;
        platform_ = platform;
        options_ = options;
        start_time_ns_ = monotonicNowNsFallback();
        enterAlternateScreen();
        hideCursor();
        render(true);
    }

    void beginAllocation(std::size_t size_bytes, std::size_t alignment) override {
        allocation_text_ = "allocating " + std::to_string(size_bytes / MB) +
                           " MiB per buffer, alignment " + std::to_string(alignment) + " bytes";
        render(true);
    }

    void finishInitialization() override {
        allocation_text_ = "buffers initialized with deterministic non-zero data";
        render(true);
    }

    void configuration(const PlatformInfo& platform,
                       const BenchmarkOptions& options,
                       std::size_t alignment,
                       bool calibration_enabled) override {
        platform_ = platform;
        options_ = options;
        alignment_ = alignment;
        calibration_enabled_ = calibration_enabled;
        render(true);
    }

    void beginCalibration(TestKind kind, std::size_t total_candidates) override {
        TestState& state = testState(kind);
        state.phase = TestPhase::Calibrating;
        state.status = "0/" + std::to_string(total_candidates);
        state.progress_current = 0;
        state.progress_total = total_candidates;
        focused_test_ = kind;
        focused_title_ = testKindToTitle(kind);
        focused_detail_ = "calibrating candidates";
        render(true);
    }

    void calibrationCandidate(const CalibrationProgress& progress) override {
        TestState& state = testState(progress.kind);
        state.phase = TestPhase::Calibrating;
        state.kernel = progress.kernel;
        state.threads = progress.threads;
        state.status = std::to_string(progress.completed) + "/" + std::to_string(progress.total);
        state.progress_current = progress.completed;
        state.progress_total = progress.total;
        state.best_kernel = progress.best_kernel;
        state.best_threads = progress.best_threads;
        state.best_gb_per_sec = progress.best_score_mb_per_sec / 1024.0;
        focused_test_ = progress.kind;
        focused_title_ = testKindToTitle(progress.kind);
        focused_detail_ = "current " + kernelToString(progress.kernel) +
                          threadText(progress.kernel, progress.threads);
        render(true);
    }

    void calibrationSelected(TestKind kind, const CalibrationCandidate& selected) override {
        TestState& state = testState(kind);
        state.kernel = selected.kernel;
        state.threads = selected.actual_threads;
        state.best_kernel = selected.kernel;
        state.best_threads = selected.actual_threads;
        state.best_gb_per_sec = selected.score_mb_per_sec / 1024.0;
        state.status = "selected";
        focused_test_ = kind;
        focused_title_ = testKindToTitle(kind);
        focused_detail_ = "selected " + kernelToString(selected.kernel) +
                          threadText(selected.kernel, selected.actual_threads);
        render(true);
    }

    void beginTest(TestKind kind, const ExecutionPlan& plan) override {
        TestState& state = testState(kind);
        state.phase = TestPhase::Measuring;
        state.kernel = plan.kernel;
        state.threads = plan.selected_threads;
        state.status = "running";
        state.progress_current = 0;
        state.progress_total = options_.measured_iterations;
        focused_test_ = kind;
        focused_title_ = testKindToTitle(kind);
        focused_detail_ = "measuring " + kernelToString(plan.kernel) +
                          threadText(plan.kernel, plan.selected_threads);
        render(true);
    }

    void testCompleted(TestKind kind,
                       const ExecutionPlan& plan,
                       unsigned int actual_threads,
                       const TestResult& result) override {
        TestState& state = testState(kind);
        state.phase = TestPhase::Done;
        state.kernel = plan.kernel;
        state.threads = actual_threads;
        state.status = "done";
        state.progress_current = options_.measured_iterations;
        state.progress_total = options_.measured_iterations;
        state.has_result = true;
        state.plan = plan;
        state.result = result;
        state.actual_threads = actual_threads;
        focused_test_ = kind;
        focused_title_ = testKindToTitle(kind);
        focused_detail_ = "completed";
        render(true);
    }

    void summary(const std::vector<BenchmarkSummaryEntry>& entries) override {
        summary_ = entries;
        render(true);
    }

    void finishRun() override {
        finished_ = true;
        finish_time_ns_ = monotonicNowNsFallback();
        render(true);
        leaveAlternateScreen();
        showCursor();
        render(false);
    }

private:
    struct TestState {
        TestPhase phase = TestPhase::Pending;
        KernelKind kernel = KernelKind::ScalarAuto;
        unsigned int threads = 0;
        std::string status = "-";
        std::size_t progress_current = 0;
        std::size_t progress_total = 0;
        bool has_result = false;
        ExecutionPlan plan;
        TestResult result;
        unsigned int actual_threads = 0;
        KernelKind best_kernel = KernelKind::ScalarAuto;
        unsigned int best_threads = 0;
        double best_gb_per_sec = 0.0;
    };

    struct Theme {
        std::string tl;
        std::string tr;
        std::string bl;
        std::string br;
        std::string h;
        std::string v;
        std::string lt;
        std::string rt;
        std::string tt;
        std::string bt;
        std::string cross;
        std::string bar_full;
        std::string bar_empty;
    };

    TestState& testState(TestKind kind) {
        switch (kind) {
            case TestKind::Read:  return read_;
            case TestKind::Write: return write_;
            case TestKind::Copy:  return copy_;
        }
        return read_;
    }

    const TestState& testState(TestKind kind) const {
        switch (kind) {
            case TestKind::Read:  return read_;
            case TestKind::Write: return write_;
            case TestKind::Copy:  return copy_;
        }
        return read_;
    }

    Theme theme() const {
        if (unicode_) {
            return {"┌", "┐", "└", "┘", "─", "│", "├", "┤", "┬", "┴", "┼", "█", "░"};
        }
        return {"+", "+", "+", "+", "-", "|", "+", "+", "+", "+", "+", "#", "-"};
    }

    void hideCursor() {
        std::cout << "\x1b[?25l";
    }

    void showCursor() {
        std::cout << "\x1b[?25h" << std::flush;
    }

    void enterAlternateScreen() {
        if (!alternate_screen_) {
            std::cout << "\x1b[?1049h";
            alternate_screen_ = true;
        }
    }

    void leaveAlternateScreen() {
        if (alternate_screen_) {
            std::cout << "\x1b[?1049l";
            alternate_screen_ = false;
        }
    }

    static std::uint64_t monotonicNowNsFallback() {
        using Clock = std::chrono::steady_clock;
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now().time_since_epoch())
                .count());
    }

    static std::string truncate(const std::string& text, std::size_t width) {
        if (text.size() <= width) {
            return text + std::string(width - text.size(), ' ');
        }
        if (width <= 1) {
            return text.substr(0, width);
        }
        return text.substr(0, width - 1) + "~";
    }

    static std::string fixed(double value, int precision) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << value;
        return oss.str();
    }

    std::string color(const std::string& code, const std::string& text) const {
        return "\x1b[" + code + "m" + text + "\x1b[0m";
    }

    std::string phaseText(TestPhase phase) const {
        switch (phase) {
            case TestPhase::Pending:     return "pending";
            case TestPhase::Calibrating: return "calibrating";
            case TestPhase::Measuring:   return "measuring";
            case TestPhase::Done:        return "done";
        }
        return "unknown";
    }

    std::string testLabel(TestKind kind) const {
        switch (kind) {
            case TestKind::Read:  return "Read";
            case TestKind::Write: return "Write";
            case TestKind::Copy:  return "Copy";
        }
        return "Unknown";
    }

    std::string threadText(KernelKind kernel, unsigned int threads) const {
        if (isMetalKernel(kernel)) {
            return ", gpu";
        }
        if (threads == 0) {
            return "";
        }
        return ", " + std::to_string(threads) + " threads";
    }

    std::string progressBar(std::size_t current, std::size_t total, std::size_t width) const {
        const Theme t = theme();
        std::size_t filled = 0;
        if (total > 0) {
            filled = std::min(width, (current * width) / total);
        }
        return std::string(filled, '\0').replace(0, filled, repeat(t.bar_full, filled)) +
               repeat(t.bar_empty, width - filled);
    }

    static std::string repeat(const std::string& text, std::size_t count) {
        std::string out;
        for (std::size_t i = 0; i < count; ++i) {
            out += text;
        }
        return out;
    }

    void line(const std::string& text = "") const {
        std::cout << text << '\n';
    }

    void boxTop(const std::string& title = "") const {
        const Theme t = theme();
        std::string body = " " + title + " ";
        const std::size_t visible = title.empty() ? 0 : title.size() + 2;
        if (title.empty()) {
            body.clear();
        }
        std::cout << t.tl << body << repeat(t.h, width_ - visible - 2) << t.tr << '\n';
    }

    void boxBottom() const {
        const Theme t = theme();
        std::cout << t.bl << repeat(t.h, width_ - 2) << t.br << '\n';
    }

    void boxLine(const std::string& text) const {
        const Theme t = theme();
        std::cout << t.v << " " << truncate(text, width_ - 4) << " " << t.v << '\n';
    }

    void render(bool clear_screen) {
        if (clear_screen) {
            std::cout << "\x1b[H\x1b[2J";
        }
        renderHeader();
        line();
        renderConfiguration();
        line();
        renderProgressTable();
        line();
        renderFocus();
        line();
        renderSamples();
        line();
        renderResults();
        if (finished_) {
            line();
            renderSummary();
        }
        std::cout << std::flush;
    }

    void renderHeader() {
        const std::string title = "MemBench v" + version_;
        boxTop(title);
        std::ostringstream sys;
        sys << operatingSystemName() << " | ";
        if (platform_.apple_silicon) {
            sys << "Apple Silicon";
        } else if (platform_.x86_avx2) {
            sys << "x86 AVX2";
        } else {
            sys << "portable CPU";
        }
        sys << " | " << formatBytes(platform_.physical_memory_bytes)
            << " | " << platform_.hardware_threads << " threads"
            << " | " << (finished_ ? elapsedSecondsText() + " done" : "running");
        boxLine(sys.str());
        boxBottom();
    }

    void renderConfiguration() {
        line(color("1;36", "Configuration"));
        boxTop();
        std::ostringstream tests;
        for (std::size_t index = 0; index < options_.tests.size(); ++index) {
            if (index > 0) {
                tests << ',';
            }
            tests << testKindToCliName(options_.tests[index]);
        }
        boxLine("Size " + std::to_string(options_.size_bytes / MB) + " MiB   Mode " +
                runModeToString(options_.mode) + "   Backend " + backendToString(options_.backend) +
                "   Tests " + tests.str());
        boxLine("Warmup " + std::to_string(options_.warmup_iterations) +
                "        Iterations " + std::to_string(options_.measured_iterations) +
                "             Calibration " + (calibration_enabled_ ? "enabled" : "disabled"));
        boxLine("Thread policy " + threadPolicyToString(options_.thread_policy) +
                "        QoS " + std::string(options_.use_qos ? "enabled" : "disabled") +
                "        Affinity " +
                (!platform_.cpu_affinity_order.empty() ? "physical-first" : "disabled"));
        if (!allocation_text_.empty()) {
            boxLine(allocation_text_);
        }
        boxBottom();
    }

    void renderProgressTable() {
        line(color("1;36", "Progress"));
        const Theme t = theme();
        boxTop();
        boxLine("Test    Phase         Kernel                    Threads    Status");
        std::cout << t.lt << repeat(t.h, width_ - 2) << t.rt << '\n';
        progressRow(TestKind::Read);
        progressRow(TestKind::Write);
        progressRow(TestKind::Copy);
        boxBottom();
    }

    void progressRow(TestKind kind) {
        const TestState& state = testState(kind);
        std::string threads = "-";
        if (state.phase != TestPhase::Pending) {
            threads = isMetalKernel(state.kernel) ? "gpu" : std::to_string(state.threads);
        }
        boxLine(truncate(testLabel(kind), 7) +
                truncate(phaseText(state.phase), 14) +
                truncate(state.phase == TestPhase::Pending ? "-" : kernelToString(state.kernel), 26) +
                truncate(threads, 11) +
                state.status);
    }

    void renderFocus() {
        line(color("1;36", focused_title_));
        const TestState& state = testState(focused_test_);
        const std::size_t total = state.progress_total == 0 ? 1 : state.progress_total;
        const std::size_t current = std::min(state.progress_current, total);
        const int percent = static_cast<int>((current * 100) / total);
        line("  " + focused_detail_);
        line("  " + progressBar(current, total, 40) + "  " + std::to_string(percent) + "%");
        if (state.best_gb_per_sec > 0.0) {
            line("  selected so far " + kernelToString(state.best_kernel) +
                 threadText(state.best_kernel, state.best_threads) +
                 ", " + fixed(state.best_gb_per_sec, 2) + " GB/s");
        } else if (state.has_result) {
            line("  median " + fixed(state.result.bandwidth_mb_per_sec.median / 1024.0, 2) +
                 " GB/s   best " + fixed(state.result.bandwidth_mb_per_sec.maximum / 1024.0, 2) +
                 " GB/s");
        } else {
            line("  waiting for first sample");
        }
    }

    void renderResults() {
        line(color("1;36", "Results"));
        const Theme t = theme();
        boxTop();
        boxLine("Test    Avg          Median       Kernel         Threads   Notes");
        std::cout << t.lt << repeat(t.h, width_ - 2) << t.rt << '\n';
        resultRow(TestKind::Read);
        resultRow(TestKind::Write);
        resultRow(TestKind::Copy);
        boxBottom();
    }

    void renderSamples() {
        line(color("1;36", "Live Samples"));
        boxTop();
        sampleRow(TestKind::Read);
        sampleRow(TestKind::Write);
        sampleRow(TestKind::Copy);
        boxBottom();
    }

    void sampleRow(TestKind kind) {
        const TestState& state = testState(kind);
        if (!state.has_result) {
            boxLine(truncate(testLabel(kind), 8) + "-");
            return;
        }
        const double avg = state.result.bandwidth_mb_per_sec.average / 1024.0;
        const double median = state.result.bandwidth_mb_per_sec.median / 1024.0;
        const double best = state.result.bandwidth_mb_per_sec.maximum / 1024.0;
        boxLine(truncate(testLabel(kind), 8) +
                "avg " + fixed(avg, 2) + " GB/s   median " +
                fixed(median, 2) + " GB/s   best " + fixed(best, 2) + " GB/s");
    }

    void resultRow(TestKind kind) {
        const TestState& state = testState(kind);
        if (!state.has_result) {
            boxLine(truncate(testLabel(kind), 8) + truncate("pending", 13) +
                    truncate("pending", 13) + truncate("-", 15) + truncate("-", 10) + "-");
            return;
        }
        const double avg = state.result.bandwidth_mb_per_sec.average / 1024.0;
        const double median = state.result.bandwidth_mb_per_sec.median / 1024.0;
        const std::string threads = isMetalKernel(state.plan.kernel)
                                        ? "gpu"
                                        : std::to_string(state.actual_threads);
        boxLine(truncate(testLabel(kind), 8) +
                truncate(fixed(avg, 2) + " GB/s", 13) +
                truncate(fixed(median, 2) + " GB/s", 13) +
                truncate(kernelToString(state.plan.kernel), 15) +
                truncate(threads, 10) +
                (state.plan.calibrated ? "calibrated" : "-"));
        if (kind == TestKind::Copy) {
            boxLine("        traffic estimate " +
                    fixed((state.result.bandwidth_mb_per_sec.average * 2.0) / 1024.0, 2) +
                    " GB/s");
        }
    }

    void renderSummary() {
        if (summary_.empty()) {
            return;
        }
        line(color("1;36", "Summary"));
        for (const auto& entry : summary_) {
            std::ostringstream oss;
            oss << "  " << testKindToTitle(entry.kind) << ": ";
            if (entry.kind == TestKind::Copy) {
                oss << fixed(entry.avg_bandwidth_gb_per_sec, 2) << " GB/s logical, "
                    << fixed(entry.avg_traffic_gb_per_sec, 2) << " GB/s traffic";
            } else {
                oss << fixed(entry.avg_bandwidth_gb_per_sec, 2) << " GB/s";
            }
            oss << "  (" << kernelToString(entry.plan.kernel);
            if (isMetalKernel(entry.plan.kernel)) {
                oss << ", gpu";
            } else {
                oss << ", " << entry.actual_threads << " threads";
            }
            if (entry.plan.calibrated) {
                oss << ", calibrated";
            }
            oss << ")";
            line(oss.str());
        }
    }

    std::string elapsedSecondsText() const {
        const std::uint64_t end = finish_time_ns_ == 0 ? monotonicNowNsFallback() : finish_time_ns_;
        const double seconds = static_cast<double>(end - start_time_ns_) / 1'000'000'000.0;
        return fixed(seconds, 2) + "s";
    }

    bool unicode_ = true;
    bool alternate_screen_ = false;
    const std::size_t width_ = 78;
    std::string version_;
    PlatformInfo platform_;
    BenchmarkOptions options_;
    std::size_t alignment_ = 0;
    bool calibration_enabled_ = false;
    bool finished_ = false;
    std::uint64_t start_time_ns_ = 0;
    std::uint64_t finish_time_ns_ = 0;
    std::string allocation_text_;
    TestKind focused_test_ = TestKind::Read;
    std::string focused_title_ = "Read";
    std::string focused_detail_ = "pending";
    TestState read_;
    TestState write_;
    TestState copy_;
    std::vector<BenchmarkSummaryEntry> summary_;
};

inline bool shouldUseTui(OutputMode mode) {
    if (mode == OutputMode::Plain) {
        return false;
    }
    if (!stdoutIsTerminal()) {
        return false;
    }
    if (!enableAnsiTerminal()) {
        return false;
    }
    return true;
}

inline bool shouldUseUnicodeTui(TuiStyle style) {
    if (style == TuiStyle::Ascii) {
        return false;
    }
    if (style == TuiStyle::Unicode) {
        return true;
    }
    return environmentLooksUtf8();
}

}  // namespace membench

#endif  // MEMBENCH_REPORTING_BENCHMARK_REPORTER_H
