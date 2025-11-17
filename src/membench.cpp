/**
 * Memory Benchmark Tool
 * Tests memory read, write, copy speed
 * Cross-platform support: Linux, macOS, Windows
 */

#include "version.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <random>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>
#include <numeric>

#ifdef _WIN32
    #include <windows.h>
    #include <intrin.h>
#else
    #include <unistd.h>
    #include <pthread.h>
    #if defined(__x86_64__) || defined(__i386__)
        #include <x86intrin.h>
    #elif defined(__aarch64__) || defined(__arm__)
        #include <arm_neon.h>
    #endif
    #if defined(__linux__)
        #include <sched.h>
        #include <sys/resource.h>
    #elif defined(__APPLE__)
        #include <mach/thread_policy.h>
        #include <mach/thread_act.h>
        #include <sys/resource.h>
    #endif
#endif

class MemoryBenchmark {
private:
    static constexpr size_t KB = 1024;
    static constexpr size_t MB = 1024 * KB;
    static constexpr size_t GB = 1024 * MB;
    static constexpr size_t DEFAULT_SIZE = 2 * GB;  // Increased to 2GB to exceed LLC
    static constexpr size_t ITERATIONS = 10;
    static constexpr size_t INNER_ITERATIONS = 5;  // Reduced inner iterations due to larger dataset
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t UNROLL_FACTOR = 16;

    std::vector<uint8_t> buffer1;
    std::vector<uint8_t> buffer2;
    unsigned int num_threads;

    // High-resolution timer
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::nanoseconds;

    // Prevent compiler optimization
    template<typename T>
    inline void doNotOptimize(T const& value) {
#if defined(__clang__) || defined(__GNUC__)
        asm volatile("" : : "r,m"(value) : "memory");
#else
        // For MSVC
        _ReadWriteBarrier();
        volatile T tmp = value;
        _ReadWriteBarrier();
        (void)tmp;
#endif
    }
    
    // Memory barrier to prevent reordering
    inline void memoryBarrier() {
#if defined(__clang__) || defined(__GNUC__)
        asm volatile("" : : : "memory");
#else
        _ReadWriteBarrier();
#endif
    }

    // Get current timestamp
    TimePoint now() {
        return Clock::now();
    }

    // Calculate bandwidth in MB/s
    double calculateBandwidth(size_t bytes, Duration duration) {
        double seconds = std::chrono::duration<double>(duration).count();
        return (bytes / (1024.0 * 1024.0)) / seconds;
    }

    // Set thread affinity and priority
    bool setThreadAffinity(unsigned int core_id) {
#if defined(__linux__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_t current_thread = pthread_self();
        return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(__APPLE__)
        // macOS thread affinity is different and often fails
        // Use affinity tag instead (groups threads that should run together)
        thread_affinity_policy_data_t policy = { static_cast<integer_t>(core_id + 1) };
        thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
        kern_return_t result = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                                                (thread_policy_t)&policy, 
                                                THREAD_AFFINITY_POLICY_COUNT);
        return result == KERN_SUCCESS;
#elif defined(_WIN32)
        DWORD_PTR mask = 1ULL << core_id;
        return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
        (void)core_id;
        return false;
#endif
    }

    // Set high priority (not real-time to avoid system issues)
    bool setHighPriority() {
#if defined(__linux__)
        // Use nice value instead of real-time for better compatibility
        return setpriority(PRIO_PROCESS, 0, -20) == 0;
#elif defined(__APPLE__)
        // Set time constraint policy for better scheduling
        thread_time_constraint_policy_data_t policy;
        policy.period = 0;
        policy.computation = 0;
        policy.constraint = 0;
        policy.preemptible = 1;
        
        thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
        kern_return_t result = thread_policy_set(mach_thread,
                                                THREAD_TIME_CONSTRAINT_POLICY,
                                                (thread_policy_t)&policy,
                                                THREAD_TIME_CONSTRAINT_POLICY_COUNT);
        
        // If that fails, just try to increase priority
        if (result != KERN_SUCCESS) {
            struct sched_param param;
            param.sched_priority = sched_get_priority_max(SCHED_OTHER);
            return pthread_setschedparam(pthread_self(), SCHED_OTHER, &param) == 0;
        }
        return result == KERN_SUCCESS;
#elif defined(_WIN32)
        if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
            return false;
        }
        return SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) != 0;
#else
        return false;
#endif
    }

    // Determine optimal thread count for memory benchmarking
    static unsigned int getOptimalThreadCount() {
        unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) {
            return 1;  // Fallback if unable to detect
        }
        
        // For memory bandwidth tests, using too many threads can cause:
        // 1. Memory controller saturation (diminishing returns)
        // 2. Cache coherency overhead
        // 3. System instability on high-core-count machines
        
        // Strategy:
        // - For <= 8 cores: use all cores (typical consumer CPUs)
        // - For 9-16 cores: use 75% of cores
        // - For 17-32 cores: use 50% of cores
        // - For 33-64 cores: use 25% of cores (16 threads max)
        // - For > 64 cores: cap at 16-24 threads
        
        unsigned int optimal;
        if (hw_threads <= 8) {
            optimal = hw_threads;
        } else if (hw_threads <= 16) {
            // optimal = hw_threads;  // not change
            optimal = 8;
        } else if (hw_threads <= 32) {
            // optimal = hw_threads / 2;  // 50%
            optimal = 8;
        } else if (hw_threads <= 64) {
            // optimal = hw_threads / 4;  // 25%, max 16
            optimal = 8;
        } else {
            // For very high core count (>64), cap at 16-24 threads
            // optimal = std::min(24u, hw_threads / 4);
            optimal = 8;
        }
        
        // Ensure at least 1 thread
        if (optimal < 1) optimal = 1;
        
        std::cout << "Hardware threads detected: " << hw_threads 
                  << ", using " << optimal << " threads for benchmark" << std::endl;
        
        return optimal;
    }

public:
    MemoryBenchmark(size_t size = DEFAULT_SIZE) 
        : num_threads(getOptimalThreadCount()) {
        std::cout << "Allocating " << size / MB << " MB (" << size / GB << " GB) memory buffers..." << std::endl;
        buffer1.resize(size);
        buffer2.resize(size);
        
        // Initialize with random data in parallel
        std::vector<std::thread> init_threads;
        size_t chunk_size = size / num_threads;
        
        for (unsigned int t = 0; t < num_threads; ++t) {
            init_threads.emplace_back([this, t, chunk_size, size]() {
                std::random_device rd;
                std::mt19937 gen(rd() + t);
                std::uniform_int_distribution<int> dis(0, 255);
                
                size_t start = t * chunk_size;
                size_t end = (t == num_threads - 1) ? size : (t + 1) * chunk_size;
                
                for (size_t i = start; i < end; ++i) {
                    buffer1[i] = static_cast<uint8_t>(dis(gen));
                }
            });
        }
        
        for (auto& thread : init_threads) {
            thread.join();
        }
        
        std::cout << "Memory initialized with " << num_threads << " threads." << std::endl;
    }

    // Sequential read test with autovectorization (compiler-optimized)
    void testSequentialRead() {
        std::cout << "\n=== Sequential Read Test (Multi-threaded, Auto-vectorized) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB (" << buffer1.size() / GB << " GB)" << std::endl;
        std::cout << "Threads: " << num_threads << std::endl;
        std::cout << "Total data per iteration: " << (buffer1.size() * INNER_ITERATIONS) / MB << " MB" << std::endl;
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            std::atomic<bool> start_flag{false};
            std::vector<std::thread> threads;
            std::vector<double> thread_bandwidths(num_threads);
            
            size_t chunk_size = buffer1.size() / num_threads;
            
            // Launch threads
            for (unsigned int t = 0; t < num_threads; ++t) {
                threads.emplace_back([this, t, chunk_size, &start_flag, &thread_bandwidths]() {
                    // Try to set thread affinity (silently fail if not supported)
                    setThreadAffinity(t);
                    // Try to set high priority
                    setHighPriority();
                    
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? buffer1.size() : (t + 1) * chunk_size;
                    size_t local_size = end - start;
                    
                    const uint64_t* data = reinterpret_cast<const uint64_t*>(buffer1.data() + start);
                    size_t count = local_size / sizeof(uint64_t);
                    
                    // Warmup
                    uint64_t warmup_sum = 0;
                    for (size_t i = 0; i < count; ++i) {
                        warmup_sum += data[i];
                    }
                    doNotOptimize(warmup_sum);
                    
                    // Wait for all threads to be ready
                    while (!start_flag.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    
                    memoryBarrier();
                    auto start_time = now();
                    
                    // Multiple passes with simple reduction pattern for autovectorization
                    for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                        uint64_t sum = 0;
                        // Simple loop that compilers can easily autovectorize
                        // Stride by cache line (8 uint64_t = 64 bytes) for better performance
                        for (size_t i = 0; i < count; i += 8) {
                            sum += data[i];
                        }
                        doNotOptimize(sum);
                    }
                    
                    auto end_time = now();
                    memoryBarrier();
                    
                    auto duration = std::chrono::duration_cast<Duration>(end_time - start_time);
                    thread_bandwidths[t] = calculateBandwidth(local_size * INNER_ITERATIONS, duration);
                });
            }
            
            // Small delay to ensure all threads are ready
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Start all threads simultaneously
            start_flag.store(true, std::memory_order_release);
            
            // Wait for all threads
            for (auto& thread : threads) {
                thread.join();
            }
            
            // Sum up bandwidth from all threads
            double total_bandwidth = std::accumulate(thread_bandwidths.begin(), 
                                                    thread_bandwidths.end(), 0.0);
            bandwidths.push_back(total_bandwidth);
        }
        
        printStatistics(bandwidths);
    }

    // Sequential write test with memset (optimized by compiler/library)
    void testSequentialWrite() {
        std::cout << "\n=== Sequential Write Test (Multi-threaded memset) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB (" << buffer1.size() / GB << " GB)" << std::endl;
        std::cout << "Threads: " << num_threads << std::endl;
        std::cout << "Total data per iteration: " << (buffer1.size() * INNER_ITERATIONS) / MB << " MB" << std::endl;
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            std::atomic<bool> start_flag{false};
            std::vector<std::thread> threads;
            std::vector<double> thread_bandwidths(num_threads);
            
            size_t chunk_size = buffer1.size() / num_threads;
            
            for (unsigned int t = 0; t < num_threads; ++t) {
                threads.emplace_back([this, t, chunk_size, &start_flag, &thread_bandwidths]() {
                    setThreadAffinity(t);
                    setHighPriority();
                    
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? buffer1.size() : (t + 1) * chunk_size;
                    size_t local_size = end - start;
                    
                    // Warmup
                    std::memset(buffer1.data() + start, 0, local_size);
                    
                    while (!start_flag.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    
                    memoryBarrier();
                    auto start_time = now();
                    
                    for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                        std::memset(buffer1.data() + start, 0xAA, local_size);
                    }
                    
                    auto end_time = now();
                    memoryBarrier();
                    
                    auto duration = std::chrono::duration_cast<Duration>(end_time - start_time);
                    thread_bandwidths[t] = calculateBandwidth(local_size * INNER_ITERATIONS, duration);
                });
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            start_flag.store(true, std::memory_order_release);
            
            for (auto& thread : threads) {
                thread.join();
            }
            
            double total_bandwidth = std::accumulate(thread_bandwidths.begin(), 
                                                    thread_bandwidths.end(), 0.0);
            bandwidths.push_back(total_bandwidth);
        }
        
        printStatistics(bandwidths);
    }

    // Memory copy test (using memcpy) with multi-threading
    void testMemoryCopy() {
        std::cout << "\n=== Memory Copy Test (Multi-threaded memcpy) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB (" << buffer1.size() / GB << " GB)" << std::endl;
        std::cout << "Threads: " << num_threads << std::endl;
        std::cout << "Total data per iteration: " << (buffer1.size() * INNER_ITERATIONS) / MB << " MB" << std::endl;
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            std::atomic<bool> start_flag{false};
            std::vector<std::thread> threads;
            std::vector<double> thread_bandwidths(num_threads);
            
            size_t chunk_size = buffer1.size() / num_threads;
            
            for (unsigned int t = 0; t < num_threads; ++t) {
                threads.emplace_back([this, t, chunk_size, &start_flag, &thread_bandwidths]() {
                    setThreadAffinity(t);
                    setHighPriority();
                    
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? buffer1.size() : (t + 1) * chunk_size;
                    size_t local_size = end - start;
                    
                    // Warmup
                    std::memcpy(buffer2.data() + start, buffer1.data() + start, local_size);
                    
                    while (!start_flag.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    
                    memoryBarrier();
                    auto start_time = now();
                    
                    for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                        std::memcpy(buffer2.data() + start, buffer1.data() + start, local_size);
                    }
                    
                    auto end_time = now();
                    memoryBarrier();
                    
                    auto duration = std::chrono::duration_cast<Duration>(end_time - start_time);
                    // Calculate bandwidth based on actual data copied (not read+write)
                    thread_bandwidths[t] = calculateBandwidth(local_size * INNER_ITERATIONS, duration);
                });
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            start_flag.store(true, std::memory_order_release);
            
            for (auto& thread : threads) {
                thread.join();
            }
            
            double total_bandwidth = std::accumulate(thread_bandwidths.begin(), 
                                                    thread_bandwidths.end(), 0.0);
            bandwidths.push_back(total_bandwidth);
        }
        
        printStatistics(bandwidths);
    }

    // Print system information
    static void printSystemInfo() {
        std::cout << "=== System Information ===" << std::endl;
        
#ifdef _WIN32
        std::cout << "Operating System: Windows" << std::endl;
        
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        std::cout << "Page size: " << sysInfo.dwPageSize << " bytes" << std::endl;
        std::cout << "Number of processors: " << sysInfo.dwNumberOfProcessors << std::endl;
        
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        std::cout << "Total physical memory: " << memInfo.ullTotalPhys / (1024 * 1024) << " MB" << std::endl;
#elif __APPLE__
        std::cout << "Operating System: macOS" << std::endl;
        std::cout << "Page size: " << getpagesize() << " bytes" << std::endl;
#elif __linux__
        std::cout << "Operating System: Linux" << std::endl;
        std::cout << "Page size: " << getpagesize() << " bytes" << std::endl;
        
        // Try to read memory info
        std::cout << "CPU cores: " << sysconf(_SC_NPROCESSORS_ONLN) << std::endl;
#else
        std::cout << "Operating System: Unknown" << std::endl;
#endif
        
        std::cout << std::endl;
    }

private:
    void printStatistics(const std::vector<double>& bandwidths) {
        double avg = calculateAverage(bandwidths);
        double min = *std::min_element(bandwidths.begin(), bandwidths.end());
        double max = *std::max_element(bandwidths.begin(), bandwidths.end());
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Avg bandwidth: " << (avg / 1024.0) << " GB/s (" << avg << " MB/s)" << std::endl;
        std::cout << "Min bandwidth: " << (min / 1024.0) << " GB/s (" << min << " MB/s)" << std::endl;
        std::cout << "Max bandwidth: " << (max / 1024.0) << " GB/s (" << max << " MB/s)" << std::endl;
    }

    double calculateAverage(const std::vector<double>& values) {
        double sum = 0.0;
        for (auto v : values) {
            sum += v;
        }
        return sum / values.size();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════╗" << std::endl;
    std::cout << "║   Memory Benchmark Tool v" << MEMBENCH_VERSION << "       ║" << std::endl;
    std::cout << "║   Multi-threaded Memory Testing      ║" << std::endl;
    std::cout << "║   With SIMD & Loop Optimizations     ║" << std::endl;
    std::cout << "╚══════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    MemoryBenchmark::printSystemInfo();

    // Parse command line arguments for buffer size
    size_t bufferSize = 2ULL * 1024 * 1024 * 1024; // Default 2GB
    if (argc > 1) {
        try {
            std::stringstream ss(argv[1]);
            unsigned long long sizeMB = 0;
            ss >> sizeMB;
            if (!ss.fail() && sizeMB > 0 && sizeMB <= 16384) { // Max 16GB
                bufferSize = sizeMB * 1024 * 1024;
                std::cout << "Using custom buffer size: " << bufferSize / (1024 * 1024) << " MB" << std::endl;
            } else {
                std::cerr << "Invalid buffer size (must be 1-16384 MB), using default 2048MB" << std::endl;
            }
        } catch (...) {
            std::cerr << "Invalid buffer size argument, using default 2048MB" << std::endl;
        }
    }
    
    std::cout << "\nNote: This benchmark uses multi-threading and high priority scheduling." << std::endl;
    std::cout << "For best results:" << std::endl;
    std::cout << "  - Close other applications to minimize interference" << std::endl;
    std::cout << "  - On Linux, run with sudo for better thread scheduling" << std::endl;
    std::cout << std::endl;

    MemoryBenchmark benchmark(bufferSize);

    // Bandwidth tests (with multi-threading)
    benchmark.testSequentialRead();
    benchmark.testSequentialWrite();
    benchmark.testMemoryCopy();
    
    // std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
    // std::cout << "║          BENCHMARK COMPLETE                       ║" << std::endl;
    // std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
