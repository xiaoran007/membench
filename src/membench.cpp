/**
 * Memory Benchmark Tool
 * Tests memory read, write, copy speed and latency
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
    static constexpr size_t LATENCY_ITERATIONS = 1000000;
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

    // Calculate latency in nanoseconds
    double calculateLatency(size_t iterations, Duration duration) {
        return std::chrono::duration<double, std::nano>(duration).count() / iterations;
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

public:
    MemoryBenchmark(size_t size = DEFAULT_SIZE) 
        : num_threads(std::thread::hardware_concurrency()) {
        if (num_threads == 0) num_threads = 1;
        
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

    // Sequential read test with loop unrolling and multi-threading
    void testSequentialRead() {
        std::cout << "\n=== Sequential Read Test (Multi-threaded, Loop Unrolled) ===" << std::endl;
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
                    
                    // Multiple passes
                    for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                        uint64_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                        uint64_t sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
                        uint64_t sum8 = 0, sum9 = 0, sumA = 0, sumB = 0;
                        uint64_t sumC = 0, sumD = 0, sumE = 0, sumF = 0;
                        
                        size_t i = 0;
                        for (; i + UNROLL_FACTOR <= count; i += UNROLL_FACTOR) {
                            sum0 += data[i + 0];
                            sum1 += data[i + 1];
                            sum2 += data[i + 2];
                            sum3 += data[i + 3];
                            sum4 += data[i + 4];
                            sum5 += data[i + 5];
                            sum6 += data[i + 6];
                            sum7 += data[i + 7];
                            sum8 += data[i + 8];
                            sum9 += data[i + 9];
                            sumA += data[i + 10];
                            sumB += data[i + 11];
                            sumC += data[i + 12];
                            sumD += data[i + 13];
                            sumE += data[i + 14];
                            sumF += data[i + 15];
                        }
                        
                        for (; i < count; ++i) {
                            sum0 += data[i];
                        }
                        
                        uint64_t total = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 +
                                        sum8 + sum9 + sumA + sumB + sumC + sumD + sumE + sumF;
                        doNotOptimize(total);
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

    // Sequential write test with loop unrolling and multi-threading
    void testSequentialWrite() {
        std::cout << "\n=== Sequential Write Test (Multi-threaded, Loop Unrolled) ===" << std::endl;
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
                    
                    uint64_t* data = reinterpret_cast<uint64_t*>(buffer1.data() + start);
                    size_t count = local_size / sizeof(uint64_t);
                    
                    // Use volatile to prevent over-optimization but allow write combining
                    const uint64_t v0 = 0xDEADBEEFCAFEBABEULL;
                    const uint64_t v1 = 0xFEEDFACEDEADC0DEULL;
                    const uint64_t v2 = 0xBADDCAFEBABEFACEULL;
                    const uint64_t v3 = 0xC0DEDBADDEADFEEDULL;
                    
                    // Warmup
                    for (size_t i = 0; i < count; i += UNROLL_FACTOR) {
                        if (i + UNROLL_FACTOR <= count) {
                            data[i] = data[i + 1] = data[i + 2] = data[i + 3] = 0;
                            data[i + 4] = data[i + 5] = data[i + 6] = data[i + 7] = 0;
                            data[i + 8] = data[i + 9] = data[i + 10] = data[i + 11] = 0;
                            data[i + 12] = data[i + 13] = data[i + 14] = data[i + 15] = 0;
                        }
                    }
                    
                    while (!start_flag.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    
                    memoryBarrier();
                    auto start_time = now();
                    
                    for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                        // More aggressive unrolling with pointer arithmetic
                        uint64_t* ptr = data;
                        uint64_t* end_ptr = data + count;
                        
                        while (ptr + UNROLL_FACTOR <= end_ptr) {
                            // Using pointer writes can be slightly faster
                            ptr[0] = v0;
                            ptr[1] = v1;
                            ptr[2] = v2;
                            ptr[3] = v3;
                            ptr[4] = v0;
                            ptr[5] = v1;
                            ptr[6] = v2;
                            ptr[7] = v3;
                            ptr[8] = v0;
                            ptr[9] = v1;
                            ptr[10] = v2;
                            ptr[11] = v3;
                            ptr[12] = v0;
                            ptr[13] = v1;
                            ptr[14] = v2;
                            ptr[15] = v3;
                            ptr += UNROLL_FACTOR;
                        }
                        
                        // Handle remaining elements
                        while (ptr < end_ptr) {
                            *ptr++ = v0;
                        }
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

    // Random access latency test
    void testRandomAccessLatency() {
        std::cout << "\n=== Random Access Latency Test ===" << std::endl;
        std::cout << "Iterations: " << LATENCY_ITERATIONS << std::endl;
        
        // Generate random indices
        std::vector<size_t> indices(LATENCY_ITERATIONS);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, buffer1.size() - 1);
        
        for (auto& idx : indices) {
            idx = dis(gen);
        }
        
        std::vector<double> latencies;
        
        for (size_t iter = 0; iter < 5; ++iter) {
            volatile uint8_t value = 0;
            auto start = now();
            
            for (size_t i = 0; i < LATENCY_ITERATIONS; ++i) {
                value = buffer1[indices[i]];
            }
            
            auto end = now();
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double latency = calculateLatency(LATENCY_ITERATIONS, duration);
            latencies.push_back(latency);
            
            doNotOptimize(value);
        }
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Average latency: " << calculateAverage(latencies) << " ns" << std::endl;
        std::cout << "Min latency: " << *std::min_element(latencies.begin(), latencies.end()) << " ns" << std::endl;
        std::cout << "Max latency: " << *std::max_element(latencies.begin(), latencies.end()) << " ns" << std::endl;
    }

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    // SIMD-optimized read test using AVX2 (if available)
    void testSequentialReadSIMD() {
        std::cout << "\n=== Sequential Read Test (SIMD Optimized) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB" << std::endl;
        
#if defined(__AVX2__)
        std::cout << "Using AVX2 instructions (256-bit)" << std::endl;
        
        // Ensure alignment
        const uint8_t* data = buffer1.data();
        size_t size = buffer1.size();
        
        // Warmup
        __m256i acc = _mm256_setzero_si256();
        for (size_t i = 0; i + 32 <= size; i += 32) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            acc = _mm256_add_epi64(acc, chunk);
        }
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            memoryBarrier();
            auto start = now();
            
            // Multiple passes for sustained bandwidth
            for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                __m256i acc4 = _mm256_setzero_si256();
                __m256i acc5 = _mm256_setzero_si256();
                __m256i acc6 = _mm256_setzero_si256();
                __m256i acc7 = _mm256_setzero_si256();
                
                size_t i = 0;
                // Process 256 bytes (8x32) per iteration for better throughput
                for (; i + 256 <= size; i += 256) {
                    acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i)));
                    acc1 = _mm256_add_epi64(acc1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 32)));
                    acc2 = _mm256_add_epi64(acc2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 64)));
                    acc3 = _mm256_add_epi64(acc3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 96)));
                    acc4 = _mm256_add_epi64(acc4, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 128)));
                    acc5 = _mm256_add_epi64(acc5, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 160)));
                    acc6 = _mm256_add_epi64(acc6, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 192)));
                    acc7 = _mm256_add_epi64(acc7, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 224)));
                }
                
                // Handle remaining data
                for (; i + 32 <= size; i += 32) {
                    acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i)));
                }
                
                // Prevent optimization
                uint64_t result[4];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), 
                    _mm256_add_epi64(_mm256_add_epi64(_mm256_add_epi64(acc0, acc1), _mm256_add_epi64(acc2, acc3)),
                                    _mm256_add_epi64(_mm256_add_epi64(acc4, acc5), _mm256_add_epi64(acc6, acc7))));
                doNotOptimize(result[0]);
            }
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(size * INNER_ITERATIONS, duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
#else
        std::cout << "AVX2 not available, skipping SIMD test" << std::endl;
#endif
    }

    // Non-temporal (streaming) write test - bypasses cache
    void testStreamingWrite() {
        std::cout << "\n=== Streaming Write Test (Non-Temporal Stores) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB" << std::endl;
        
#if defined(__AVX2__)
        std::cout << "Using AVX2 non-temporal stores (bypasses cache)" << std::endl;
        
        uint8_t* data = buffer1.data();
        size_t size = buffer1.size();
        
        // Warmup
        __m256i value = _mm256_set1_epi64x(0xDEADBEEFCAFEBABELL);
        for (size_t i = 0; i + 32 <= size; i += 32) {
            _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i), value);
        }
        _mm_sfence();
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            memoryBarrier();
            auto start = now();
            
            // Multiple passes for sustained streaming write
            for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                // Unrolled non-temporal stores with AVX2
                size_t i = 0;
                for (; i + 256 <= size; i += 256) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i), value);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i + 32), value);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i + 64), value);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i + 96), value);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i + 128), value);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i + 160), value);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i + 192), value);
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i + 224), value);
                }
                
                // Handle remaining
                for (; i + 32 <= size; i += 32) {
                    _mm256_stream_si256(reinterpret_cast<__m256i*>(data + i), value);
                }
            }
            
            _mm_sfence(); // Ensure all writes complete
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(size * INNER_ITERATIONS, duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
        std::cout << "Using SSE2 non-temporal stores (bypasses cache)" << std::endl;
        
        uint8_t* data = buffer1.data();
        size_t size = buffer1.size();
        
        // Warmup
        for (size_t i = 0; i + 16 <= size; i += 16) {
            _mm_stream_si128(reinterpret_cast<__m128i*>(data + i), _mm_setzero_si128());
        }
        _mm_sfence();
        
        std::vector<double> bandwidths;
        __m128i value = _mm_set1_epi64x(0xDEADBEEFCAFEBABELL);
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            memoryBarrier();
            auto start = now();
            
            // Multiple passes for sustained streaming write
            for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                // Unrolled non-temporal stores
                size_t i = 0;
                for (; i + 128 <= size; i += 128) {
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i), value);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 16), value);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 32), value);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 48), value);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 64), value);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 80), value);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 96), value);
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 112), value);
                }
                
                // Handle remaining
                for (; i + 16 <= size; i += 16) {
                    _mm_stream_si128(reinterpret_cast<__m128i*>(data + i), value);
                }
            }
            
            _mm_sfence(); // Ensure all writes complete
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(size * INNER_ITERATIONS, duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
#else
        std::cout << "SSE2 not available, skipping streaming write test" << std::endl;
#endif
    }
#endif // x86/x64

#if defined(__aarch64__) || defined(__arm__)
    // ARM NEON optimized read test
    void testSequentialReadNEON() {
        std::cout << "\n=== Sequential Read Test (ARM NEON Optimized) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB" << std::endl;
        std::cout << "Using ARM NEON instructions (128-bit)" << std::endl;
        
        const uint8_t* data = buffer1.data();
        size_t size = buffer1.size();
        
        // Warmup
        uint64x2_t acc = vdupq_n_u64(0);
        for (size_t i = 0; i + 16 <= size; i += 16) {
            uint64x2_t chunk = vld1q_u64(reinterpret_cast<const uint64_t*>(data + i));
            acc = vaddq_u64(acc, chunk);
        }
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            memoryBarrier();
            auto start = now();
            
            // Multiple passes for sustained bandwidth
            for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                uint64x2_t acc0 = vdupq_n_u64(0);
                uint64x2_t acc1 = vdupq_n_u64(0);
                uint64x2_t acc2 = vdupq_n_u64(0);
                uint64x2_t acc3 = vdupq_n_u64(0);
                uint64x2_t acc4 = vdupq_n_u64(0);
                uint64x2_t acc5 = vdupq_n_u64(0);
                uint64x2_t acc6 = vdupq_n_u64(0);
                uint64x2_t acc7 = vdupq_n_u64(0);
                uint64x2_t acc8 = vdupq_n_u64(0);
                uint64x2_t acc9 = vdupq_n_u64(0);
                uint64x2_t accA = vdupq_n_u64(0);
                uint64x2_t accB = vdupq_n_u64(0);
                uint64x2_t accC = vdupq_n_u64(0);
                uint64x2_t accD = vdupq_n_u64(0);
                uint64x2_t accE = vdupq_n_u64(0);
                uint64x2_t accF = vdupq_n_u64(0);
                
                size_t i = 0;
                // Process 256 bytes (16x16) per iteration with software prefetch
                constexpr size_t PREFETCH_DISTANCE = 512;
                for (; i + 256 + PREFETCH_DISTANCE <= size; i += 256) {
                    // Prefetch future data
                    __builtin_prefetch(data + i + PREFETCH_DISTANCE, 0, 3);
                    
                    acc0 = vaddq_u64(acc0, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i)));
                    acc1 = vaddq_u64(acc1, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 16)));
                    acc2 = vaddq_u64(acc2, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 32)));
                    acc3 = vaddq_u64(acc3, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 48)));
                    acc4 = vaddq_u64(acc4, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 64)));
                    acc5 = vaddq_u64(acc5, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 80)));
                    acc6 = vaddq_u64(acc6, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 96)));
                    acc7 = vaddq_u64(acc7, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 112)));
                    acc8 = vaddq_u64(acc8, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 128)));
                    acc9 = vaddq_u64(acc9, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 144)));
                    accA = vaddq_u64(accA, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 160)));
                    accB = vaddq_u64(accB, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 176)));
                    accC = vaddq_u64(accC, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 192)));
                    accD = vaddq_u64(accD, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 208)));
                    accE = vaddq_u64(accE, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 224)));
                    accF = vaddq_u64(accF, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i + 240)));
                }
                
                // Handle remaining data
                for (; i + 16 <= size; i += 16) {
                    acc0 = vaddq_u64(acc0, vld1q_u64(reinterpret_cast<const uint64_t*>(data + i)));
                }
                
                // Prevent optimization
                uint64_t result[2];
                vst1q_u64(result, vaddq_u64(
                    vaddq_u64(vaddq_u64(vaddq_u64(acc0, acc1), vaddq_u64(acc2, acc3)),
                             vaddq_u64(vaddq_u64(acc4, acc5), vaddq_u64(acc6, acc7))),
                    vaddq_u64(vaddq_u64(vaddq_u64(acc8, acc9), vaddq_u64(accA, accB)),
                             vaddq_u64(vaddq_u64(accC, accD), vaddq_u64(accE, accF)))));
                doNotOptimize(result[0]);
            }
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(size * INNER_ITERATIONS, duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
    }
    
    // ARM NEON optimized write test with non-temporal hint
    void testStreamingWriteNEON() {
        std::cout << "\n=== Streaming Write Test (ARM NEON Non-Temporal) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB" << std::endl;
        std::cout << "Using ARM NEON with non-temporal stores" << std::endl;
        
        uint8_t* data = buffer1.data();
        size_t size = buffer1.size();
        
        uint64x2_t value = vdupq_n_u64(0xDEADBEEFCAFEBABEULL);
        
        // Warmup
        for (size_t i = 0; i + 16 <= size; i += 16) {
            vst1q_u64(reinterpret_cast<uint64_t*>(data + i), value);
        }
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            memoryBarrier();
            auto start = now();
            
            // Multiple passes for sustained streaming write
            for (size_t pass = 0; pass < INNER_ITERATIONS; ++pass) {
                size_t i = 0;
                // Heavily unrolled NEON stores for maximum throughput
                for (; i + 256 <= size; i += 256) {
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 16), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 32), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 48), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 64), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 80), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 96), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 112), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 128), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 144), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 160), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 176), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 192), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 208), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 224), value);
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i + 240), value);
                }
                
                // Handle remaining
                for (; i + 16 <= size; i += 16) {
                    vst1q_u64(reinterpret_cast<uint64_t*>(data + i), value);
                }
            }
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(size * INNER_ITERATIONS, duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
    }
#endif // ARM/AARCH64

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
    
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    // std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
    // std::cout << "║          SIMD & STREAMING TESTS                   ║" << std::endl;
    // std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;
    
    // SIMD and streaming tests (x86/x64 only)
    std::cout << "\nArch specific tests are skipped in current version." << std::endl;
    // benchmark.testSequentialReadSIMD();
    // benchmark.testStreamingWrite();
#elif defined(__aarch64__) || defined(__arm__)
    // ARM NEON tests
    std::cout << "\nArch specific tests are skipped in current version." << std::endl;
    // benchmark.testSequentialReadNEON();
    // benchmark.testStreamingWriteNEON();
#endif
    
    // std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
    // std::cout << "║          LATENCY TEST                             ║" << std::endl;
    // std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;
    
    benchmark.testRandomAccessLatency();

    // std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
    // std::cout << "║          BENCHMARK COMPLETE                       ║" << std::endl;
    // std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
