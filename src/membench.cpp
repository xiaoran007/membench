/**
 * Memory Benchmark Tool
 * Tests memory read, write, copy speed and latency
 * Cross-platform support: Linux, macOS, Windows
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <random>
#include <string>
#include <sstream>

#ifdef _WIN32
    #include <windows.h>
    #include <intrin.h>
#else
    #include <unistd.h>
    #if defined(__x86_64__) || defined(__i386__)
        #include <x86intrin.h>
    #endif
#endif

class MemoryBenchmark {
private:
    static constexpr size_t KB = 1024;
    static constexpr size_t MB = 1024 * KB;
    static constexpr size_t DEFAULT_SIZE = 64 * MB;
    static constexpr size_t ITERATIONS = 10;
    static constexpr size_t LATENCY_ITERATIONS = 1000000;
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t UNROLL_FACTOR = 8;

    std::vector<uint8_t> buffer1;
    std::vector<uint8_t> buffer2;

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

public:
    MemoryBenchmark(size_t size = DEFAULT_SIZE) {
        buffer1.resize(size);
        buffer2.resize(size);
        
        // Initialize with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        
        for (auto& byte : buffer1) {
            byte = static_cast<uint8_t>(dis(gen));
        }
    }

    // Sequential read test with loop unrolling
    void testSequentialRead() {
        std::cout << "\n=== Sequential Read Test (Loop Unrolled) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB" << std::endl;
        
        // Warmup - touch all pages
        uint64_t warmup_sum = 0;
        const uint64_t* data = reinterpret_cast<const uint64_t*>(buffer1.data());
        size_t count = buffer1.size() / sizeof(uint64_t);
        for (size_t i = 0; i < count; ++i) {
            warmup_sum += data[i];
        }
        doNotOptimize(warmup_sum);
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            // Use multiple accumulators to improve ILP (Instruction Level Parallelism)
            uint64_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            uint64_t sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
            
            memoryBarrier();
            auto start = now();
            
            // Loop unrolling with multiple accumulators
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
            }
            
            // Handle remaining elements
            for (; i < count; ++i) {
                sum0 += data[i];
            }
            
            auto end = now();
            memoryBarrier();
            
            // Combine all sums
            uint64_t total = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
            doNotOptimize(total);
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(buffer1.size(), duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
    }

    // Sequential write test with loop unrolling
    void testSequentialWrite() {
        std::cout << "\n=== Sequential Write Test (Loop Unrolled) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB" << std::endl;
        
        // Warmup - touch all pages
        uint64_t* data = reinterpret_cast<uint64_t*>(buffer1.data());
        size_t count = buffer1.size() / sizeof(uint64_t);
        for (size_t i = 0; i < count; ++i) {
            data[i] = 0;
        }
        
        std::vector<double> bandwidths;
        const uint64_t v0 = 0xDEADBEEFCAFEBABEULL;
        const uint64_t v1 = 0xFEEDFACEDEADC0DEULL;
        const uint64_t v2 = 0xBADDCAFEBABEFACEULL;
        const uint64_t v3 = 0xC0DEDBADDEADFEEDULL;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            memoryBarrier();
            auto start = now();
            
            // Loop unrolling for better throughput
            size_t i = 0;
            for (; i + UNROLL_FACTOR <= count; i += UNROLL_FACTOR) {
                data[i + 0] = v0;
                data[i + 1] = v1;
                data[i + 2] = v2;
                data[i + 3] = v3;
                data[i + 4] = v0;
                data[i + 5] = v1;
                data[i + 6] = v2;
                data[i + 7] = v3;
            }
            
            // Handle remaining elements
            for (; i < count; ++i) {
                data[i] = v0;
            }
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(buffer1.size(), duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
    }

    // Memory copy test (using memcpy)
    void testMemoryCopy() {
        std::cout << "\n=== Memory Copy Test (memcpy) ===" << std::endl;
        std::cout << "Buffer size: " << buffer1.size() / MB << " MB" << std::endl;
        
        // Warmup
        std::memcpy(buffer2.data(), buffer1.data(), buffer1.size());
        
        std::vector<double> bandwidths;
        
        for (size_t iter = 0; iter < ITERATIONS; ++iter) {
            memoryBarrier();
            auto start = now();
            
            std::memcpy(buffer2.data(), buffer1.data(), buffer1.size());
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(buffer1.size(), duration);
            bandwidths.push_back(bandwidth);
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
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            
            memoryBarrier();
            auto start = now();
            
            size_t i = 0;
            // Process 128 bytes (4x32) per iteration for better throughput
            for (; i + 128 <= size; i += 128) {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i)));
                acc1 = _mm256_add_epi64(acc1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 32)));
                acc2 = _mm256_add_epi64(acc2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 64)));
                acc3 = _mm256_add_epi64(acc3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i + 96)));
            }
            
            // Handle remaining data
            for (; i + 32 <= size; i += 32) {
                acc0 = _mm256_add_epi64(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i)));
            }
            
            auto end = now();
            memoryBarrier();
            
            // Prevent optimization
            uint64_t result[4];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), 
                _mm256_add_epi64(_mm256_add_epi64(acc0, acc1), _mm256_add_epi64(acc2, acc3)));
            doNotOptimize(result[0]);
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(size, duration);
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
        
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
        std::cout << "Using non-temporal stores (bypasses cache)" << std::endl;
        
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
            
            // Unrolled non-temporal stores
            size_t i = 0;
            for (; i + 64 <= size; i += 64) {
                _mm_stream_si128(reinterpret_cast<__m128i*>(data + i), value);
                _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 16), value);
                _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 32), value);
                _mm_stream_si128(reinterpret_cast<__m128i*>(data + i + 48), value);
            }
            
            // Handle remaining
            for (; i + 16 <= size; i += 16) {
                _mm_stream_si128(reinterpret_cast<__m128i*>(data + i), value);
            }
            
            _mm_sfence(); // Ensure all writes complete
            
            auto end = now();
            memoryBarrier();
            
            auto duration = std::chrono::duration_cast<Duration>(end - start);
            double bandwidth = calculateBandwidth(size, duration);
            bandwidths.push_back(bandwidth);
        }
        
        printStatistics(bandwidths);
#else
        std::cout << "SSE2 not available, skipping streaming write test" << std::endl;
#endif
    }
#endif // x86/x64

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
        std::cout << "Average bandwidth: " << avg << " MB/s" << std::endl;
        std::cout << "Min bandwidth: " << min << " MB/s" << std::endl;
        std::cout << "Max bandwidth: " << max << " MB/s" << std::endl;
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
    std::cout << "║   Memory Benchmark Tool v0.1.0       ║" << std::endl;
    std::cout << "║   Cross-platform Memory Testing      ║" << std::endl;
    std::cout << "║   With SIMD & Loop Optimizations     ║" << std::endl;
    std::cout << "╚══════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    MemoryBenchmark::printSystemInfo();

    // Parse command line arguments for buffer size
    size_t bufferSize = 64 * 1024 * 1024; // Default 64MB
    if (argc > 1) {
        try {
            // Use stringstream for better cross-platform compatibility
            std::stringstream ss(argv[1]);
            unsigned long long sizeMB = 0;
            ss >> sizeMB;
            if (!ss.fail() && sizeMB > 0 && sizeMB <= 10240) { // Max 10GB
                bufferSize = sizeMB * 1024 * 1024; // Convert MB to bytes
                std::cout << "Using custom buffer size: " << bufferSize / (1024 * 1024) << " MB" << std::endl;
            } else {
                std::cerr << "Invalid buffer size (must be 1-10240 MB), using default 64MB" << std::endl;
            }
        } catch (...) {
            std::cerr << "Invalid buffer size argument, using default 64MB" << std::endl;
        }
    }

    MemoryBenchmark benchmark(bufferSize);

    // std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
    // std::cout << "║          BANDWIDTH TESTS                          ║" << std::endl;
    // std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;
    
    // Bandwidth tests (with loop unrolling optimization)
    benchmark.testSequentialRead();
    benchmark.testSequentialWrite();
    benchmark.testMemoryCopy();
    
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    // std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
    // std::cout << "║          SIMD & STREAMING TESTS                   ║" << std::endl;
    // std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;
    
    // SIMD and streaming tests (x86/x64 only)
    benchmark.testSequentialReadSIMD();
    benchmark.testStreamingWrite();
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
