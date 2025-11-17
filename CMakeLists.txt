cmake_minimum_required(VERSION 3.10)
project(membench VERSION 0.3.0 LANGUAGES CXX)

# Configure version header
configure_file(
    "${PROJECT_SOURCE_DIR}/src/version.h.in"
    "${PROJECT_BINARY_DIR}/version.h"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect CPU architecture and enable SIMD
include(CheckCXXCompilerFlag)

# Platform-specific settings
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    # Enable high-resolution timer on Windows
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
    # Enable AVX2 on Windows
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        message(STATUS "AVX2 support enabled")
    endif()
elseif(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    message(STATUS "Using -march=native for optimal CPU-specific optimizations")
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    message(STATUS "Using -march=native for optimal CPU-specific optimizations")
    # Link pthread on Linux
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

# Add executable
add_executable(membench src/membench.cpp)

# Include binary directory for generated version.h
target_include_directories(membench PRIVATE ${PROJECT_BINARY_DIR})

# Link libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(membench PRIVATE Threads::Threads)
endif()

# Installation rules
install(TARGETS membench DESTINATION bin)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
