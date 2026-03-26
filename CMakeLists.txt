cmake_minimum_required(VERSION 3.10)
project(membench VERSION 0.5.0 LANGUAGES CXX)

# Configure version header
configure_file(
    "${PROJECT_SOURCE_DIR}/src/version.h.in"
    "${PROJECT_BINARY_DIR}/version.h"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(membench src/membench.cpp)
target_include_directories(membench PRIVATE ${PROJECT_BINARY_DIR})

if(MSVC)
    target_compile_definitions(membench PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(membench PRIVATE /W4 "$<$<NOT:$<CONFIG:Debug>>:/O2>")
else()
    target_compile_options(membench PRIVATE -Wall -Wextra "$<$<NOT:$<CONFIG:Debug>>:-O3>")
endif()

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    check_cxx_compiler_flag("-mcpu=native" COMPILER_SUPPORTS_MCPU_NATIVE)
    if(COMPILER_SUPPORTS_MCPU_NATIVE)
        target_compile_options(membench PRIVATE -mcpu=native)
        message(STATUS "Enabled -mcpu=native for Apple Silicon builds")
    else()
        message(STATUS "Compiler does not support -mcpu=native; using generic Apple Silicon codegen")
    endif()
elseif((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i.86") OR
        (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i.86"))
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(membench PRIVATE -march=native)
        message(STATUS "Enabled -march=native for x86-family builds")
    endif()
endif()

if(UNIX AND NOT APPLE)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(membench PRIVATE Threads::Threads)
endif()

install(TARGETS membench DESTINATION bin)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
