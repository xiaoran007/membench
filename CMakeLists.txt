cmake_minimum_required(VERSION 3.10)
project(membench VERSION 0.5.0 LANGUAGES CXX)

set(MEMBENCH_ISPC_DEFAULT OFF)
if(UNIX AND NOT APPLE)
    set(MEMBENCH_ISPC_DEFAULT ON)
endif()
option(MEMBENCH_ENABLE_ISPC "Build ISPC CPU SIMD kernels" ${MEMBENCH_ISPC_DEFAULT})
set(MEMBENCH_ISPC_TARGET_DEFAULT "")
if((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i.86") OR
   (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i.86"))
    set(MEMBENCH_ISPC_TARGET_DEFAULT "avx2-i32x8")
endif()
set(MEMBENCH_ISPC_TARGET "${MEMBENCH_ISPC_TARGET_DEFAULT}" CACHE STRING "Optional explicit ISPC target, for example avx2-i32x8")

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

set(MEMBENCH_SOURCES
    src/membench.cpp
    src/core/format.cpp
    src/core/statistics.cpp
    src/kernels/kernel_registry.cpp
    src/memory/aligned_buffer.cpp
    src/platform/memory_probe.cpp
    src/platform/platform.cpp
    src/planner/execution_planner.cpp
)

add_executable(membench ${MEMBENCH_SOURCES})
target_include_directories(membench PRIVATE
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

if(MEMBENCH_ENABLE_ISPC)
    find_program(ISPC_EXECUTABLE ispc)
    if(NOT ISPC_EXECUTABLE)
        message(WARNING
            "ISPC compiler not found; building without optional ISPC CPU kernels. "
            "Install ispc to enable ispc_read/ispc_write/ispc_copy.")
    else()
        set(ISPC_SOURCE "${PROJECT_SOURCE_DIR}/src/ispc_kernels.ispc")
        set(ISPC_OBJECT "${PROJECT_BINARY_DIR}/membench_ispc.o")
        set(ISPC_HEADER "${PROJECT_BINARY_DIR}/membench_ispc.h")
        set(ISPC_EFFECTIVE_TARGET "${MEMBENCH_ISPC_TARGET}")
        if(NOT ISPC_EFFECTIVE_TARGET)
            set(ISPC_EFFECTIVE_TARGET "${MEMBENCH_ISPC_TARGET_DEFAULT}")
        endif()
        set(ISPC_COMMAND
            "${ISPC_EXECUTABLE}" "${ISPC_SOURCE}"
            -O3 --addressing=64 --pic
            --emit-obj -o "${ISPC_OBJECT}"
            -h "${ISPC_HEADER}"
        )
        if(ISPC_EFFECTIVE_TARGET)
            list(APPEND ISPC_COMMAND --target=${ISPC_EFFECTIVE_TARGET})
        endif()

        add_custom_command(
            OUTPUT "${ISPC_OBJECT}" "${ISPC_HEADER}"
            COMMAND ${ISPC_COMMAND}
            DEPENDS "${ISPC_SOURCE}"
            COMMENT "Compiling ISPC CPU kernels"
            VERBATIM
        )
        add_custom_target(membench_ispc_codegen DEPENDS "${ISPC_OBJECT}" "${ISPC_HEADER}")
        set_source_files_properties("${ISPC_OBJECT}" PROPERTIES
            EXTERNAL_OBJECT TRUE
            GENERATED TRUE
        )
        target_sources(membench PRIVATE "${ISPC_OBJECT}")
        target_compile_definitions(membench PRIVATE MEMBENCH_HAS_ISPC=1)
        add_dependencies(membench membench_ispc_codegen)
        message(STATUS "ISPC CPU kernels enabled: ${ISPC_EXECUTABLE}")
        if(ISPC_EFFECTIVE_TARGET)
            message(STATUS "ISPC target: ${ISPC_EFFECTIVE_TARGET}")
        else()
            message(STATUS "ISPC target: host default")
        endif()
    endif()
endif()

if(WIN32)
    target_compile_definitions(membench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

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

    # --- Metal GPU backend ---------------------------------------------------
    # Shaders are compiled from source at runtime via newLibraryWithSource:,
    # so no build-time Metal toolchain dependency is required.
    enable_language(OBJCXX)

    target_sources(membench PRIVATE src/backends/metal_backend.mm)
    set_source_files_properties(src/backends/metal_backend.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
    target_compile_definitions(membench PRIVATE MEMBENCH_HAS_METAL=1)
    target_link_libraries(membench PRIVATE "-framework Metal" "-framework Foundation")

    message(STATUS "Metal GPU backend enabled (runtime shader compilation)")
    # -------------------------------------------------------------------------

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

enable_testing()

add_executable(membench_unit_tests
    tests/unit_tests.cpp
    src/core/format.cpp
    src/core/statistics.cpp
    src/platform/memory_probe.cpp
    src/platform/platform.cpp
    src/planner/execution_planner.cpp
    src/kernels/kernel_registry.cpp
)
target_include_directories(membench_unit_tests PRIVATE
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/src
)
if(WIN32)
    target_compile_definitions(membench_unit_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if(MSVC)
    target_compile_definitions(membench_unit_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(membench_unit_tests PRIVATE /W4 "$<$<NOT:$<CONFIG:Debug>>:/O2>")
else()
    target_compile_options(membench_unit_tests PRIVATE -Wall -Wextra "$<$<NOT:$<CONFIG:Debug>>:-O3>")
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(membench_unit_tests PRIVATE Threads::Threads)
endif()

add_test(NAME membench_unit_tests COMMAND membench_unit_tests)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
