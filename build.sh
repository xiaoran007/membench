#!/bin/bash
# Build script for Linux/macOS

set -e

echo "Building MemBench..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo
echo "Build complete! Binary located at: build/membench"
echo "Try: ./build/membench --help"
