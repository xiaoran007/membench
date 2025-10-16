#!/bin/bash
# Build script for Linux/macOS

set -e

echo "Building MemBench..."

# Create build directory
mkdir -p build
cd build

# Configure
cmake ..

# Build
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Build complete! Binary located at: build/membench"
echo "Run with: ./build/membench"
