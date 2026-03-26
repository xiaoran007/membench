@echo off
REM Build script for Windows

echo Building MemBench...

cmake -S . -B build
cmake --build build --config Release

echo.
echo Build complete! Binary located at: build\Release\membench.exe
echo Try: .\build\Release\membench.exe --help
