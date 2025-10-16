@echo off
REM Build script for Windows

echo Building MemBench...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure
cmake ..

REM Build
cmake --build . --config Release

echo.
echo Build complete! Binary located at: build\Release\membench.exe
echo Run with: .\build\Release\membench.exe

cd ..
