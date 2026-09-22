@echo off
setlocal
REM Build Windows x64 binaries using Docker Desktop in Linux containers mode.

pushd "%~dp0"
if errorlevel 1 exit /b 1

echo Building MemBench for Windows x64 with Docker...
docker build --file docker/windows.Dockerfile --output type=local,dest=build/Release --progress=plain .
set "DOCKER_BUILD_EXIT=%ERRORLEVEL%"
popd

if not "%DOCKER_BUILD_EXIT%"=="0" (
    echo Docker build failed. See the output above.
    exit /b %DOCKER_BUILD_EXIT%
)

echo.
echo Build complete! Binaries located at:
echo   "%~dp0build\Release\membench.exe"
echo   "%~dp0build\Release\membench_memory_probe.exe"
echo Try: "%~dp0build\Release\membench.exe" --help
exit /b 0
