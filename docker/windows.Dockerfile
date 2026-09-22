FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake ninja-build g++-mingw-w64-x86-64-posix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake/ cmake/
COPY src/ src/
COPY tests/ tests/

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchains/windows-x86_64.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DMEMBENCH_ENABLE_ISPC=OFF \
    && cmake --build /build --target membench membench_memory_probe --parallel

FROM scratch AS artifacts
COPY --from=build /build/membench.exe /build/membench_memory_probe.exe /
