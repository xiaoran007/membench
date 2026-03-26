# 安装和构建指南

## 依赖要求

- CMake 3.10 或更高版本
- 支持 C++17 的编译器
  - macOS: Apple Clang 17+ 推荐
  - Linux: GCC 7+ 或 Clang 5+
  - Windows: MSVC 2017+

## 安装依赖

### macOS

```bash
brew install cmake
cmake --version
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install cmake build-essential
cmake --version
g++ --version
```

### CentOS / RHEL / Fedora

```bash
sudo yum install cmake gcc-c++ make
# 或
sudo dnf install cmake gcc-c++ make
cmake --version
g++ --version
```

### Windows

方法 1：Visual Studio

1. 安装 [Visual Studio Community](https://visualstudio.microsoft.com/)
2. 勾选 `Desktop development with C++`
3. 勾选 `CMake tools for Windows`

方法 2：MinGW-w64

1. 安装 [CMake](https://cmake.org/download/)
2. 安装 [MinGW-w64](https://www.mingw-w64.org/)
3. 将二者的 `bin` 目录加入 `PATH`

## 快速构建

### macOS / Linux

```bash
./build.sh
```

或手动构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Windows

```cmd
build.bat
```

或手动构建：

```cmd
cmake -S . -B build
cmake --build build --config Release
```

## 运行程序

### 基本用法

```bash
./build/membench
./build/membench 1024
./build/membench --mode standard
./build/membench --mode peak --size-mb 1024 --tests read,copy
```

Windows:

```cmd
.\build\Release\membench.exe
.\build\Release\membench.exe 1024
.\build\Release\membench.exe --mode standard
```

### 常用参数

- `--size-mb <n>`：每个 buffer 的大小，单位 MiB
- `--threads <n>`：显式指定线程数
- `--tests read,write,copy`：只运行部分测试
- `--iterations <n>`：测量轮数
- `--warmup <n>`：预热轮数
- `--thread-policy perf|all`：限制线程搜索范围
- `--mode standard|peak`：稳定模式或峰值模式
- `--no-calibrate`：关闭峰值模式的短校准
- `--no-qos`：关闭 macOS QoS 提示
- `--help`：查看帮助

## 默认行为

- 默认 buffer 大小：`min(1 GiB, 物理内存 / 8)`，下限 `256 MiB`
- 默认测量轮数：`7`
- 默认预热轮数：`2`
- Apple Silicon 默认模式：`peak`
- Apple Silicon 默认线程策略：`all`
- 非 Apple Silicon 默认模式：`standard`

Apple Silicon 的 `peak` 模式会对每项测试单独做短校准，然后选择最优线程数和内核。

## 结果说明

每项测试都会输出：

- `mode`
- `kernel`
- `selected_threads`
- `calibrated`
- `size_mb`
- `logical_bytes_per_iteration`
- `measured_elapsed_ms`
- `avg / median / min / max / stdev bandwidth`

`copy` 会额外输出两种带宽口径：

- `logical ... bandwidth`：逻辑复制吞吐
- `estimated traffic ... bandwidth`：逻辑吞吐乘以 2 的估算流量

第二种口径只是为了更方便和芯片标称内存带宽对照，不等于硬件计数器观测值。

## 使用建议

1. 使用 Release 构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

2. 想做稳定对比时，优先使用：

```bash
./build/membench --mode standard
```

3. 想尽量逼近 Apple Silicon 峰值时，直接运行默认命令或显式使用：

```bash
./build/membench --mode peak
```

4. 小 buffer 更容易被缓存影响；想看主内存带宽时优先使用较大尺寸。
5. 测试时尽量关闭后台内存密集型应用。

## 故障排查

### CMake 版本过低

```bash
# macOS
brew upgrade cmake

# Ubuntu / Debian
sudo apt-get install --only-upgrade cmake
```

### 编译器不支持 C++17

升级到以下版本之一：

- GCC 7+
- Clang 5+
- MSVC 2017+

### 内存不足

减小 buffer 大小，例如：

```bash
./build/membench --size-mb 256
```

### Windows 上找不到编译器

确认已安装 Visual Studio 或 MinGW-w64，并且其工具链在 `PATH` 中。

## 验证构建

成功构建后，可以先运行：

```bash
./build/membench --help
./build/membench --size-mb 64 --iterations 3 --warmup 1
```

你应该会看到：

- 系统信息
- benchmark 配置
- `read` / `write` / `copy` 测试结果
- Apple Silicon 上的 `mode`、`kernel`、`selected_threads`、`calibrated`
- `copy` 的逻辑吞吐与估算流量两种输出
