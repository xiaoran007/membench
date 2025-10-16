# 安装和构建指南

## 安装依赖

### macOS

```bash
# 使用 Homebrew 安装 CMake
brew install cmake

# 确认安装
cmake --version
```

### Linux (Ubuntu/Debian)

```bash
# 安装 CMake 和构建工具
sudo apt-get update
sudo apt-get install cmake build-essential

# 确认安装
cmake --version
g++ --version
```

### Linux (CentOS/RHEL/Fedora)

```bash
# 安装 CMake 和构建工具
sudo yum install cmake gcc-c++ make
# 或者使用 dnf (Fedora)
sudo dnf install cmake gcc-c++ make

# 确认安装
cmake --version
g++ --version
```

### Windows

**方法 1: 使用 Visual Studio**

1. 下载并安装 [Visual Studio Community](https://visualstudio.microsoft.com/)
2. 在安装时选择 "Desktop development with C++"
3. 勾选 "CMake tools for Windows"

**方法 2: 使用 MinGW-w64**

1. 下载 [CMake](https://cmake.org/download/)
2. 下载 [MinGW-w64](https://www.mingw-w64.org/)
3. 将两者的 bin 目录添加到系统 PATH

## 快速构建

### Linux / macOS

使用提供的构建脚本：

```bash
./build.sh
```

或手动构建：

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### Windows

使用提供的构建脚本：

```cmd
build.bat
```

或手动构建（Visual Studio）：

```cmd
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

或手动构建（MinGW）：

```cmd
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

## 运行程序

### Linux / macOS

```bash
# 使用默认 64MB 缓冲区
./build/membench

# 使用 128MB 缓冲区
./build/membench 128
```

### Windows (Visual Studio)

```cmd
.\build\Release\membench.exe
.\build\Release\membench.exe 128
```

### Windows (MinGW)

```cmd
.\build\membench.exe
.\build\membench.exe 128
```

## 性能建议

1. **使用 Release 构建**：确保使用优化选项编译
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

2. **关闭后台程序**：测试时关闭其他占用内存的应用

3. **固定 CPU 频率**（Linux）：
   ```bash
   # 需要 root 权限
   sudo cpupower frequency-set -g performance
   ```

4. **禁用节能模式**（macOS）：
   - 系统偏好设置 → 节能 → 关闭自动睡眠

5. **提升进程优先级**：
   ```bash
   # Linux/macOS
   sudo nice -n -20 ./build/membench
   
   # Windows (以管理员身份运行 PowerShell)
   Start-Process -FilePath ".\build\Release\membench.exe" -Verb RunAs
   ```

## 故障排除

### CMake 版本过低

```bash
# 升级 CMake
# macOS
brew upgrade cmake

# Linux
sudo apt-get install --only-upgrade cmake
```

### 编译器不支持 C++17

更新编译器到支持 C++17 的版本：
- GCC 7+
- Clang 5+
- MSVC 2017+

### 内存不足

减小测试缓冲区大小：

```bash
# 使用 32MB 而不是默认的 64MB
./build/membench 32
```

### Windows 上找不到编译器

确保已安装 Visual Studio 或 MinGW，并将其添加到系统 PATH。

## 验证构建

成功构建后，运行程序应该看到类似输出：

```
╔══════════════════════════════════════╗
║   Memory Benchmark Tool v1.0.0      ║
║   Cross-platform Memory Testing     ║
╚══════════════════════════════════════╝

=== System Information ===
Operating System: macOS
Page size: 16384 bytes
...
```

如果看到此输出，说明构建成功！
