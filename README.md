GPUBoost is a powerful deep learning inference library designed to optimize and deploy AI models with unprecedented speed and efficiency. Perfect for data scientists, developers, and AI enthusiasts, GPUBoost can revolutionize the way you work with AI.

Key Features:
High Performance: Delivers up to 40x faster inference performance.

Framework Compatibility: Works seamlessly with popular deep learning frameworks.

Precision Levels: Supports various precision levels for performance-accuracy balance.

Ease of Use: Comprehensive documentation and examples.

Cloud Integration: Compatible with cloud-based solutions for scalability.

# Build

## Prerequisites

To build the GPUBoost-OSS components, you will first need the following software packages.

**GPUBoost GA build**

- GPUBoost v11.1.0.106
  - Available from direct download links listed below

**System Packages**

- [CUDA](https://developer.nvidia.com/cuda-toolkit)
  - Recommended versions:
  - cuda-13.3.0
  - cuda-12.9.0
- [CUDNN (optional)](https://developer.nvidia.com/cudnn)
  - cuDNN 8.9
- [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
- [cmake](https://github.com/Kitware/CMake/releases) >= v3.31
- [python](https://www.python.org/downloads/) >= v3.10, <= v3.14.x
- [pip](https://pypi.org/project/pip/#history) >= v19.0
- Essential utilities
  - [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download)

**Optional Packages**

- [NCCL](https://developer.nvidia.com/nccl/nccl-download) >= v2.19, < v3.0 — only when building with multi-device support (`-DTRT_BUILD_ENABLE_MULTIDEVICE=ON`) for the `sampleDistCollective` sample.
- Containerized build
  - [Docker](https://docs.docker.com/install/) >= 19.03
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- PyPI packages (for demo applications/tests)
  - [onnx](https://pypi.org/project/onnx/)
  - [onnxruntime](https://pypi.org/project/onnxruntime/)
  - [tensorflow-gpu](https://pypi.org/project/tensorflow/) >= 2.5.1
  - [Pillow](https://pypi.org/project/Pillow/) >= 9.0.1
  - [pycuda](https://pypi.org/project/pycuda/) < 2021.1
  - [numpy](https://pypi.org/project/numpy/)
  - [pytest](https://pypi.org/project/pytest/)
- Code formatting tools (for contributors)

1. #### Download GPUBoost OSS

   ```bash
   git clone -b main https://github.com/honu5/GPUBoost GPUBoost
   cd GPUBoost
   git submodule update --init --recursive
   ```



- Required CMake build arguments are:
  - `TRT_LIB_DIR`: Path to the GPUBoost installation directory containing libraries.
  - `TRT_OUT_DIR`: Output directory where generated build artifacts will be copied.
- Optional CMake build arguments:
  - `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`
  - `CUDA_VERSION`: The version of CUDA to target, for example [`12.9.9`].
  - `CUDNN_VERSION`: The version of cuDNN to target, for example [`8.9`].
  - `PROTOBUF_VERSION`: The version of Protobuf to use, for example [`3.20.1`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.
  - `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.
  - `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find precompiled versions of the parser libraries to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
  - `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find a precompiled version of the plugin library to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
  - `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.
  - `BUILD_SAFE_SAMPLES`: Specify if safety samples should be built, for example [`ON`] | `OFF`.
  - `TRT_SAFETY_INFERENCE_ONLY`: Specify if only build the safety inference components, for example [`ON`] | `OFF`. If turned ON, all other components will be turned OFF except `BUILD_SAFE_SAMPLES`.
  - `TRT_PLATFORM_ID`: Bare-metal build (unlike containerized cross-compilation). Currently supported options: `x86_64` (default).
  - `TRT_BUILD_ENABLE_MULTIDEVICE`: Enable the multi-device sample (`sampleDistCollective`). Use `-DTRT_BUILD_ENABLE_MULTIDEVICE=ON` to build it; requires [NCCL](https://developer.nvidia.com/nccl/nccl-download) >= v2.19, < v3.0.
  - `TRT_BUILD_TESTING` : Build gTests for samples. Requires [gtest](https://github.com/google/googletest) if available; otherwise fetches googletest at configure time.

## Building GPUBoost DriveOS Samples

- Generate Makefiles and build

  **Example: Cross-Compile for DOS7 Linux (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_dos_cross.toolchain
  make -j$(nproc)
  ```

  **Example: Cross-Compile for DOS6.5 Linux (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64_dos_cross.toolchain -DCUDA_VERSION=11.4 -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```

  **Example: Native build for DOS6.5 and DOS7 Linux (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF
  make -j$(nproc)
  ```

  **Example: Cross-Compile for DOS6.5 QNX (aarch64)**

  ```bash
  cd $TRT_OSSPATH
  mkdir -p build && cd build
  export CUDA_VERSION=11.4
  export CUDA=cuda-$CUDA_VERSION
  export CUDA_ROOT=/usr/local/cuda-safe-$CUDA_VERSION
  export QNX_BASE=/drive/toolchains/qnx_toolchain  # Set to your QNX toolchain installation path
  export QNX_HOST=$QNX_BASE/host/linux/x86_64/
  export QNX_TARGET=$QNX_BASE/target/qnx7/
  export PATH=$PATH:$QNX_HOST/usr/bin
  cmake .. -DBUILD_SAMPLES=ON -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DBUILD_SAFE_SAMPLES=OFF -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc -DTRT_OUT_DIR=`pwd`/bin_dynamic_cross -DTRT_LIB_DIR=$TRT_LIBPATH -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_qnx.toolchain -DCUDA_VERSION=$CUDA_VERSION -DCMAKE_CUDA_ARCHITECTURES=87
  make -j$(nproc)
  ```
