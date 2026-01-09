# ORB-SLAM3 with GPU Acceleration (CUDA)

<p align="center">
<img src="https://img.shields.io/badge/CUDA-13.0+-green.svg" alt="CUDA">
<img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++17">
<img src="https://img.shields.io/badge/OpenCV-4.4+-orange.svg" alt="OpenCV">
<img src="https://img.shields.io/badge/Platform-Ubuntu%2022.04%2F24.04-lightgrey.svg" alt="Platform">
<img src="https://img.shields.io/badge/GPU-RTX%2030%2F40%20Series-76B900.svg" alt="GPU">
</p>

## 📖 Overview

This is a **GPU-accelerated fork** of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), the first real-time SLAM library capable of performing **Visual, Visual-Inertial and Multi-Map SLAM** with **monocular, stereo and RGB-D** cameras.

This implementation offloads computationally intensive feature extraction and matching operations to the GPU using **NVIDIA CUDA**, achieving significant speedup while maintaining accuracy comparable to the original CPU implementation.

### Key Improvements

| Module | Implementation | Speedup |
|--------|---------------|---------|
| **ORB Feature Extraction** | `ORBExtractorCuda.cu` | ~3-5x faster |
| **Feature Matching** | `ORBMatcherCuda.cu` | ~4-8x faster |
| **Spatial Indexing** | `FeatureGridCuda.cu` | ~2-3x faster |
| **Image Preprocessing** | `ImagePreprocessCuda.cu` | ~5-10x faster |

### Target Hardware
- **GPU**: NVIDIA RTX 30/40 series (Ada Lovelace, Compute Capability 8.6+)
- **Tested on**: NVIDIA RTX 4060 (SM 8.9)

---

## 📋 Prerequisites

### Hardware Requirements
- NVIDIA GPU with **Compute Capability 8.6+** (RTX 30xx, RTX 40xx)
- Minimum 8GB GPU VRAM recommended
- CPU: Intel i5/i7 or AMD Ryzen 5/7+ 

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Ubuntu | 22.04/24.04 | Tested platforms |
| CUDA Toolkit | 13.0+ | Required for GPU acceleration |
| CMake | 3.22+ | Build system |
| OpenCV | 4.4+ | **Must be built with CUDA support** |
| GCC | 11+ | C++17 support required |
| Pangolin | Latest | Visualization |
| Eigen3 | 3.1.0+ | Linear algebra |

### Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU information
nvidia-smi

# Verify compute capability (should be 8.6+)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

### Install Dependencies (Ubuntu)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install basic dependencies
sudo apt install -y build-essential cmake git libglew-dev \
    libboost-all-dev libssl-dev libeigen3-dev

# Install Pangolin dependencies
sudo apt install -y libgl1-mesa-dev libwayland-dev libxkbcommon-dev \
    wayland-protocols libegl1-mesa-dev libgles2-mesa-dev

# Clone and build Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
cd ../..

# Install OpenCV with CUDA support (critical!)
# Option 1: Build from source (recommended)
# Follow: https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html

# Option 2: Use pre-built packages (if available)
# sudo apt install libopencv-dev
```

---

## 🔨 Build Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ORB_SLAM3_CUDA.git ORB_SLAM3
cd ORB_SLAM3
```

### 2. Extract Vocabulary

```bash
# The vocabulary file should be extracted
cd Vocabulary
# If ORBvoc.txt is compressed:
tar -xf ORBvoc.txt.tar.gz  # or unzip if .zip
cd ..
```

### 3. Build Third-Party Dependencies

```bash
# Build DBoW2
cd Thirdparty/DBoW2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../..

# Build g2o
cd Thirdparty/g2o
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../..
```

### 4. Build ORB-SLAM3 with CUDA

```bash
mkdir build && cd build

# Configure with CUDA enabled (default)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON

# Build
make -j$(nproc)
```

#### Alternative: CPU-only Build (without GPU acceleration)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
make -j$(nproc)
```

### 5. Verify Build

After successful build, you should see:
```
-- =====================================
--   ORB-SLAM3 CUDA Configuration
-- =====================================
-- CUDA Version: 13.0
-- CUDA Include: /usr/local/cuda/include
-- CUDA Architectures: 86;89
-- Added GPU benchmark executable: benchmark_gpu_pipeline
```

Output files:
- `lib/libORB_SLAM3.so` or `lib/libORB_SLAM3_cuda.a`
- Executables in `Examples/` subdirectories

---

## 🚀 Running Examples

### Monocular Webcam (Quick Test)

```bash
# CPU version
./Examples/Monocular/mono_webcam \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/webcam.yaml

# GPU-accelerated version
./Examples/Monocular/mono_webcam_cuda \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/webcam_cuda.yaml
```

### Video File Input (GPU)

```bash
./Examples/Monocular/mono_video_cuda \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/webcam_cuda.yaml \
    /path/to/your/video.mp4
```

### EuRoC Dataset

```bash
# Download EuRoC dataset: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

./Examples/Monocular/mono_euroc \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/EuRoC.yaml \
    /path/to/MH_01_easy \
    Examples/Monocular/EuRoC_TimeStamps/MH01.txt
```

### TUM Dataset

```bash
./Examples/Monocular/mono_tum \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/TUM1.yaml \
    /path/to/rgbd_dataset_freiburg1_xyz
```

### Run Benchmarks

```bash
# Compare CPU vs GPU performance
./Examples/run_comparison_benchmark.sh

# Validate accuracy
./Examples/run_accuracy_validation.sh
```

---

## 📁 Project Structure

```
ORB_SLAM3/
├── include/                    # Header files
│   ├── cuda/                   # CUDA module headers
│   │   ├── GpuPipeline.h       # Unified GPU pipeline
│   │   ├── ORBExtractorCuda.h  # GPU ORB extraction
│   │   ├── ORBMatcherCuda.h    # GPU feature matching
│   │   ├── FeatureGridCuda.h   # GPU spatial indexing
│   │   └── ...
│   └── *.h                     # Original ORB-SLAM3 headers
├── src/                        # Source files
│   ├── cuda/                   # CUDA implementations
│   │   ├── GpuPipeline.cu      
│   │   ├── ORBExtractorCuda.cu 
│   │   ├── ORBMatcherCuda.cu   
│   │   └── ...
│   └── *.cc                    # Original ORB-SLAM3 sources
├── Examples/                   # Example applications
│   ├── Monocular/              # Monocular camera examples
│   ├── Stereo/                 # Stereo camera examples
│   ├── RGB-D/                  # RGB-D camera examples
│   ├── *-Inertial/             # Inertial variants
│   └── Calibration/            # Camera calibration tools
├── Thirdparty/                 # Third-party libraries
│   ├── DBoW2/                  # Bag of Words library
│   ├── g2o/                    # Graph optimization
│   └── Sophus/                 # Lie groups library
├── Vocabulary/                 # ORB vocabulary file
├── evaluation/                 # Evaluation scripts
├── cmake/                      # CMake modules
└── docs/                       # Documentation
```

---

## 🎯 GPU Parallelization Strategy

### Modules Offloaded to GPU

| Module | Strategy | Description |
|--------|----------|-------------|
| **ORB Extraction** | Data Parallelism | 2D grid decomposition, 1 thread per pixel |
| **Descriptor Computation** | Warp-Level Parallelism | 32 threads (1 warp) per descriptor |
| **Feature Matching** | Embarrassingly Parallel | Brute-force Hamming distance |
| **Spatial Indexing** | Atomic Operations | Parallel cell assignment |

### Key Optimizations

1. **Zero Host-Device Round-Trips**: All processing stays on GPU during frame processing
2. **Multi-Stream Execution**: Stereo images processed in parallel streams
3. **SoA Memory Layout**: Structure of Arrays for coalesced memory access
4. **Hardware Intrinsics**: `__popc()` for Hamming distance, `__shfl_sync()` for warp communication

For detailed technical analysis, see:
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) - English version
- [TECHNICAL_REPORT_VI.md](TECHNICAL_REPORT_VI.md) - Vietnamese version

---

## 📸 Camera Calibration

For best results, calibrate your camera. See [WEBCAM_GUIDE.md](WEBCAM_GUIDE.md) for detailed instructions.

Quick calibration config example (`webcam.yaml`):

```yaml
%YAML:1.0
Camera.type: "PinHole"
Camera.fx: 500.0
Camera.fy: 500.0
Camera.cx: 320.0
Camera.cy: 240.0
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0
Camera.width: 640
Camera.height: 480
Camera.fps: 30
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
```

---

## ⚠️ Troubleshooting

### Build Issues

**CUDA not found:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**OpenCV CUDA support missing:**
- Rebuild OpenCV with `-DWITH_CUDA=ON -DWITH_CUDNN=ON`

**C++17 not supported:**
- Upgrade GCC: `sudo apt install g++-11`

### Runtime Issues

**Tracking lost immediately:**
- Calibrate your camera
- Ensure good lighting and textured environment
- Move camera slowly during initialization

**Low FPS:**
- Check GPU utilization: `nvidia-smi`
- Reduce image resolution
- Reduce number of features (`ORBextractor.nFeatures`)

**Out of GPU memory:**
- Close other GPU applications
- Reduce image resolution
- Use smaller pyramid levels

---

## 📚 References

### Original ORB-SLAM3
```bibtex
@article{ORBSLAM3_TRO,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map {SLAM}},
  author={Campos, Carlos AND Elvira, Richard AND G\'omez, Juan J. AND Montiel, Jos\'e M. M. AND Tard\'os, Juan D.},
  journal={IEEE Transactions on Robotics}, 
  volume={37},
  number={6},
  pages={1874-1890},
  year={2021}
}
```

### Related Publications
- [ORB-SLAM3 Paper](https://arxiv.org/abs/2007.11898)
- [IMU-Initialization](https://arxiv.org/pdf/2003.05766.pdf)
- [ORBSLAM-Atlas](https://arxiv.org/pdf/1908.11585.pdf)

---

## 📄 License

ORB-SLAM3 is released under [GPLv3 license](LICENSE). 

For a list of all code/library dependencies and their licenses, see [Dependencies.md](Dependencies.md).

---

## 👥 Contributors

### Original ORB-SLAM3 Authors
Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, Juan D. Tardós

### GPU Acceleration
- Parallel Programming Course Project, HUST 2025

---

## 🔗 Useful Links

- [ORB-SLAM3 Original Repository](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [EuRoC Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [TUM RGB-D Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)
