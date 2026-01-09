#!/bin/bash
#===============================================================================
# CUDA-Accelerated Real-time Webcam SLAM Launcher
#
# This script launches the CUDA-optimized webcam SLAM application with
# proper environment setup for NVIDIA RTX 4060.
#
# Usage: ./run_webcam_cuda.sh [camera_id]
#
# Requirements:
# - NVIDIA RTX 4060 GPU (or compatible CUDA GPU)
# - CUDA 13.0 installed
# - ORB-SLAM3 built with CUDA support
#
# Author: CUDA Optimization Team
# Target: Ubuntu 24.04, RTX 4060, CUDA 13.0
#===============================================================================

set -e  # Exit on error

#===============================================================================
# Configuration
#===============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths
VOCABULARY="${PROJECT_ROOT}/Vocabulary/ORBvoc.txt"
SETTINGS="${PROJECT_ROOT}/Examples/Monocular/webcam_cuda.yaml"
EXECUTABLE="${PROJECT_ROOT}/Examples/Monocular/mono_webcam_cuda"

# Default camera
CAMERA_ID=${1:-0}

#===============================================================================
# Color Output Functions
#===============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     ORB-SLAM3 CUDA Real-time Webcam SLAM                     ║"
    echo "║     Optimized for NVIDIA RTX 4060                            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

#===============================================================================
# Environment Setup
#===============================================================================
setup_environment() {
    print_info "Setting up CUDA environment..."

    # CUDA paths (adjust if your CUDA is installed elsewhere)
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-13.0" ]; then
        export CUDA_HOME="/usr/local/cuda-13.0"
    fi

    if [ -n "$CUDA_HOME" ]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    fi

    # Add ORB-SLAM3 library path
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/Thirdparty/DBoW2/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/Thirdparty/g2o/lib:$LD_LIBRARY_PATH"

    # GPU Performance Settings
    # Set high performance mode for NVIDIA GPU (requires sudo for some operations)
    if command -v nvidia-smi &> /dev/null; then
        print_info "Configuring GPU for maximum performance..."
        
        # Check GPU persistence mode (optional, requires sudo)
        # nvidia-smi -pm 1 2>/dev/null || true
        
        # Query GPU info
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1)
        print_success "GPU detected: $GPU_NAME ($GPU_MEMORY)"
    fi
}

#===============================================================================
# Validation
#===============================================================================
validate_dependencies() {
    print_info "Validating dependencies..."

    # Check executable
    if [ ! -f "$EXECUTABLE" ]; then
        print_error "Executable not found: $EXECUTABLE"
        print_info "Please build the project first with: ./build.sh"
        exit 1
    fi
    print_success "Executable found"

    # Check vocabulary
    if [ ! -f "$VOCABULARY" ]; then
        print_error "Vocabulary file not found: $VOCABULARY"
        exit 1
    fi
    print_success "Vocabulary found"

    # Check settings
    if [ ! -f "$SETTINGS" ]; then
        print_error "Settings file not found: $SETTINGS"
        exit 1
    fi
    print_success "Settings found"

    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        print_success "NVIDIA Driver version: $CUDA_VERSION"
    else
        print_warning "nvidia-smi not found. CUDA may not be properly installed."
    fi

    # Check camera
    if [ -e "/dev/video${CAMERA_ID}" ]; then
        print_success "Camera /dev/video${CAMERA_ID} found"
    else
        print_warning "Camera /dev/video${CAMERA_ID} not found. Will attempt to open anyway."
    fi
}

#===============================================================================
# Camera Check
#===============================================================================
check_camera() {
    print_info "Checking camera access..."

    # Check if v4l-utils is available
    if command -v v4l2-ctl &> /dev/null; then
        if v4l2-ctl --device=/dev/video${CAMERA_ID} --info &>/dev/null; then
            print_success "Camera accessible via V4L2"
            v4l2-ctl --device=/dev/video${CAMERA_ID} --info 2>/dev/null | grep -E "Card type|Driver name" || true
        fi
    fi

    # Check camera permissions
    if [ -c "/dev/video${CAMERA_ID}" ]; then
        if [ -r "/dev/video${CAMERA_ID}" ] && [ -w "/dev/video${CAMERA_ID}" ]; then
            print_success "Camera permissions OK"
        else
            print_warning "Camera may have permission issues. Try: sudo chmod 666 /dev/video${CAMERA_ID}"
        fi
    fi
}

#===============================================================================
# Cleanup Handler
#===============================================================================
cleanup() {
    print_info "Cleaning up..."
    # Kill any leftover processes if needed
    jobs -p | xargs -r kill 2>/dev/null || true
}

trap cleanup EXIT

#===============================================================================
# Main Execution
#===============================================================================
main() {
    print_header

    # Parse arguments
    echo -e "${BLUE}Configuration:${NC}"
    echo "  Camera ID: ${CAMERA_ID}"
    echo "  Vocabulary: ${VOCABULARY}"
    echo "  Settings: ${SETTINGS}"
    echo ""

    # Setup and validate
    setup_environment
    validate_dependencies
    check_camera

    echo ""
    print_info "Launching CUDA Webcam SLAM..."
    echo ""
    echo -e "${YELLOW}Controls:${NC}"
    echo "  [Q/ESC] - Quit"
    echo "  [S]     - Save trajectory"
    echo "  [R]     - Reset map"
    echo "  [P]     - Toggle performance overlay"
    echo ""

    # Launch the application
    cd "$PROJECT_ROOT"
    "$EXECUTABLE" "$VOCABULARY" "$SETTINGS" "$CAMERA_ID"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        print_success "SLAM session completed successfully!"
    else
        print_error "SLAM session ended with exit code: $EXIT_CODE"
    fi

    return $EXIT_CODE
}

# Run main function
main "$@"
