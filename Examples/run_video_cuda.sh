#!/bin/bash
#===============================================================================
# CUDA-Accelerated Video File SLAM Launcher
#
# This script launches the CUDA-optimized video processing SLAM application
# with proper environment setup for NVIDIA RTX 4060.
#
# Usage: ./run_video_cuda.sh <video_file> [options]
#
# Examples:
#   ./run_video_cuda.sh /path/to/video.mp4
#   ./run_video_cuda.sh /path/to/video.mp4 --mode realtime
#   ./run_video_cuda.sh /path/to/video.mp4 --mode fast --start 100 --end 500
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
EXECUTABLE="${PROJECT_ROOT}/Examples/Monocular/mono_video_cuda"

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
    echo "║     ORB-SLAM3 CUDA Video Processing Demo                     ║"
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

print_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 <video_file> [options]"
    echo ""
    echo -e "${YELLOW}Required:${NC}"
    echo "  video_file            Path to input video file (.mp4, .avi, .mkv, etc.)"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --mode MODE           Playback mode: realtime, fast, step (default: fast)"
    echo "  --start N             Start processing from frame N (default: 0)"
    echo "  --end N               Stop processing at frame N (default: all)"
    echo "  --settings FILE       Use custom settings YAML file"
    echo "  --no-viewer           Disable the 3D viewer"
    echo "  --no-save             Don't save trajectory"
    echo "  --help                Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 ~/videos/test.mp4"
    echo "  $0 ~/videos/test.mp4 --mode realtime"
    echo "  $0 ~/videos/test.mp4 --mode fast --start 100 --end 1000"
    echo "  $0 ~/videos/test.mp4 --settings custom_camera.yaml"
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
    if command -v nvidia-smi &> /dev/null; then
        print_info "Configuring GPU for maximum performance..."
        
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
}

#===============================================================================
# Video Validation
#===============================================================================
validate_video() {
    local video_file="$1"

    print_info "Validating video file..."

    if [ ! -f "$video_file" ]; then
        print_error "Video file not found: $video_file"
        exit 1
    fi

    # Check file extension
    local extension="${video_file##*.}"
    local extension_lower=$(echo "$extension" | tr '[:upper:]' '[:lower:]')
    
    case "$extension_lower" in
        mp4|avi|mkv|mov|wmv|flv|webm|m4v)
            print_success "Video format recognized: $extension_lower"
            ;;
        *)
            print_warning "Unknown video format: $extension_lower (will try to process anyway)"
            ;;
    esac

    # Get video info using ffprobe if available
    if command -v ffprobe &> /dev/null; then
        local duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video_file" 2>/dev/null)
        local resolution=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$video_file" 2>/dev/null)
        local fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video_file" 2>/dev/null)
        
        if [ -n "$resolution" ]; then
            print_success "Video resolution: $resolution"
        fi
        if [ -n "$fps" ]; then
            print_success "Video FPS: $fps"
        fi
        if [ -n "$duration" ]; then
            print_success "Video duration: ${duration%.*} seconds"
        fi
    fi

    print_success "Video file validated"
}

#===============================================================================
# Cleanup Handler
#===============================================================================
cleanup() {
    print_info "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
}

trap cleanup EXIT

#===============================================================================
# Parse Arguments
#===============================================================================
parse_args() {
    VIDEO_FILE=""
    MODE_ARG=""
    START_ARG=""
    END_ARG=""
    EXTRA_ARGS=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                print_header
                print_usage
                exit 0
                ;;
            --mode)
                MODE_ARG="--mode $2"
                shift 2
                ;;
            --start)
                START_ARG="--start $2"
                shift 2
                ;;
            --end)
                END_ARG="--end $2"
                shift 2
                ;;
            --settings)
                SETTINGS="$2"
                shift 2
                ;;
            --no-viewer)
                EXTRA_ARGS="$EXTRA_ARGS --no-viewer"
                shift
                ;;
            --no-save)
                EXTRA_ARGS="$EXTRA_ARGS --no-save"
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                if [ -z "$VIDEO_FILE" ]; then
                    VIDEO_FILE="$1"
                else
                    print_error "Unexpected argument: $1"
                    print_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [ -z "$VIDEO_FILE" ]; then
        print_error "Video file is required!"
        echo ""
        print_usage
        exit 1
    fi
}

#===============================================================================
# Main Execution
#===============================================================================
main() {
    print_header
    
    # Parse command line arguments
    parse_args "$@"

    echo -e "${BLUE}Configuration:${NC}"
    echo "  Video: ${VIDEO_FILE}"
    echo "  Vocabulary: ${VOCABULARY}"
    echo "  Settings: ${SETTINGS}"
    [ -n "$MODE_ARG" ] && echo "  Mode: ${MODE_ARG#--mode }"
    [ -n "$START_ARG" ] && echo "  Start Frame: ${START_ARG#--start }"
    [ -n "$END_ARG" ] && echo "  End Frame: ${END_ARG#--end }"
    echo ""

    # Setup and validate
    setup_environment
    validate_dependencies
    validate_video "$VIDEO_FILE"

    echo ""
    print_info "Launching CUDA Video Processing SLAM..."
    echo ""
    echo -e "${YELLOW}Controls (in step mode):${NC}"
    echo "  [SPACE] - Next frame"
    echo "  [Q/ESC] - Quit"
    echo "  [S]     - Save trajectory"
    echo ""

    # Build full command
    CMD="$EXECUTABLE $VOCABULARY $SETTINGS $VIDEO_FILE $MODE_ARG $START_ARG $END_ARG $EXTRA_ARGS"
    
    print_info "Executing: $CMD"
    echo ""

    # Launch the application
    cd "$PROJECT_ROOT"
    eval "$CMD"

    EXIT_CODE=$?

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        print_success "Video processing completed successfully!"
        
        # Check for output files
        if [ -f "KeyFrameTrajectory_video_cuda.txt" ]; then
            print_success "Trajectory saved to: KeyFrameTrajectory_video_cuda.txt"
        fi
    else
        print_error "Video processing ended with exit code: $EXIT_CODE"
    fi

    return $EXIT_CODE
}

# Run main function
main "$@"
