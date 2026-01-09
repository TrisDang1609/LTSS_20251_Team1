#!/bin/bash
#
# run_accuracy_validation.sh
#
# Validates GPU implementation accuracy against CPU baseline.
# 
# Usage: ./run_accuracy_validation.sh <video_path> [num_frames]
#
# Example:
#   ./run_accuracy_validation.sh ../video_benchmark/flycam.mp4 100
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "======================================================"
echo "   ORB-SLAM3 CPU vs GPU Accuracy Validation"
echo "======================================================"
echo -e "${NC}"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_path> [num_frames]"
    echo "Example: $0 ../video_benchmark/flycam.mp4 100"
    exit 1
fi

VIDEO_PATH=$1
NUM_FRAMES=${2:-100}  # Default to 100 frames for quick validation

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/validation_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Video: $VIDEO_PATH"
echo "  Frames: $NUM_FRAMES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check executables exist
CPU_EXEC="${SCRIPT_DIR}/validate_cpu_accuracy"
GPU_EXEC="${SCRIPT_DIR}/validate_gpu_accuracy"

if [ ! -f "$CPU_EXEC" ]; then
    echo -e "${RED}Error: CPU validation executable not found: $CPU_EXEC${NC}"
    echo "Please build the project first with: cd build && make validate_cpu_accuracy"
    exit 1
fi

if [ ! -f "$GPU_EXEC" ]; then
    echo -e "${RED}Error: GPU validation executable not found: $GPU_EXEC${NC}"
    echo "Please build the project first with: cd build && make validate_gpu_accuracy"
    exit 1
fi

# ================================================
# Step 1: Run CPU Validation
# ================================================
echo -e "${BLUE}"
echo "======================================================"
echo "   Step 1: Running CPU Accuracy Export"
echo "======================================================"
echo -e "${NC}"

"$CPU_EXEC" "$VIDEO_PATH" "$NUM_FRAMES" "$OUTPUT_DIR" 2>&1 | tee "${OUTPUT_DIR}/cpu_validation.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}CPU validation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}CPU validation complete!${NC}"
echo ""

# ================================================
# Step 2: Run GPU Validation
# ================================================
echo -e "${BLUE}"
echo "======================================================"
echo "   Step 2: Running GPU Accuracy Export"
echo "======================================================"
echo -e "${NC}"

"$GPU_EXEC" "$VIDEO_PATH" "$NUM_FRAMES" "$OUTPUT_DIR" 2>&1 | tee "${OUTPUT_DIR}/gpu_validation.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}GPU validation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}GPU validation complete!${NC}"
echo ""

# ================================================
# Step 3: Compare Results
# ================================================
echo -e "${BLUE}"
echo "======================================================"
echo "   Step 3: Comparing CPU vs GPU Results"
echo "======================================================"
echo -e "${NC}"

# Check if Python script exists
COMPARE_SCRIPT="${SCRIPT_DIR}/../evaluation/compare_accuracy.py"

if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo -e "${RED}Error: Comparison script not found: $COMPARE_SCRIPT${NC}"
    exit 1
fi

python3 "$COMPARE_SCRIPT" "$OUTPUT_DIR" 2>&1 | tee "${OUTPUT_DIR}/comparison.log"

echo ""
echo -e "${GREEN}"
echo "======================================================"
echo "   Validation Complete!"
echo "======================================================"
echo -e "${NC}"

echo "Output files:"
echo "  CPU Log:        ${OUTPUT_DIR}/cpu_accuracy.log"
echo "  GPU Log:        ${OUTPUT_DIR}/gpu_accuracy.log"
echo "  Comparison:     ${OUTPUT_DIR}/accuracy_comparison.json"
echo "  CPU Summary:    ${OUTPUT_DIR}/cpu_summary.json"
echo "  GPU Summary:    ${OUTPUT_DIR}/gpu_summary.json"
echo ""
