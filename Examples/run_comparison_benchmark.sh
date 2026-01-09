#!/bin/bash
#
# ORB-SLAM3 GPU vs CPU Comparative Benchmark
# 
# This script runs both GPU and CPU benchmarks on the same video
# and produces a comparison report.
#
# Usage: ./run_comparison_benchmark.sh [video_path] [num_frames]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_PATH="${1:-$SCRIPT_DIR/../video_benchmark/flycam.mp4}"
NUM_FRAMES="${2:--1}"  # -1 means all frames
OUTPUT_DIR="$SCRIPT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   ORB-SLAM3 GPU vs CPU Comparative Benchmark${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""
echo -e "Video: ${GREEN}$VIDEO_PATH${NC}"
echo -e "Frames: ${GREEN}$NUM_FRAMES${NC} (-1 = all)"
echo -e "Output: ${GREEN}$OUTPUT_DIR${NC}"
echo ""

# Check if executables exist
GPU_EXEC="$SCRIPT_DIR/benchmark_gpu_pipeline"
CPU_EXEC="$SCRIPT_DIR/benchmark_cpu_baseline"

if [ ! -f "$GPU_EXEC" ]; then
    echo -e "${RED}Error: GPU benchmark executable not found: $GPU_EXEC${NC}"
    echo "Please build the project first with: cd build && make -j\$(nproc)"
    exit 1
fi

if [ ! -f "$CPU_EXEC" ]; then
    echo -e "${RED}Error: CPU benchmark executable not found: $CPU_EXEC${NC}"
    echo "Please build the project first with: cd build && make -j\$(nproc)"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo -e "${RED}Error: Video file not found: $VIDEO_PATH${NC}"
    exit 1
fi

# GPU Benchmark
echo -e "${YELLOW}======================================================${NC}"
echo -e "${YELLOW}   Step 1: Running GPU Benchmark${NC}"
echo -e "${YELLOW}======================================================${NC}"
echo ""

GPU_JSON="$OUTPUT_DIR/gpu_result_$TIMESTAMP.json"
GPU_LOG="$OUTPUT_DIR/gpu_log_$TIMESTAMP.txt"

if [ "$NUM_FRAMES" -gt 0 ] 2>/dev/null; then
    # num_frames is a positive number (benchmark_gpu_pipeline doesn't support frame limit directly)
    "$GPU_EXEC" "$VIDEO_PATH" 2>&1 | tee "$GPU_LOG"
else
    "$GPU_EXEC" "$VIDEO_PATH" 2>&1 | tee "$GPU_LOG"
fi

# Extract GPU metrics from log
GPU_FPS=$(grep -oP "Throughput: \K[0-9.]+" "$GPU_LOG" || echo "N/A")
GPU_MEAN=$(grep -oP "Mean: \K[0-9.]+" "$GPU_LOG" | tail -1 || echo "N/A")
GPU_P95=$(grep -oP "P95:\s+\K[0-9.]+" "$GPU_LOG" || echo "N/A")
GPU_KEYPOINTS=$(grep -oP "Average per frame: \K[0-9]+" "$GPU_LOG" || echo "N/A")

echo ""
echo -e "${GREEN}GPU Benchmark Complete!${NC}"
echo ""

# CPU Benchmark
echo -e "${YELLOW}======================================================${NC}"
echo -e "${YELLOW}   Step 2: Running CPU Benchmark${NC}"
echo -e "${YELLOW}======================================================${NC}"
echo ""

CPU_JSON="$OUTPUT_DIR/cpu_result_$TIMESTAMP.json"
CPU_LOG="$OUTPUT_DIR/cpu_log_$TIMESTAMP.txt"

"$CPU_EXEC" "$VIDEO_PATH" "$NUM_FRAMES" "$CPU_JSON" 2>&1 | tee "$CPU_LOG"

# Extract CPU metrics from log
CPU_FPS=$(grep -oP "Throughput: \K[0-9.]+" "$CPU_LOG" || echo "N/A")
CPU_MEAN=$(grep -oP "Mean: \K[0-9.]+" "$CPU_LOG" | tail -1 || echo "N/A")
CPU_P95=$(grep -oP "P95:\s+\K[0-9.]+" "$CPU_LOG" || echo "N/A")
CPU_KEYPOINTS=$(grep -oP "Average per frame: \K[0-9]+" "$CPU_LOG" || echo "N/A")

echo ""
echo -e "${GREEN}CPU Benchmark Complete!${NC}"
echo ""

# Comparison Report
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   Step 3: Performance Comparison Report${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# Calculate speedup
if [[ "$GPU_FPS" != "N/A" && "$CPU_FPS" != "N/A" ]]; then
    SPEEDUP=$(echo "scale=2; $GPU_FPS / $CPU_FPS" | bc)
else
    SPEEDUP="N/A"
fi

if [[ "$GPU_MEAN" != "N/A" && "$CPU_MEAN" != "N/A" ]]; then
    LATENCY_REDUCTION=$(echo "scale=2; ($CPU_MEAN - $GPU_MEAN) / $CPU_MEAN * 100" | bc)
else
    LATENCY_REDUCTION="N/A"
fi

# Create comparison report
REPORT_FILE="$OUTPUT_DIR/comparison_report_$TIMESTAMP.txt"

cat > "$REPORT_FILE" << EOF
================================================================================
                    ORB-SLAM3 GPU vs CPU Performance Comparison
================================================================================
Date: $(date)
Video: $VIDEO_PATH
================================================================================

                              GPU                CPU              Speedup
--------------------------------------------------------------------------------
Throughput (FPS):        ${GPU_FPS} FPS         ${CPU_FPS} FPS           ${SPEEDUP}x
Mean Latency (ms):       ${GPU_MEAN} ms            ${CPU_MEAN} ms
P95 Latency (ms):        ${GPU_P95} ms            ${CPU_P95} ms
Avg Keypoints:           ${GPU_KEYPOINTS}                ${CPU_KEYPOINTS}

================================================================================
                              Analysis Summary
================================================================================

EOF

if [[ "$SPEEDUP" != "N/A" ]]; then
    echo "The GPU implementation is ${SPEEDUP}x faster than the CPU baseline." >> "$REPORT_FILE"
    if [[ "$LATENCY_REDUCTION" != "N/A" ]]; then
        echo "Latency reduction: ${LATENCY_REDUCTION}%" >> "$REPORT_FILE"
    fi
fi

echo "" >> "$REPORT_FILE"
echo "Detailed results saved to:" >> "$REPORT_FILE"
echo "  GPU Log: $GPU_LOG" >> "$REPORT_FILE"
echo "  CPU Log: $CPU_LOG" >> "$REPORT_FILE"
echo "  CPU JSON: $CPU_JSON" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Print the report
cat "$REPORT_FILE"

echo ""
echo -e "${GREEN}Comparison report saved to: $REPORT_FILE${NC}"
echo ""

# Summary table
echo -e "${BLUE}┌─────────────────────────────────────────────────────────────────┐${NC}"
echo -e "${BLUE}│                    PERFORMANCE SUMMARY                         │${NC}"
echo -e "${BLUE}├─────────────────────────────────────────────────────────────────┤${NC}"
printf "${BLUE}│${NC} %-20s │ %15s │ %15s ${BLUE}│${NC}\n" "Metric" "GPU" "CPU"
echo -e "${BLUE}├─────────────────────────────────────────────────────────────────┤${NC}"
printf "${BLUE}│${NC} %-20s │ %12s FPS │ %12s FPS ${BLUE}│${NC}\n" "Throughput" "$GPU_FPS" "$CPU_FPS"
printf "${BLUE}│${NC} %-20s │ %13s ms │ %13s ms ${BLUE}│${NC}\n" "Mean Latency" "$GPU_MEAN" "$CPU_MEAN"
printf "${BLUE}│${NC} %-20s │ %13s ms │ %13s ms ${BLUE}│${NC}\n" "P95 Latency" "$GPU_P95" "$CPU_P95"
echo -e "${BLUE}├─────────────────────────────────────────────────────────────────┤${NC}"
echo -e "${BLUE}│${NC}                     ${GREEN}SPEEDUP: ${SPEEDUP}x${NC}                           ${BLUE}│${NC}"
echo -e "${BLUE}└─────────────────────────────────────────────────────────────────┘${NC}"
echo ""
