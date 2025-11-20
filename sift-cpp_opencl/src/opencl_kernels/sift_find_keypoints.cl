/*
================================================================================
GPU-ACCELERATED SIFT KEYPOINT DETECTION
================================================================================
Optimized for NVIDIA RTX 4060 (Ada Lovelace Architecture, 8GB VRAM)

ARCHITECTURAL OBJECTIVE: Zero Round-Trips
- Input: DoG pyramid (already in VRAM)
- Operation: Parallel 26-neighbor extremum detection (GPU-only)
- Output: Candidate keypoints buffer (stays in VRAM for refinement)

CPU REFERENCE LOGIC (sift.cpp):
    std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid &dog_pyramid,
                                         float contrast_thresh, float
edge_thresh)
    {
        std::vector<Keypoint> keypoints;
        for (int i = 0; i < dog_pyramid.num_octaves; i++)
        {
            const std::vector<Image> &octave = dog_pyramid.octaves[i];
            for (int j = 1; j < dog_pyramid.imgs_per_octave - 1; j++)
            {
                const Image &img = octave[j];
                for (int x = 1; x < img.width - 1; x++)
                {
                    for (int y = 1; y < img.height - 1; y++)
                    {
                        if (std::abs(img.get_pixel(x, y, 0)) < 0.8 *
contrast_thresh) continue; if (point_is_extremum(octave, j, x, y))
                        {
                            Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                            bool kp_is_valid = refine_or_discard_keypoint(kp,
octave, contrast_thresh, edge_thresh); if (kp_is_valid) keypoints.push_back(kp);
                        }
                    }
                }
            }
        }
        return keypoints;
    }

GPU IMPLEMENTATION STRATEGY:
    Phase 1: Extremum detection (this kernel)
        - One work-item per pixel in DoG scale
        - Check 26 neighbors (3×3×3 cube in scale-space)
        - Write candidate keypoints to output buffer

    Phase 2: Sub-pixel refinement (separate kernel)
        - Quadratic fit via Newton's method
        - Edge response test (Harris corner detector)
        - Atomic compaction of valid keypoints

PERFORMANCE TARGETS (RTX 4060):
    - DoG pyramid: ~210 MB (3840×2160 + 1920×1080 + ... across octaves)
    - Pixels processed: ~10M pixels (all scales combined)
    - Extrema found: ~50K candidates → ~5K valid keypoints (10% pass rate)
    - Execution time: ~8-10ms (vs ~150ms CPU)
    - Speedup: 15-18× faster

MEMORY LAYOUT:
    - DoG pyramid: octave_widths × octave_heights × scales_per_octave
    - Candidate buffer: max 50K keypoints × 32 bytes = 1.6 MB
    - Counter buffer: 1 atomic int (4 bytes) for output index

Ada Lovelace Optimizations:
    - Coalesced memory access (sequential x,y reads)
    - Early exit (80% rejection from contrast threshold)
    - Register-heavy computation (no shared memory needed)
    - Warp divergence minimized (neighbor checks parallel)
================================================================================
*/

/*
Helper function: Get DoG pixel value with boundary clamping.
Equivalent to dog_pyramid.octaves[octave][scale].get_pixel(x, y, 0).

Parameters:
- dog_data: DoG image buffer
- width, height: Image dimensions
- x, y: Pixel coordinates

Returns: DoG value at (x,y) with clamped boundaries
*/
inline float get_dog_pixel(__global const float *dog_data, int width,
                           int height, int x, int y) {
  // Clamp to boundaries
  x = clamp(x, 0, width - 1);
  y = clamp(y, 0, height - 1);
  return dog_data[y * width + x];
}

/*
Helper function: Check if pixel is 26-neighbor extremum in scale-space.
Equivalent to point_is_extremum() in sift.cpp.

Checks 3×3×3 cube:
- 9 neighbors in scale below (prev_scale)
- 8 neighbors in current scale (excluding center)
- 9 neighbors in scale above (next_scale)

Parameters:
- dog_prev, dog_curr, dog_next: Three consecutive DoG scales
- width, height: Image dimensions
- x, y: Center pixel coordinates

Returns: 1 if local extremum (max or min), 0 otherwise
*/
inline int is_dog_extremum(__global const float *dog_prev,
                           __global const float *dog_curr,
                           __global const float *dog_next, int width,
                           int height, int x, int y) {
  float center_val = get_dog_pixel(dog_curr, width, height, x, y);
  int is_max = 1;
  int is_min = 1;

  // Check 3×3×3 neighborhood
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      // Previous scale (9 neighbors)
      float neighbor = get_dog_pixel(dog_prev, width, height, x + dx, y + dy);
      if (neighbor > center_val)
        is_max = 0;
      if (neighbor < center_val)
        is_min = 0;

      // Next scale (9 neighbors)
      neighbor = get_dog_pixel(dog_next, width, height, x + dx, y + dy);
      if (neighbor > center_val)
        is_max = 0;
      if (neighbor < center_val)
        is_min = 0;

      // Current scale (8 neighbors, skip center)
      if (dx != 0 || dy != 0) {
        neighbor = get_dog_pixel(dog_curr, width, height, x + dx, y + dy);
        if (neighbor > center_val)
          is_max = 0;
        if (neighbor < center_val)
          is_min = 0;
      }

      // Early exit if neither max nor min
      if (!is_max && !is_min)
        return 0;
    }
  }

  return (is_max || is_min);
}

/*
Main kernel: Find DoG extrema (candidate keypoints) in scale-space.

This kernel performs parallel extremum detection across entire DoG pyramid.
Each work-item processes one pixel in one DoG scale.

Kernel parameters:
- dog_prev: DoG scale i-1 buffer
- dog_curr: DoG scale i buffer
- dog_next: DoG scale i+1 buffer
- width, height: Dimensions of current octave
- octave_idx: Current octave index
- scale_idx: Current scale index within octave
- contrast_thresh: Contrast threshold (typically 0.015)
- candidate_buffer: Output buffer for candidate keypoints
- candidate_counter: Atomic counter for output indexing

Work-item organization:
- Global work size: (width-2, height-2) [exclude borders]
- Each work-item checks one pixel for extremum
- Atomic append to candidate_buffer if extremum found

Output format (candidate keypoints):
    struct {
        int i, j;          // Discrete coordinates
        int octave, scale; // Pyramid location
        float val;         // DoG extremum value
    };

Note: This kernel only detects candidates. Refinement (sub-pixel localization,
edge rejection) is performed in a separate kernel to maintain GPU occupancy.
*/
__kernel void find_keypoints(__global const float *dog_prev,
                             __global const float *dog_curr,
                             __global const float *dog_next,
                             __global int *candidate_buffer,
                             __global int *candidate_counter, int width,
                             int height, int octave, int scale,
                             float contrast_thresh, int max_candidates) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  // 1. Boundary check: pure_linear loops from 1 to width-1
  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
    return;
  }

  float val = dog_curr[y * width + x];

  // 2. Pre-filter Threshold: pure_linear uses 0.8 * contrast_thresh
  // (pure_linear/sift.cpp line 230)
  if (fabs(val) < 0.8f * contrast_thresh) {
    return;
  }

  // 3. Extremum Check: Match pure_linear logic EXACTLY
  // pure_linear/sift.cpp line 81:
  // if (neighbor > val) is_max = false;
  // if (neighbor < val) is_min = false;
  // This implies if neighbor == val, it can still be an extremum.

  bool is_max = true;
  bool is_min = true;

  // Check 26 neighbors
  // We unroll loops manually or use arrays to ensure we check all

  // Prev scale
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      float neighbor = dog_prev[(y + dy) * width + (x + dx)];
      if (neighbor > val)
        is_max = false;
      if (neighbor < val)
        is_min = false;
    }
  }

  // Curr scale
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0)
        continue;
      float neighbor = dog_curr[(y + dy) * width + (x + dx)];
      if (neighbor > val)
        is_max = false;
      if (neighbor < val)
        is_min = false;
    }
  }

  // Next scale
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      float neighbor = dog_next[(y + dy) * width + (x + dx)];
      if (neighbor > val)
        is_max = false;
      if (neighbor < val)
        is_min = false;
    }
  }

  // pure_linear/sift.cpp line 99: if (!is_min && !is_max) return false;
  if (is_max || is_min) {
    int old_idx = atomic_inc(candidate_counter);
    if (old_idx < max_candidates) {
      candidate_buffer[old_idx * 5 + 0] = x;
      candidate_buffer[old_idx * 5 + 1] = y;
      candidate_buffer[old_idx * 5 + 2] = octave;
      candidate_buffer[old_idx * 5 + 3] = scale;
      candidate_buffer[old_idx * 5 + 4] = 0; // padding
    }
  }
}

/*
USAGE PATTERN (Host Code in sift.cpp):

    // Allocate candidate buffer (max 50K keypoints)
    cl_mem candidate_buffer = opencl_api.create_buffer(50000 * 5 * sizeof(int),
...); cl_mem candidate_counter = opencl_api.create_buffer(sizeof(int), ...);

    // Initialize counter to 0
    int zero = 0;
    opencl_api.write_buffer(candidate_counter, sizeof(int), &zero, CL_TRUE);

    // Launch kernel for each scale in each octave
    cl_kernel kernel = opencl_api.get_kernel("find_dog_extrema");

    for (int oct = 0; oct < num_octaves; oct++)
    {
        auto [width, height] = dog_pyramid.get_dimensions(oct);

        for (int scale = 1; scale < dog_pyramid.imgs_per_octave - 1; scale++)
        {
            cl_mem dog_prev = dog_pyramid.get_buffer(oct, scale - 1);
            cl_mem dog_curr = dog_pyramid.get_buffer(oct, scale);
            cl_mem dog_next = dog_pyramid.get_buffer(oct, scale + 1);

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &dog_prev);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &dog_curr);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &dog_next);
            clSetKernelArg(kernel, 3, sizeof(int), &width);
            clSetKernelArg(kernel, 4, sizeof(int), &height);
            clSetKernelArg(kernel, 5, sizeof(int), &oct);
            clSetKernelArg(kernel, 6, sizeof(int), &scale);
            clSetKernelArg(kernel, 7, sizeof(float), &contrast_thresh);
            clSetKernelArg(kernel, 8, sizeof(cl_mem), &candidate_buffer);
            clSetKernelArg(kernel, 9, sizeof(cl_mem), &candidate_counter);

            size_t global_work_size[2] = {width - 2, height - 2};
            opencl_api.enqueue_kernel(kernel, 2, global_work_size, nullptr);
        }
    }

    // Read back candidate count
    int num_candidates = 0;
    opencl_api.read_buffer(candidate_counter, sizeof(int), &num_candidates,
CL_TRUE);

    // Read back candidate keypoints
    std::vector<int> candidates(num_candidates * 5);
    opencl_api.read_buffer(candidate_buffer, num_candidates * 5 * sizeof(int),
                          candidates.data(), CL_TRUE);

MEMORY FOOTPRINT:
    - DoG pyramid: ~210 MB (input, already in VRAM)
    - Candidate buffer: ~1.6 MB (50K × 5 × 4 bytes)
    - Counter: 4 bytes
    Total: ~212 MB (2.6% of 8GB VRAM)

PERFORMANCE ANALYSIS:
    - Work-items: ~10M (all pixels in all scales)
    - Early exits: ~8M (80% contrast rejection)
    - Extremum checks: ~2M (20% pass contrast)
    - Candidates found: ~50K (2.5% are extrema)
    - Execution time: ~8-10ms

    Breakdown:
        - Memory bandwidth: 210 MB × 3 reads = 630 MB (~4ms @ 150 GB/s)
        - Computation: 26 comparisons × 2M pixels = 52M ops (~2ms @ 30 TFLOPS)
        - Atomic writes: 50K × 5 ints = 250 KB (~0.1ms, low contention)
        - Overhead: ~2-3ms (kernel launch, synchronization)
*/

/*
USAGE PATTERN (Host Code in sift.cpp):

    // Allocate candidate buffer (max 50K keypoints)
    cl_mem candidate_buffer = opencl_api.create_buffer(50000 * 5 * sizeof(int),
...); cl_mem candidate_counter = opencl_api.create_buffer(sizeof(int), ...);

    // Initialize counter to 0
    int zero = 0;
    opencl_api.write_buffer(candidate_counter, sizeof(int), &zero, CL_TRUE);

    // Launch kernel for each scale in each octave
    cl_kernel kernel = opencl_api.get_kernel("find_keypoints");

    for (int oct = 0; oct < num_octaves; oct++)
    {
        auto [width, height] = dog_pyramid.get_dimensions(oct);

        for (int scale = 1; scale < dog_pyramid.imgs_per_octave - 1; scale++)
        {
            cl_mem dog_prev = dog_pyramid.get_buffer(oct, scale - 1);
            cl_mem dog_curr = dog_pyramid.get_buffer(oct, scale);
            cl_mem dog_next = dog_pyramid.get_buffer(oct, scale + 1);

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &dog_prev);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &dog_curr);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &dog_next);
            clSetKernelArg(kernel, 3, sizeof(int), &width);
            clSetKernelArg(kernel, 4, sizeof(int), &height);
            clSetKernelArg(kernel, 5, sizeof(int), &oct);
            clSetKernelArg(kernel, 6, sizeof(int), &scale);
            clSetKernelArg(kernel, 7, sizeof(float), &contrast_thresh);
            clSetKernelArg(kernel, 8, sizeof(cl_mem), &candidate_buffer);
            clSetKernelArg(kernel, 9, sizeof(cl_mem), &candidate_counter);

            size_t global_work_size[2] = {width, height};
            opencl_api.enqueue_kernel(kernel, 2, global_work_size, nullptr);
        }
    }

    // Read back candidate count
    int num_candidates = 0;
    opencl_api.read_buffer(candidate_counter, sizeof(int), &num_candidates,
CL_TRUE);

    // Read back candidate keypoints
    std::vector<int> candidates(num_candidates * 5);
    opencl_api.read_buffer(candidate_buffer, num_candidates * 5 * sizeof(int),
                          candidates.data(), CL_TRUE);

MEMORY FOOTPRINT:
    - DoG pyramid: ~210 MB (input, already in VRAM)
    - Candidate buffer: ~1.6 MB (50K × 5 × 4 bytes)
    - Counter: 4 bytes
    Total: ~212 MB (2.6% of 8GB VRAM)

PERFORMANCE ANALYSIS:
    - Work-items: ~10M (all pixels in all scales)
    - Early exits: ~8M (80% contrast rejection)
    - Extremum checks: ~2M (20% pass contrast)
    - Candidates found: ~50K (2.5% are extrema)
    - Execution time: ~8-10ms

    Breakdown:
        - Memory bandwidth: 210 MB × 3 reads = 630 MB (~4ms @ 150 GB/s)
        - Computation: 26 comparisons × 2M pixels = 52M ops (~2ms @ 30 TFLOPS)
        - Atomic writes: 50K × 5 ints = 250 KB (~0.1ms, low contention)
        - Overhead: ~2-3ms (kernel launch, synchronization)
*/