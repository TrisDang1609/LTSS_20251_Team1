/*
================================================================================
GPU-ACCELERATED SIFT KEYPOINT REFINEMENT
================================================================================
Optimized for NVIDIA RTX 4060 (Ada Lovelace, 8GB VRAM, 24 SMs)

ARCHITECTURAL OBJECTIVE: Zero CPU Readback
- Input: Candidate keypoints buffer (in VRAM from find_dog_extrema)
- Operation: Sub-pixel localization via quadratic fitting + edge rejection
- Output: Refined keypoints buffer (stays in VRAM for orientation assignment)

CPU REFERENCE LOGIC (sift.cpp):
    bool refine_or_discard_keypoint(Keypoint &kp, const std::vector<Image>
&octave, float contrast_thresh, float edge_thresh)
    {
        // Newton's method for sub-pixel localization (max 5 iterations)
        for (int iter = 0; iter < 5; iter++)
        {
            // Compute Hessian matrix and gradient at current position
            // Solve linear system: dx = -H^-1 * grad
            // Update position: kp.x += dx[0], kp.y += dx[1], kp.scale += dx[2]
            // Check convergence: if (|dx| < 0.5) break
        }

        // Reject low-contrast keypoints
        if (|kp.extremum_val| < contrast_thresh) return false;

        // Reject edge responses (Harris corner detector)
        float H_trace = H[0][0] + H[1][1];
        float H_det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
        float edge_ratio = (H_trace * H_trace) / H_det;
        if (edge_ratio > edge_thresh) return false;

        return true;
    }

PERFORMANCE TARGET (RTX 4060):
    - Candidates: ~50K keypoints
    - Output: ~5K refined keypoints (10% pass rate)
    - Execution time: ~3-5ms (vs ~20ms CPU)
    - Speedup: 4-6×

MEMORY LAYOUT:
    - Input: candidate_buffer [50K × 5 ints] = 1 MB
    - DoG pyramid: ~210 MB (for Hessian/gradient computation)
    - Output: refined_buffer [5K × GPUKeypoint] = 160 KB
    - Valid count: 1 atomic counter (4 bytes)

Ada Lovelace Optimizations:
    - One work-item per candidate (50K threads)
    - Local memory for 3×3×3 neighborhood (27 floats × 4 bytes = 108 bytes)
    - Matrix inversion via analytical formula (2×2 Hessian)
    - Atomic compaction only for valid keypoints (low contention)
================================================================================
*/

// Note: GPUKeypoint is defined in common_kernel.cl

/*
Helper: Get DoG pixel value with clamping
*/
inline float get_dog_value(__global const float *dog_data, int width,
                           int height, int x, int y) {
  x = clamp(x, 0, width - 1);
  y = clamp(y, 0, height - 1);
  return dog_data[y * width + x];
}

/*
Helper: Compute 3×3×3 Hessian matrix and gradient vector at keypoint location.

Returns: Hessian H (3×3 symmetric matrix) and gradient g (3×1 vector)
    H = [d²I/dx²    d²I/dxdy   d²I/dxds  ]
        [d²I/dxdy   d²I/dy²    d²I/dyds  ]
        [d²I/dxds   d²I/dyds   d²I/ds²   ]

    g = [dI/dx]
        [dI/dy]
        [dI/ds]

Uses central differences for derivatives.
*/
void compute_hessian_gradient(
    __global const float *dog_prev, __global const float *dog_curr,
    __global const float *dog_next, int width, int height, int x, int y,
    float *H, // Output: 9 elements (row-major 3×3 matrix)
    float *g  // Output: 3 elements (gradient vector)
) {
  // Read 3×3×3 neighborhood into local memory (27 values)
  float neighbor[3][3][3];

  for (int ds = -1; ds <= 1; ds++) {
    __global const float *dog_scale = (ds == -1)  ? dog_prev
                                      : (ds == 0) ? dog_curr
                                                  : dog_next;
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        neighbor[ds + 1][dx + 1][dy + 1] =
            get_dog_value(dog_scale, width, height, x + dx, y + dy);
      }
    }
  }

  // Center value
  float center = neighbor[1][1][1];

  // First derivatives (gradient) via central differences
  g[0] = (neighbor[1][2][1] - neighbor[1][0][1]) * 0.5f; // dI/dx
  g[1] = (neighbor[1][1][2] - neighbor[1][1][0]) * 0.5f; // dI/dy
  g[2] = (neighbor[2][1][1] - neighbor[0][1][1]) * 0.5f; // dI/ds

  // Second derivatives (Hessian diagonal)
  H[0] = neighbor[1][2][1] + neighbor[1][0][1] - 2.0f * center; // d²I/dx²
  H[4] = neighbor[1][1][2] + neighbor[1][1][0] - 2.0f * center; // d²I/dy²
  H[8] = neighbor[2][1][1] + neighbor[0][1][1] - 2.0f * center; // d²I/ds²

  // Cross derivatives (Hessian off-diagonal, symmetric)
  H[1] = H[3] = (neighbor[1][2][2] - neighbor[1][2][0] - neighbor[1][0][2] +
                 neighbor[1][0][0]) *
                0.25f; // d²I/dxdy
  H[2] = H[6] = (neighbor[2][2][1] - neighbor[2][0][1] - neighbor[0][2][1] +
                 neighbor[0][0][1]) *
                0.25f; // d²I/dxds
  H[5] = H[7] = (neighbor[2][1][2] - neighbor[2][1][0] - neighbor[0][1][2] +
                 neighbor[0][1][0]) *
                0.25f; // d²I/dyds
}

/*
Helper: Invert 3×3 symmetric matrix via analytical formula.

Returns: Inverse matrix H_inv (9 elements, row-major)
         Returns all zeros if matrix is singular (det ≈ 0)
*/
void invert_3x3_matrix(const float *H, float *H_inv) {
  // Compute determinant
  float det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
              H[1] * (H[3] * H[8] - H[5] * H[6]) +
              H[2] * (H[3] * H[7] - H[4] * H[6]);

  // Check for singularity
  if (fabs(det) < 1e-10f) {
    for (int i = 0; i < 9; i++)
      H_inv[i] = 0.0f;
    return;
  }

  float inv_det = 1.0f / det;

  // Compute adjugate matrix (cofactor matrix transposed)
  H_inv[0] = (H[4] * H[8] - H[5] * H[7]) * inv_det;
  H_inv[1] = (H[2] * H[7] - H[1] * H[8]) * inv_det;
  H_inv[2] = (H[1] * H[5] - H[2] * H[4]) * inv_det;
  H_inv[3] = (H[5] * H[6] - H[3] * H[8]) * inv_det;
  H_inv[4] = (H[0] * H[8] - H[2] * H[6]) * inv_det;
  H_inv[5] = (H[2] * H[3] - H[0] * H[5]) * inv_det;
  H_inv[6] = (H[3] * H[7] - H[4] * H[6]) * inv_det;
  H_inv[7] = (H[1] * H[6] - H[0] * H[7]) * inv_det;
  H_inv[8] = (H[0] * H[4] - H[1] * H[3]) * inv_det;
}

/*
Main kernel: Refine candidate keypoints via sub-pixel localization and edge
rejection.

Algorithm:
    1. Newton's method for sub-pixel localization (max 5 iterations):
       - Compute Hessian H and gradient g at current position
       - Solve: offset = -H^-1 * g
       - Update position: (x,y,s) += offset
       - Converge when |offset| < 0.5 pixels

    2. Reject low-contrast keypoints: |extremum_val| < contrast_thresh

    3. Reject edge responses (Harris corner detector):
       - Compute 2×2 spatial Hessian (ignore scale dimension)
       - edge_ratio = (trace² / det) > edge_thresh → reject

    4. Atomic append valid keypoints to output buffer

Kernel parameters:
    - candidate_buffer: Input candidates [50K × 5 ints]
    - num_candidates: Number of candidates to process
    - dog_pyramid_buffers: Array of cl_mem buffers for DoG pyramid
    - octave_widths, octave_heights: Dimensions for each octave
    - imgs_per_octave: Number of DoG scales per octave
    - contrast_thresh: Contrast threshold (typically 0.015)
    - edge_thresh: Edge threshold (typically 10.0)
    - refined_buffer: Output refined keypoints [GPUKeypoint]
    - refined_counter: Atomic counter for valid keypoints

Work-item organization:
    - Global work size: num_candidates (one thread per candidate)
    - Each thread refines one keypoint independently
    - Atomic append to output buffer only if keypoint passes all tests
*/
__kernel void refine_keypoints(
    __global const int *candidate_buffer, __global int *candidate_counter,
    __global const float *dog_prev, __global const float *dog_curr,
    __global const float *dog_next, int width, int height, int octave_idx,
    int target_scale, float contrast_thresh, float edge_thresh, float sigma_min,
    float min_pix_dist, int scales_per_octave,
    __global GPUKeypoint *refined_buffer, __global int *refined_counter) {
  int kp_idx = get_global_id(0);
  if (kp_idx >= *candidate_counter)
    return;

  // Load candidate keypoint
  int base = kp_idx * 5;
  int octave = candidate_buffer[base + 2];
  int scale = candidate_buffer[base + 3];

  // Filter by octave and scale
  if (octave != octave_idx || scale != target_scale)
    return;

  float x = (float)candidate_buffer[base + 0]; // Discrete coordinates
  float y = (float)candidate_buffer[base + 1];
  // float extremum_val = as_float(candidate_buffer[base + 4]); // Unused

  // Newton's method: refine position (max 5 iterations)
  float offset_x = 0.0f, offset_y = 0.0f, offset_s = 0.0f;
  bool converged = false;
  float final_val = 0.0f;

  for (int iter = 0; iter < 5; iter++) {
    // Compute Hessian and gradient at current position
    float H[9], g[3];
    compute_hessian_gradient(dog_prev, dog_curr, dog_next, width, height,
                             (int)round(x), (int)round(y), H, g);

    // Invert Hessian
    float H_inv[9];
    invert_3x3_matrix(H, H_inv);

    // Check if inversion succeeded
    if (H_inv[0] == 0.0f && H_inv[4] == 0.0f && H_inv[8] == 0.0f)
      return; // Singular matrix, discard keypoint

    // Solve: offset = -H^-1 * g
    offset_x = -(H_inv[0] * g[0] + H_inv[1] * g[1] + H_inv[2] * g[2]);
    offset_y = -(H_inv[3] * g[0] + H_inv[4] * g[1] + H_inv[5] * g[2]);
    offset_s = -(H_inv[6] * g[0] + H_inv[7] * g[1] + H_inv[8] * g[2]);

    // Check convergence (CPU uses 0.6)
    if (fabs(offset_x) < 0.6f && fabs(offset_y) < 0.6f &&
        fabs(offset_s) < 0.6f) {
      converged = true;
      float val =
          get_dog_value(dog_curr, width, height, (int)round(x), (int)round(y));
      final_val =
          val + 0.5f * (g[0] * offset_x + g[1] * offset_y + g[2] * offset_s);
      break;
    }

    // Update position
    int dx = (int)(offset_x > 0 ? offset_x + 0.5f : offset_x - 0.5f);
    int dy = (int)(offset_y > 0 ? offset_y + 0.5f : offset_y - 0.5f);
    int ds = (int)(offset_s > 0 ? offset_s + 0.5f : offset_s - 0.5f);

    // If offsets are large but round to 0, we are stuck. Force move or break.
    if (dx == 0 && dy == 0 && ds == 0)
      break;

    x += dx;
    y += dy;
    scale += ds;

    // Check bounds after move
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1 || scale < 1 ||
        scale > scales_per_octave) {
      // Moved out of valid area
      return;
    }
  }

  if (!converged)
    return;

  // Test 1: Reject low-contrast keypoints
  if (fabs(final_val) < contrast_thresh)
    return;

  // Test 2: Reject edge responses (2×2 spatial Hessian only)
  float H[9], g[3];
  compute_hessian_gradient(dog_prev, dog_curr, dog_next, width, height,
                           (int)round(x), (int)round(y), H, g);

  float H_trace = H[0] + H[4]; // d²I/dx² + d²I/dy²
  float H_det =
      H[0] * H[4] - H[1] * H[3]; // det([d²I/dx², d²I/dxdy; d²I/dxdy, d²I/dy²])

  if (H_det <= 0.0f ||
      (H_trace * H_trace / H_det) >
          ((edge_thresh + 1.0f) * (edge_thresh + 1.0f) / edge_thresh))
    return;

  // Compute continuous coordinates in input image space
  float pix_dist = min_pix_dist * pown(2.0f, octave);

  // Keypoint passed all tests! Atomic append to output
  int out_idx = atomic_inc(refined_counter);

  refined_buffer[out_idx].x = (x + offset_x) * pix_dist;
  refined_buffer[out_idx].y = (y + offset_y) * pix_dist;

  float k = pow(2.0f, 1.0f / scales_per_octave);
  refined_buffer[out_idx].sigma =
      sigma_min * pow(k, scale + offset_s) * pown(2.0f, octave);

  refined_buffer[out_idx].octave = octave;
  refined_buffer[out_idx].scale = scale;
}

/*
USAGE PATTERN (Host Code in sift.cpp):

    // After find_dog_extrema kernel
    cl_mem refined_buffer = opencl_api.create_buffer(MAX_KEYPOINTS *
sizeof(GPUKeypoint), ...); cl_mem refined_counter =
opencl_api.create_buffer(sizeof(int), ...);

    int zero = 0;
    opencl_api.write_buffer(refined_counter, sizeof(int), &zero, CL_TRUE);

    cl_kernel kernel = opencl_api.get_kernel("refine_keypoints");

    // Launch refinement for each octave (candidates are sorted by octave)
    for (int oct = 0; oct < num_octaves; oct++)
    {
        auto [width, height] = dog_pyramid.get_dimensions(oct);

        // Set kernel arguments (octave-specific DoG buffers)
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &candidate_buffer);
        clSetKernelArg(kernel, 1, sizeof(int), &num_candidates_in_octave);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &dog_prev);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &dog_curr);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &dog_next);
        // ... more args

        size_t global_work_size = num_candidates_in_octave;
        opencl_api.enqueue_kernel(kernel, 1, &global_work_size, nullptr);
    }

    // Read refined count (STILL IN VRAM, NO D2H YET!)
    int num_refined = 0;
    opencl_api.read_buffer(refined_counter, sizeof(int), &num_refined, CL_TRUE);

PERFORMANCE ANALYSIS:
    - Candidates: 50K keypoints
    - Refined output: ~5K keypoints (90% rejection rate)
    - Computation per keypoint:
        * Newton iterations: 5 × (Hessian + inversion + update) = ~500 FLOPs
        * Edge test: ~20 FLOPs
        * Total: ~520 FLOPs/keypoint
    - Total FLOPs: 50K × 520 = 26M FLOPs (~0.8ms @ 30 TFLOPS)
    - Memory reads: 50K × 27 neighbors = 5.4 MB (~0.03ms @ 150 GB/s)
    - Atomic writes: 5K × 32 bytes = 160 KB (~0.001ms)
    - Expected time: ~3-5ms (vs ~20ms CPU) = 4-6× speedup

ZERO READBACK VALIDATION:
    Input: candidate_buffer (already in VRAM from find_dog_extrema)
    DoG data: dog_pyramid (already in VRAM)
    Output: refined_buffer (stays in VRAM for orientation kernel)
    Result: NO H2D or D2H transfers during refinement!
*/
