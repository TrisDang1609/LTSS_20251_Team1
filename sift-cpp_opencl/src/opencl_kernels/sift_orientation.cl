/*
================================================================================
GPU-ACCELERATED SIFT ORIENTATION ASSIGNMENT
================================================================================
Optimized for NVIDIA RTX 4060 (Ada Lovelace, 8GB VRAM, 24 SMs)

ARCHITECTURAL OBJECTIVE: Zero CPU Readback
- Input: Refined keypoints buffer (in VRAM from refine_keypoints)
- Operation: Build orientation histogram, find dominant peaks (GPU-only)
- Output: Keypoints with assigned orientations (stays in VRAM for descriptors)

CPU REFERENCE LOGIC (sift.cpp):
    std::vector<float> find_keypoint_orientations(Keypoint &kp,
                                                  const ScaleSpacePyramid
&grad_pyramid, float lambda_ori, float lambda_desc)
    {
        // Build 36-bin orientation histogram
        float hist[36] = {};
        float patch_sigma = lambda_ori * kp.sigma;
        float patch_radius = 3 * patch_sigma;

        for (int x = x_start; x <= x_end; x++) {
            for (int y = y_start; y <= y_end; y++) {
                float gx = img_grad.get_pixel(x, y, 0);  // magnitude
                float gy = img_grad.get_pixel(x, y, 1);  // orientation
                float grad_norm = sqrt(gx*gx + gy*gy);
                float weight = exp(-dist² / (2*patch_sigma²));
                float theta = atan2(gy, gx);
                int bin = round(36/(2π) * theta) % 36;
                hist[bin] += weight * grad_norm;
            }
        }

        smooth_histogram(hist);  // 6× box filter

        // Find peaks >= 0.8 * max
        std::vector<float> orientations;
        for (int j = 0; j < 36; j++) {
            if (hist[j] >= 0.8 * ori_max && is_local_max(hist, j)) {
                float theta = refine_peak_position(hist, j);  // Parabolic
interpolation orientations.push_back(theta);
            }
        }
        return orientations;
    }

PERFORMANCE TARGET (RTX 4060):
    - Input: ~5K refined keypoints
    - Patch size: ~30×30 pixels average
    - Output: ~7K keypoint-orientation pairs (1.4 orientations/keypoint average)
    - Execution time: ~5-8ms (vs ~50ms CPU)
    - Speedup: 6-10×

MEMORY LAYOUT:
    - Input: refined_buffer [5K × GPUKeypoint] = 160 KB
    - Gradient pyramid: ~253 MB (magnitude + orientation interleaved)
    - Histograms: [5K × 36 floats] = 720 KB (local memory per work-item)
    - Output: oriented_keypoints [7K × (GPUKeypoint + float orientation)] = 252
KB

Ada Lovelace Optimizations:
    - One work-item per keypoint (5K threads, LOW occupancy issue)
    - Local memory for 36-bin histogram (144 bytes, fits in L1 cache)
    - Atomic-free histogram (each thread owns its histogram)
    - Dynamic output allocation via atomic append
    - Coalesced reads from gradient pyramid
================================================================================
*/

// Note: GPUKeypoint and OrientedKeypoint are defined in common_kernel.cl

#define N_BINS 36
#define TWO_PI 6.28318530718f

/*
Helper: Smooth histogram with 6 iterations of 3-tap box filter.
Equivalent to smooth_histogram() in sift.cpp.
*/
void smooth_histogram_local(float *hist) {
  for (int iter = 0; iter < 6; iter++) {
    float prev = hist[N_BINS - 1];
    float temp;

    for (int i = 0; i < N_BINS; i++) {
      temp = hist[i];
      hist[i] = (prev + hist[i] + hist[(i + 1) % N_BINS]) / 3.0f;
      prev = temp;
    }
  }
}

/*
Helper: Check if bin is local maximum in histogram.
*/
inline int is_local_max(const float *hist, int bin) {
  int prev_bin = (bin - 1 + N_BINS) % N_BINS;
  int next_bin = (bin + 1) % N_BINS;
  return (hist[bin] >= hist[prev_bin] && hist[bin] >= hist[next_bin]);
}

/*
Helper: Refine peak position via parabolic interpolation.
Returns refined orientation angle in [0, 2π].
*/
float refine_peak_position(const float *hist, int bin) {
  int prev_bin = (bin - 1 + N_BINS) % N_BINS;
  int next_bin = (bin + 1) % N_BINS;

  float prev = hist[prev_bin];
  float curr = hist[bin];
  float next = hist[next_bin];

  // Parabolic interpolation: offset = (prev - next) / (2*(prev - 2*curr +
  // next))
  float denom = 2.0f * (prev - 2.0f * curr + next);
  float offset = (fabs(denom) > 1e-6f) ? (prev - next) / denom : 0.0f;

  // Compute refined angle
  // Match CPU logic: theta = 2*PI*(bin + 1 + offset)/N_BINS
  float theta = TWO_PI * (bin + 1.0f + offset) / N_BINS;

  // Normalize to [0, 2π]
  if (theta < 0.0f)
    theta += TWO_PI;
  if (theta >= TWO_PI)
    theta -= TWO_PI;

  return theta;
}

/*
Helper: Get gradient magnitude and orientation at pixel (x, y).
Gradient pyramid stores interleaved [magnitude, orientation] per pixel.

CRITICAL FIX #3: Added bounds clamping to prevent out-of-bounds reads
when keypoints are near image borders.
*/
void get_gradient(__global const float *grad_img, int width, int height, int x,
                  int y, float *magnitude, float *orientation) {
  // CRITICAL: Clamp coordinates to prevent OOB access
  x = clamp(x, 0, width - 1);
  y = clamp(y, 0, height - 1);

  // Read RAW gradients (gx, gy) from gradient pyramid
  // This matches CPU implementation where gradient pyramid stores raw gradients
  int idx = 2 * (y * width + x);
  float gx = grad_img[idx + 0];
  float gy = grad_img[idx + 1];

  // Compute magnitude and orientation on-the-fly
  // Matches CPU: grad_norm = std::sqrt(gx*gx + gy*gy); theta = std::atan2(gy,
  // gx);
  *magnitude = sqrt(gx * gx + gy * gy);
  *orientation = atan2(gy, gx); // Range: [-π, +π]
}

/*
Main kernel: Compute dominant orientations for each refined keypoint.

Algorithm:
    1. For each keypoint, extract patch around it (radius = 3 * lambda_ori *
sigma)
    2. Build 36-bin orientation histogram weighted by:
       - Gradient magnitude
       - Gaussian weight (distance from keypoint center)
    3. Smooth histogram 6× with box filter
    4. Find all peaks >= 0.8 × max_bin
    5. For each peak, create new oriented keypoint via atomic append

Kernel parameters:
    - refined_keypoints: Input refined keypoints [GPUKeypoint]
    - num_refined: Number of refined keypoints
    - grad_pyramid_buffer: Gradient pyramid buffer (2-channel: magnitude,
orientation)
    - octave_widths, octave_heights: Gradient pyramid dimensions
    - lambda_ori: Orientation patch scale factor (typically 1.5)
    - lambda_desc: Descriptor patch scale factor (typically 1.5, for border
check)
    - min_pix_dist: Minimum pixel distance (typically 0.5)
    - oriented_keypoints: Output keypoints with orientations [OrientedKeypoint]
    - oriented_counter: Atomic counter for output

Work-item organization:
    - Global work size: num_refined (one thread per keypoint)
    - Each thread builds its own histogram (no atomics!)
    - Atomic append to output only for detected peaks
*/
__kernel void
compute_orientations(__global const GPUKeypoint *refined_keypoints,
                     __global int *refined_counter,
                     __global const float *grad_pyramid_buffer, int width,
                     int height, int octave_idx, int scale_idx,
                     float lambda_ori, float lambda_desc, float min_pix_dist,
                     __global OrientedKeypoint *oriented_keypoints,
                     __global int *oriented_counter) {
  int kp_idx = get_global_id(0);
  if (kp_idx >= *refined_counter)
    return;

  // Load keypoint
  GPUKeypoint kp = refined_keypoints[kp_idx];

  // Filter by octave and scale
  if (kp.octave != octave_idx || kp.scale != scale_idx)
    return;

  // Compute pixel distance at this octave
  float pix_dist = min_pix_dist * pow(2.0f, (float)kp.octave);

  // Check border constraint (discard if too close to edge)
  float min_dist_from_border =
      fmin(fmin(kp.x, kp.y),
           fmin(pix_dist * width - kp.x, pix_dist * height - kp.y));

  if (min_dist_from_border <= sqrt(2.0f) * lambda_desc * kp.sigma)
    return; // Too close to border, discard

  // Build orientation histogram in local memory (36 bins)
  float hist[N_BINS];
  for (int i = 0; i < N_BINS; i++)
    hist[i] = 0.0f;

  // Patch parameters
  float patch_sigma = lambda_ori * kp.sigma;
  float patch_radius = 3.0f * patch_sigma;

  int x_start = (int)round((kp.x - patch_radius) / pix_dist);
  int x_end = (int)round((kp.x + patch_radius) / pix_dist);
  int y_start = (int)round((kp.y - patch_radius) / pix_dist);
  int y_end = (int)round((kp.y + patch_radius) / pix_dist);

  // Accumulate weighted gradients into histogram
  for (int x = x_start; x <= x_end; x++) {
    for (int y = y_start; y <= y_end; y++) {
      // Get gradient at this pixel
      float magnitude, orientation;
      get_gradient(grad_pyramid_buffer, width, height, x, y, &magnitude,
                   &orientation);

      // Compute Gaussian weight based on distance from keypoint center
      float dx = x * pix_dist - kp.x;
      float dy = y * pix_dist - kp.y;
      float dist_sq = dx * dx + dy * dy;
      float weight = exp(-dist_sq / (2.0f * patch_sigma * patch_sigma));

      // Normalize orientation to [0, 2π]
      float theta = orientation;
      if (theta < 0.0f)
        theta += TWO_PI;

      // Compute histogram bin
      int bin = (int)round(N_BINS / TWO_PI * theta) % N_BINS;

      // Accumulate weighted magnitude
      hist[bin] += weight * magnitude;
    }
  }

  // Smooth histogram (6 iterations of 3-tap box filter)
  smooth_histogram_local(hist);

  // Find maximum bin value
  float ori_max = 0.0f;
  for (int i = 0; i < N_BINS; i++) {
    if (hist[i] > ori_max)
      ori_max = hist[i];
  }

  // Find all peaks >= 0.8 * ori_max
  float ori_thresh = 0.8f * ori_max;

  for (int bin = 0; bin < N_BINS; bin++) {
    if (hist[bin] >= ori_thresh && is_local_max(hist, bin)) {
      // Refine peak position via parabolic interpolation
      float theta = refine_peak_position(hist, bin);

      // Atomic append to output
      int out_idx = atomic_inc(oriented_counter);

      oriented_keypoints[out_idx].kp = kp;
      oriented_keypoints[out_idx].orientation = theta;
    }
  }
}

/*
USAGE PATTERN (Host Code in sift.cpp):

    // After refine_keypoints kernel
    cl_mem oriented_buffer = opencl_api.create_buffer(MAX_ORIENTED_KPS *
sizeof(OrientedKeypoint), ...); cl_mem oriented_counter =
opencl_api.create_buffer(sizeof(int), ...);

    int zero = 0;
    opencl_api.write_buffer(oriented_counter, sizeof(int), &zero, CL_TRUE);

    cl_kernel kernel = opencl_api.get_kernel("compute_orientations");

    // Launch for each octave
    for (int oct = 0; oct < num_octaves; oct++)
    {
        auto [width, height] = grad_pyramid.get_dimensions(oct);

        // Filter keypoints by octave (can be done in separate kernel or on
host)
        // For simplicity, launch for all keypoints and let kernel skip wrong
octaves

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &refined_buffer);
        clSetKernelArg(kernel, 1, sizeof(int), &num_refined);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &grad_pyramid_buffer);
        clSetKernelArg(kernel, 3, sizeof(int), &width);
        clSetKernelArg(kernel, 4, sizeof(int), &height);
        clSetKernelArg(kernel, 5, sizeof(float), &lambda_ori);
        clSetKernelArg(kernel, 6, sizeof(float), &lambda_desc);
        clSetKernelArg(kernel, 7, sizeof(float), &min_pix_dist);
        clSetKernelArg(kernel, 8, sizeof(cl_mem), &oriented_buffer);
        clSetKernelArg(kernel, 9, sizeof(cl_mem), &oriented_counter);

        size_t global_work_size = num_refined;
        opencl_api.enqueue_kernel(kernel, 1, &global_work_size, nullptr);
    }

    // Read oriented count (STILL NO D2H TRANSFER OF ACTUAL DATA!)
    int num_oriented = 0;
    opencl_api.read_buffer(oriented_counter, sizeof(int), &num_oriented,
CL_TRUE);

PERFORMANCE ANALYSIS:
    - Input: 5K refined keypoints
    - Patch size: ~30×30 pixels average = 900 pixels/keypoint
    - Histogram: 36 bins (local memory, no atomics)
    - Smoothing: 6 iterations × 36 bins = 216 FLOPs
    - Peak finding: 36 comparisons + 3 interpolations = ~50 FLOPs
    - Total FLOPs: 5K × (900×10 + 216 + 50) = ~45M FLOPs (~1.5ms @ 30 TFLOPS)
    - Memory reads: 5K × 900 pixels × 8 bytes = 36 MB (~0.24ms @ 150 GB/s)
    - Atomic writes: 7K × 36 bytes = 252 KB (~0.002ms)
    - Expected time: ~5-8ms (vs ~50ms CPU) = 6-10× speedup

ZERO READBACK VALIDATION:
    Input: refined_buffer (already in VRAM from refine_keypoints)
    Gradient data: grad_pyramid (already in VRAM)
    Output: oriented_buffer (stays in VRAM for descriptor kernel)
    Result: NO H2D or D2H transfers during orientation assignment!

RTX 4060 OCCUPANCY ANALYSIS:
    - Threads: 5K (LOW - GPU has 24,576 CUDA cores!)
    - Registers/thread: ~40 (histogram + temps)
    - Shared memory: 0 bytes (using local memory)
    - Occupancy: ~20% (low but acceptable, limited by workload size)
    - Strategy: Maximize ILP (instruction-level parallelism) since occupancy is
low
*/
