/*
================================================================================
GPU-ACCELERATED SIFT DIFFERENCE OF GAUSSIANS (DoG) PYRAMID GENERATION
================================================================================
Optimized for NVIDIA RTX 4060 (Ada Lovelace Architecture, 8GB VRAM)

ARCHITECTURAL OBJECTIVE: Zero Round-Trips
- Input: GPUScaleSpacePyramid (Gaussian pyramid already in VRAM)
- Operation: Subtract adjacent scales (GPU-to-GPU)
- Output: DoG pyramid (all buffers in VRAM)

CPU REFERENCE LOGIC (sift.cpp):
    ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid &img_pyramid)
    {
        ScaleSpacePyramid dog_pyramid = {
            img_pyramid.num_octaves,
            img_pyramid.imgs_per_octave - 1,
            std::vector<std::vector<Image>>(img_pyramid.num_octaves)};

        for (int i = 0; i < dog_pyramid.num_octaves; i++)
        {
            dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
            for (int j = 1; j < img_pyramid.imgs_per_octave; j++)
            {
                // DoG[j-1] = Gaussian[j] - Gaussian[j-1]
                const Image &img1 = img_pyramid.octaves[i][j];
                const Image &img2 = img_pyramid.octaves[i][j - 1];
                dog_pyramid.octaves[i].push_back(img1 - img2);
            }
        }
        return dog_pyramid;
    }

KERNEL STRATEGY:
    Parallel pixel-wise subtraction:
    - Each work-item computes one DoG pixel: dog[x,y] = gauss_high[x,y] -
gauss_low[x,y]
    - Highly parallelizable (millions of independent operations)
    - Memory bandwidth bound (2 reads + 1 write per pixel)

PERFORMANCE TARGETS (RTX 4060):
    - DoG computation: ~1-2ms per octave (all scales in parallel)
    - Total for 4 octaves: ~4-8ms (vs ~50-100ms on CPU)
    - 10-20× speedup from GPU parallelism

Ada Lovelace Optimizations:
    - Coalesced memory access (sequential pixel reads)
    - No shared memory needed (simple element-wise operation)
    - Efficient memory bandwidth utilization (~90% peak)
================================================================================
*/

/*
DoG Subtraction Kernel: Compute Difference of Gaussians between two scale
levels.

Parameters:
- src_high: Higher sigma Gaussian image (scale j)
- src_low: Lower sigma Gaussian image (scale j-1)
- dst_dog: Output DoG image (dst = src_high - src_low)
- width: Image width for this octave
- height: Image height for this octave

Work-item organization:
- Global work size: (width, height)
- Each work-item computes one output pixel

Mathematical operation:
    DoG(x, y, σ_high, σ_low) = G(x, y, σ_high) - G(x, y, σ_low)

This captures blob-like structures at specific scales, forming the basis
for SIFT keypoint detection (local extrema in DoG scale-space).
*/
__kernel void compute_dog_subtraction(__global const float *src_high,
                                      __global const float *src_low,
                                      __global float *dst_dog, int width,
                                      int height) {
  // Get global work-item IDs (one per output pixel)
  int x = get_global_id(0);
  int y = get_global_id(1);

  // Bounds check: ensure we don't process out-of-range pixels
  if (x >= width || y >= height)
    return;

  // Compute linear index (CHW layout: single channel)
  int idx = y * width + x;

  // Perform DoG subtraction
  // DoG = Gaussian(high_sigma) - Gaussian(low_sigma)
  float high_val = src_high[idx];
  float low_val = src_low[idx];
  dst_dog[idx] = high_val - low_val;
}

/*
USAGE PATTERN (Host Code):
    for (int oct = 0; oct < num_octaves; oct++)
    {
        for (int scale = 0; scale < imgs_per_octave - 1; scale++)
        {
            cl_kernel kernel = opencl_api.get_kernel("compute_dog_subtraction");

            clSetKernelArg(kernel, 0, sizeof(cl_mem),
&gaussian_pyramid.octaves[oct][scale + 1]); clSetKernelArg(kernel, 1,
sizeof(cl_mem), &gaussian_pyramid.octaves[oct][scale]); clSetKernelArg(kernel,
2, sizeof(cl_mem), &dog_pyramid.octaves[oct][scale]); clSetKernelArg(kernel, 3,
sizeof(int), &width); clSetKernelArg(kernel, 4, sizeof(int), &height);

            size_t global_work_size[2] = {width, height};
            opencl_api.enqueue_kernel(kernel, 2, global_work_size, nullptr);
        }
    }

MEMORY LAYOUT (Example: 1920×1080, 4 octaves, 5 DoG scales/octave):
    Octave 0: 3840×2160 × 5 images = ~158 MB
    Octave 1: 1920×1080 × 5 images = ~40 MB
    Octave 2: 960×540 × 5 images   = ~10 MB
    Octave 3: 480×270 × 5 images   = ~2.5 MB
    Total: ~210 MB (2.6% of 8GB VRAM)

Combined with Gaussian pyramid: ~253 MB + ~210 MB = ~463 MB (5.8% VRAM)
*/
