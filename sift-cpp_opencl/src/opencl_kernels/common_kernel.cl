#ifndef COMMON_KERNEL_CL
#define COMMON_KERNEL_CL

// ============================================================================
// SIFT Data Structures (Shared across all kernels)
// Must match definitions in libs/sift.hpp
// ============================================================================

typedef struct {
  int i, j;           // Discrete coordinates in DoG image
  int octave, scale;  // Pyramid location
  float x, y;         // Continuous (interpolated) coordinates in input image
  float sigma;        // Scale (sigma value)
  float extremum_val; // Interpolated DoG extremum value
} GPUKeypoint;

typedef struct {
  GPUKeypoint kp;    // Base keypoint information
  float orientation; // Dominant orientation angle [0, 2π]
} OrientedKeypoint;

// ============================================================================
// Helper Functions
// ============================================================================

/*
Helper function: Get pixel value from source image with boundary clamping.
Equivalent to Image::get_pixel() in image.cpp.
Parameters:
- src_data: Source image data (flattened array in channel-height-width order)
- width: Source image width
- height: Source image height
- x, y: Pixel coordinates
- c: Channel index
Returns: Pixel value at (x, y, c) with clamped boundary handling
*/
inline float get_pixel(__global const float *src_data, int width, int height,
                       int x, int y, int c) {
  // Clamp coordinates to image boundaries
  if (x < 0)
    x = 0;
  if (x >= width)
    x = width - 1;
  if (y < 0)
    y = 0;
  if (y >= height)
    y = height - 1;

  // Data is stored as: data[c*width*height + y*width + x]
  return src_data[c * width * height + y * width + x];
}

/*
Helper function: Set pixel value in destination image.
Equivalent to Image::set_pixel() in image.cpp (without bounds checking for
performance). Parameters:
- dst_data: Destination image data (flattened array in channel-height-width
order)
- width: Destination image width
- height: Destination image height
- x, y: Pixel coordinates
- c: Channel index
- val: Value to set
Note: Assumes coordinates are already validated by the caller (kernel bounds
check).
*/
inline void set_pixel(__global float *dst_data, int width, int height, int x,
                      int y, int c, float val) {
  // Data is stored as: data[c*width*height + y*width + x]
  dst_data[c * width * height + y * width + x] = val;
}

/*
Helper function: Set pixel value with bounds checking.
This version includes bounds validation for safe pixel setting.
Parameters:
- dst_data: Destination image data (flattened array in channel-height-width
order)
- width: Image width
- height: Image height
- channels: Number of channels in the image
- x, y: Pixel coordinates
- c: Channel index
- val: Value to set
Returns: 1 if pixel was set, 0 if out of bounds
*/
inline int set_pixel_safe(__global float *dst_data, int width, int height,
                          int channels, int x, int y, int c, float val) {
  // Bounds check
  if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels)
    return 0;

  // Data is stored as: data[c*width*height + y*width + x]
  dst_data[c * width * height + y * width + x] = val;
  return 1;
}

/*
Helper function: Swap two integer values.
Parameters:
- a: Pointer to first integer
- b: Pointer to second integer
*/
inline void swap_int(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

#endif // COMMON_KERNEL_CL
