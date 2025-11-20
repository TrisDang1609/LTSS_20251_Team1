// // separable 2D gaussian blur for 1 channel image
// Image gaussian_blur(const Image& img, float sigma)
// {
//     assert(img.channels == 1);

//     int size = std::ceil(6 * sigma);
//     if (size % 2 == 0)
//         size++;
//     int center = size / 2;
//     Image kernel(size, 1, 1);
//     float sum = 0;
//     for (int k = -size/2; k <= size/2; k++) {
//         float val = std::exp(-(k*k) / (2*sigma*sigma));
//         kernel.set_pixel(center+k, 0, 0, val);
//         sum += val;
//     }
//     for (int k = 0; k < size; k++)
//         kernel.data[k] /= sum;

//     Image tmp(img.width, img.height, 1);
//     Image filtered(img.width, img.height, 1);

//     // convolve vertical
//     for (int x = 0; x < img.width; x++) {
//         for (int y = 0; y < img.height; y++) {
//             float sum = 0;
//             for (int k = 0; k < size; k++) {
//                 int dy = -center + k;
//                 sum += img.get_pixel(x, y+dy, 0) * kernel.data[k];
//             }
//             tmp.set_pixel(x, y, 0, sum);
//         }
//     }
//     // convolve horizontal
//     for (int x = 0; x < img.width; x++) {
//         for (int y = 0; y < img.height; y++) {
//             float sum = 0;
//             for (int k = 0; k < size; k++) {
//                 int dx = -center + k;
//                 sum += tmp.get_pixel(x+dx, y, 0) * kernel.data[k];
//             }
//             filtered.set_pixel(x, y, 0, sum);
//         }
//     }
//     return filtered;
// }

/*
This OpenCL kernel performs separable 2D Gaussian blur on a single-channel
grayscale image. The blur is implemented as two separate 1D convolutions
(vertical then horizontal) for optimal performance.

This is the FIRST PASS: Vertical convolution kernel.
Each work-item convolves one pixel in the vertical direction.

Kernel Parameters:
- src_data: Source image data in global memory (single channel, grayscale)
- dst_data: Destination image data (temporary buffer for intermediate result)
- kernel_data: 1D Gaussian kernel weights (normalized)
- width: Image width
- height: Image height
- kernel_size: Size of the Gaussian kernel (must be odd)
- center: Center index of the kernel (kernel_size / 2)

Details:
- Each work-item processes one output pixel by convolving vertically
- Uses boundary clamping via get_pixel() for edge handling
- Optimized for coalesced memory access patterns on GPU
*/
__kernel void gaussian_blur_vertical(__global const float *src_data,
                                     __global float *dst_data,
                                     __global const float *kernel_data,
                                     int width, int height, int kernel_size,
                                     int center) {
  // Get global work-item IDs (one per output pixel)
  int x = get_global_id(0); // Output x coordinate
  int y = get_global_id(1); // Output y coordinate

  // Bounds check: skip work-items outside image dimensions
  if (x >= width || y >= height)
    return;

  // Perform vertical convolution
  float sum = 0.0f;
  for (int k = 0; k < kernel_size; k++) {
    int dy = -center + k; // Offset from center
    // get_pixel handles boundary clamping automatically
    float pixel_val = get_pixel(src_data, width, height, x, y + dy, 0);
    sum += pixel_val * kernel_data[k];
  }

  // Write result to temporary buffer (no bounds check needed - already
  // validated)
  set_pixel(dst_data, width, height, x, y, 0, sum);
}

/*
This OpenCL kernel performs separable 2D Gaussian blur on a single-channel
grayscale image. The blur is implemented as two separate 1D convolutions
(vertical then horizontal) for optimal performance.

This is the SECOND PASS: Horizontal convolution kernel.
Each work-item convolves one pixel in the horizontal direction.

Kernel Parameters:
- src_data: Source image data (temporary buffer from vertical pass)
- dst_data: Destination image data (final blurred result)
- kernel_data: 1D Gaussian kernel weights (normalized, same as vertical pass)
- width: Image width
- height: Image height
- kernel_size: Size of the Gaussian kernel (must be odd)
- center: Center index of the kernel (kernel_size / 2)

Details:
- Each work-item processes one output pixel by convolving horizontally
- Uses boundary clamping via get_pixel() for edge handling
- Separable convolution reduces complexity from O(n²) to O(2n) per pixel
- Optimized for GPU parallel execution with independent work-items
*/
__kernel void gaussian_blur_horizontal(__global const float *src_data,
                                       __global float *dst_data,
                                       __global const float *kernel_data,
                                       int width, int height, int kernel_size,
                                       int center) {
  // Get global work-item IDs (one per output pixel)
  int x = get_global_id(0); // Output x coordinate
  int y = get_global_id(1); // Output y coordinate

  // Bounds check: skip work-items outside image dimensions
  if (x >= width || y >= height)
    return;

  // Perform horizontal convolution
  float sum = 0.0f;
  for (int k = 0; k < kernel_size; k++) {
    int dx = -center + k; // Offset from center
    // get_pixel handles boundary clamping automatically
    float pixel_val = get_pixel(src_data, width, height, x + dx, y, 0);
    sum += pixel_val * kernel_data[k];
  }

  // Write final blurred result (no bounds check needed - already validated)
  set_pixel(dst_data, width, height, x, y, 0, sum);
}

// Flow to call GAUSSIAN BLUR: vertical pass -> horizontal pass
