// Image grayscale_to_rgb(const Image& img)
// {
//     assert(img.channels == 1);
//     Image rgb(img.width, img.height, 3);
//     for (int x = 0; x < img.width; x++) {
//         for (int y = 0; y < img.height; y++) {
//             float gray_val = img.get_pixel(x, y, 0);
//             rgb.set_pixel(x, y, 0, gray_val);
//             rgb.set_pixel(x, y, 1, gray_val);
//             rgb.set_pixel(x, y, 2, gray_val);
//         }
//     }
//     return rgb;
// }

/*
This OpenCL kernel converts a single-channel grayscale image to a 3-channel RGB
image. Each RGB channel is filled with the same grayscale value, producing a
grayscale-looking RGB image.

Kernel Parameters:
- src_data: A pointer to the source grayscale image data in global memory
(single channel).
- dst_data: A pointer to the destination RGB image data in global memory (3
channels).
- width: The width of the image.
- height: The height of the image.

Details:
- Each work-item processes one pixel of the output image.
- The kernel retrieves the grayscale value from the source image using
`get_pixel`.
- The same grayscale value is written to all three RGB channels (R, G, B) using
`set_pixel`.
- This implementation is optimized for GPU execution with parallel processing of
independent pixels.
*/
__kernel void grayscale_to_rgb(__global const float *src_data,
                               __global float *dst_data, int width,
                               int height) {
  // Get global work-item IDs (one per output pixel)
  int x = get_global_id(0); // Output x coordinate
  int y = get_global_id(1); // Output y coordinate

  // Bounds check: skip work-items outside image dimensions
  if (x >= width || y >= height)
    return;

  // Get grayscale value from source image (channel 0)
  float gray_val = get_pixel(src_data, width, height, x, y, 0);

  // Set the same grayscale value to all three RGB channels
  // Optimized: Direct memory writes without redundant bounds checking
  set_pixel(dst_data, width, height, x, y, 0, gray_val); // Red channel
  set_pixel(dst_data, width, height, x, y, 1, gray_val); // Green channel
  set_pixel(dst_data, width, height, x, y, 2, gray_val); // Blue channel
}
