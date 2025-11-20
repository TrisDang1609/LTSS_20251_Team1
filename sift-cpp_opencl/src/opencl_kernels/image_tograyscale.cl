// Image rgb_to_grayscale(const Image& img)
// {
//     assert(img.channels == 3);
//     Image gray(img.width, img.height, 1);
//     for (int x = 0; x < img.width; x++) {
//         for (int y = 0; y < img.height; y++) {
//             float red, green, blue;
//             red = img.get_pixel(x, y, 0);
//             green = img.get_pixel(x, y, 1);
//             blue = img.get_pixel(x, y, 2);
//             gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
//         }
//     }
//     return gray;
// }

/*
This OpenCL kernel converts an RGB image to a grayscale image using the
luminosity method.

Kernel Parameters:
- src_data: A pointer to the source image data in global memory. The data is
stored in a flattened array with channel-major order (channel-height-width).
- dst_data: A pointer to the destination image data in global memory. The output
is a single-channel grayscale image.
- width: The width of the input image.
- height: The height of the input image.

Details:
- Each work-item processes one pixel of the output image.
- The kernel retrieves the RGB values of the corresponding pixel from the source
image using the helper function `get_pixel`.
- The grayscale value is computed using the luminosity method: 0.299*R + 0.587*G
+ 0.114*B.
- The computed grayscale value is stored in the destination image buffer.
*/
__kernel void rgb_to_grayscale(__global const float *src_data,
                               __global float *dst_data, int width,
                               int height) {
  // Get global work-item IDs (one per output pixel)
  int x = get_global_id(0); // Output x coordinate
  int y = get_global_id(1); // Output y coordinate

  if (x >= width || y >= height)
    return; // Out of bounds check

  // Get RGB values from source image
  float red = get_pixel(src_data, width, height, x, y, 0);
  float green = get_pixel(src_data, width, height, x, y, 1);
  float blue = get_pixel(src_data, width, height, x, y, 2);

  // Compute grayscale value using luminosity method
  float gray = 0.299f * red + 0.587f * green + 0.114f * blue;

  // Set grayscale value in destination image
  dst_data[y * width + x] = gray;
}