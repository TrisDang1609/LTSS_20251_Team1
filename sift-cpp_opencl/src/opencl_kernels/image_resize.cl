/*
OpenCL kernel for image resizing with bilinear or nearest-neighbor
interpolation. Optimized for GPU execution (RTX 4060).

Reference CPU implementation from image.cpp:
Image Image::resize(int new_w, int new_h, Interpolation method) const
{
    Image resized(new_w, new_h, this->channels);
    float value = 0;
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                if (method == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (method == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}
*/

/*
Helper function: Map coordinate from 0-current_max range to 0-new_max range.
Equivalent to map_coordinate() in image.cpp.
*/
inline float map_coordinate(float new_max, float current_max, float coord) {
  float a = new_max / current_max;
  float b = -0.5f + a * 0.5f;
  return a * coord + b;
}

/*
Helper function: Bilinear interpolation for smooth image resizing.
Equivalent to bilinear_interpolate() in image.cpp.
Parameters:
- src_data: Source image data
- width: Source image width
- height: Source image height
- x, y: Floating-point coordinates in source image
- c: Channel index
Returns: Interpolated pixel value
*/
inline float bilinear_interpolate(__global const float *src_data, int width,
                                  int height, float x, float y, int c) {
  float x_floor = floor(x);
  float y_floor = floor(y);
  float x_ceil = x_floor + 1.0f;
  float y_ceil = y_floor + 1.0f;

  // Get four neighboring pixels
  float p1 = get_pixel(src_data, width, height, (int)x_floor, (int)y_floor, c);
  float p2 = get_pixel(src_data, width, height, (int)x_ceil, (int)y_floor, c);
  float p3 = get_pixel(src_data, width, height, (int)x_floor, (int)y_ceil, c);
  float p4 = get_pixel(src_data, width, height, (int)x_ceil, (int)y_ceil, c);

  // Interpolate vertically
  float q1 = (y_ceil - y) * p1 + (y - y_floor) * p3;
  float q2 = (y_ceil - y) * p2 + (y - y_floor) * p4;

  // Interpolate horizontally
  return (x_ceil - x) * q1 + (x - x_floor) * q2;
}

/*
Helper function: Nearest-neighbor interpolation for fast image resizing.
Equivalent to nn_interpolate() in image.cpp.
Parameters:
- src_data: Source image data
- width: Source image width
- height: Source image height
- x, y: Floating-point coordinates in source image
- c: Channel index
Returns: Pixel value from nearest neighbor
*/
inline float nn_interpolate(__global const float *src_data, int width,
                            int height, float x, float y, int c) {
  return get_pixel(src_data, width, height, (int)round(x), (int)round(y), c);
}

/*
Main OpenCL kernel for image resizing.
This kernel is optimized for GPU execution with each work-item processing one
output pixel.

Parameters:
- src_data: Source image data (channel-height-width order)
- dst_data: Destination image data (channel-height-width order)
- src_width: Source image width
- src_height: Source image height
- dst_width: Destination image width
- dst_height: Destination image height
- channels: Number of channels (1 for grayscale, 3 for RGB)
- use_bilinear: 1 for bilinear interpolation, 0 for nearest-neighbor

Work-item organization:
- Global work size: (dst_width, dst_height, channels)
- Each work-item computes one pixel in the output image
*/
__kernel void resize_image_kernel(__global const float *src_data,
                                  __global float *dst_data, int src_width,
                                  int src_height, int dst_width, int dst_height,
                                  int channels, int use_bilinear) {
  // Get global work-item IDs (one per output pixel)
  int x = get_global_id(0); // Output x coordinate
  int y = get_global_id(1); // Output y coordinate
  int c = get_global_id(2); // Channel index

  // Bounds check: ensure we don't process out-of-range pixels
  if (x >= dst_width || y >= dst_height || c >= channels)
    return;

  // Map output coordinates to source image coordinates
  float old_x = map_coordinate((float)src_width, (float)dst_width, (float)x);
  float old_y = map_coordinate((float)src_height, (float)dst_height, (float)y);

  // Perform interpolation based on method
  float value;
  if (use_bilinear)
    value =
        bilinear_interpolate(src_data, src_width, src_height, old_x, old_y, c);
  else
    value = nn_interpolate(src_data, src_width, src_height, old_x, old_y, c);

  // Write result to destination buffer
  // Data layout: dst_data[c*dst_height*dst_width + y*dst_width + x]
  int dst_idx = c * dst_height * dst_width + y * dst_width + x;
  dst_data[dst_idx] = value;
}
