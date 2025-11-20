// void draw_line(Image& img, int x1, int y1, int x2, int y2)
// {
//     if (x2 < x1) {
//         std::swap(x1, x2);
//         std::swap(y1, y2);
//     }
//     int dx = x2 - x1, dy = y2 - y1;
//     for (int x = x1; x < x2; x++) {
//         int y = y1 + dy*(x-x1)/dx;
//         if (img.channels == 3) {
//             img.set_pixel(x, y, 0, 0.f);
//             img.set_pixel(x, y, 1, 1.f);
//             img.set_pixel(x, y, 2, 0.f);
//         } else {
//             img.set_pixel(x, y, 0, 1.f);
//         }
//     }
// }

/*
This OpenCL kernel draws a line segment on an image using a parallelized
approach. The line is drawn by dividing it into segments, with each work-item
drawing one pixel.

Kernel Parameters:
- img_data: Image data in global memory (can be 1 or 3 channels)
- width: Image width
- height: Image height
- channels: Number of channels (1 for grayscale, 3 for RGB)
- x1: Starting point X coordinate
- y1: Starting point Y coordinate
- x2: Ending point X coordinate
- y2: Ending point Y coordinate

Details:
- Each work-item processes one pixel along the line
- Uses linear interpolation to calculate y coordinate for each x
- For RGB images (channels==3): draws green line (R=0, G=1, B=0)
- For grayscale images (channels==1): draws white line (value=1)
- Automatically handles boundary checking to avoid out-of-bounds writes
- Work-items are distributed across the x-range of the line

Note: The algorithm assumes x2 >= x1 (coordinates should be pre-swapped if
needed) Global work size should be set to (x2-x1+1) for optimal coverage
*/
__kernel void draw_line(__global float *img_data, int width, int height,
                        int channels, int x1_in, int y1_in, int x2_in,
                        int y2_in) {
  // Ensure x1 <= x2 by swapping if necessary
  int x1 = x1_in, y1 = y1_in, x2 = x2_in, y2 = y2_in;
  if (x2 < x1) {
    swap_int(&x1, &x2);
    swap_int(&y1, &y2);
  }

  // Get global work-item ID (represents position along the line)
  int idx = get_global_id(0);

  // Calculate the current x coordinate
  int x = x1 + idx;

  // Bounds check: skip if beyond the line's x-range
  if (x > x2)
    return;

  // Calculate dx and dy for interpolation
  int dx = x2 - x1;
  int dy = y2 - y1;

  // Calculate y coordinate using linear interpolation
  // Handle special case when dx == 0 (vertical line)
  int y;
  if (dx == 0) {
    // Vertical line: each work-item draws a pixel if within y range
    // This is a simplified case; for true vertical lines, a different strategy
    // is better
    y = y1;
  } else {
    // General case: y = y1 + dy * (x - x1) / dx
    y = y1 + (dy * (x - x1)) / dx;
  }

  // Bounds check: skip if pixel is outside image
  if (x < 0 || x >= width || y < 0 || y >= height)
    return;

  // Draw the pixel based on number of channels
  if (channels == 3) {
    // RGB image: draw green line
    set_pixel(img_data, width, height, x, y, 0, 0.0f); // Red channel
    set_pixel(img_data, width, height, x, y, 1, 1.0f); // Green channel
    set_pixel(img_data, width, height, x, y, 2, 0.0f); // Blue channel
  } else {
    // Grayscale image: draw white line
    set_pixel(img_data, width, height, x, y, 0, 1.0f);
  }
}
