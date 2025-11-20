// void draw_point(Image& img, int x, int y, int size)
// {
//     for (int i = x-size/2; i <= x+size/2; i++) {
//         for (int j = y-size/2; j <= y+size/2; j++) {
//             if (i < 0 || i >= img.width) continue;
//             if (j < 0 || j >= img.height) continue;
//             if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
//             if (img.channels == 3) {
//                 img.set_pixel(i, j, 0, 1.f);
//                 img.set_pixel(i, j, 1, 0.f);
//                 img.set_pixel(i, j, 2, 0.f);
//             } else {
//                 img.set_pixel(i, j, 0, 1.f);
//             }
//         }
//     }
// }

/*
This OpenCL kernel draws a point (diamond-shaped marker) on an image.
The point is drawn using Manhattan distance (L1 norm) to create a diamond shape.

Kernel Parameters:
- img_data: Image data in global memory (can be 1 or 3 channels)
- width: Image width
- height: Image height
- channels: Number of channels (1 for grayscale, 3 for RGB)
- center_x: X coordinate of the point center
- center_y: Y coordinate of the point center
- point_size: Size of the point (radius in Manhattan distance)

Details:
- Each work-item processes one pixel within the bounding box of the point
- Uses Manhattan distance to create a diamond shape: |i-x| + |j-y| <= size/2
- For RGB images (channels==3): draws red point (R=1, G=0, B=0)
- For grayscale images (channels==1): draws white point (value=1)
- Automatically handles boundary checking to avoid out-of-bounds writes
- Optimized for GPU with parallel processing of independent pixels

Note: The global work size should be set to (point_size+1, point_size+1) for
efficiency
*/
__kernel void draw_point(__global float *img_data, int width, int height,
                         int channels, int center_x, int center_y,
                         int point_size) {
  // Get local offset within the point's bounding box
  int local_i = get_global_id(0); // Offset in x direction
  int local_j = get_global_id(1); // Offset in y direction

  // Calculate actual pixel coordinates
  int i = center_x - point_size / 2 + local_i;
  int j = center_y - point_size / 2 + local_j;

  // Bounds check: skip if outside image boundaries
  if (i < 0 || i >= width || j < 0 || j >= height)
    return;

  // Check if pixel is within the point's bounding box range
  if (local_i > point_size || local_j > point_size)
    return;

  // Manhattan distance check: create diamond shape
  int manhattan_dist = abs(i - center_x) + abs(j - center_y);
  if (manhattan_dist > point_size / 2)
    return;

  // Draw the point based on number of channels
  if (channels == 3) {
    // RGB image: draw red point
    set_pixel(img_data, width, height, i, j, 0, 1.0f); // Red channel
    set_pixel(img_data, width, height, i, j, 1, 0.0f); // Green channel
    set_pixel(img_data, width, height, i, j, 2, 0.0f); // Blue channel
  } else {
    // Grayscale image: draw white point
    set_pixel(img_data, width, height, i, j, 0, 1.0f);
  }
}
