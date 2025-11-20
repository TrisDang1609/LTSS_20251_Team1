/*
================================================================================
GPU-ACCELERATED SIFT DESCRIPTOR COMPUTATION
================================================================================
Matches CPU implementation in sift.cpp lines 1705-1760 exactly.

Note: GPUKeypoint and OrientedKeypoint structs are defined in common_kernel.cl
*/

#define N_HIST 4
#define N_ORI 8
#define DESCRIPTOR_SIZE 128
#define TWO_PI 6.28318530718f

/*
Helper: Linear interpolation to distribute gradient contribution
into 4×4×8 histogram (matches update_histograms in sift.cpp lines 1705-1729).
*/
void update_histogram_linear(float *hist, float x, float y, float contrib,
                             float theta_mn, float lambda_desc) {
  float x_i, y_j;
  for (int i = 1; i <= N_HIST; i++) {
    x_i = (i - (1.0f + (float)N_HIST) / 2.0f) * 2.0f * lambda_desc / N_HIST;

    if (fabs(x_i - x) > 2.0f * lambda_desc / N_HIST)
      continue;

    for (int j = 1; j <= N_HIST; j++) {
      y_j = (j - (1.0f + (float)N_HIST) / 2.0f) * 2.0f * lambda_desc / N_HIST;

      if (fabs(y_j - y) > 2.0f * lambda_desc / N_HIST)
        continue;

      float hist_weight = (1.0f - N_HIST * 0.5f / lambda_desc * fabs(x_i - x)) *
                          (1.0f - N_HIST * 0.5f / lambda_desc * fabs(y_j - y));

      for (int k = 1; k <= N_ORI; k++) {
        float theta_k = TWO_PI * (k - 1) / N_ORI;
        float theta_diff = fmod(theta_k - theta_mn + TWO_PI, TWO_PI);

        // Fix circular wrap-around distance
        if (theta_diff >= M_PI_F)
          theta_diff = TWO_PI - theta_diff;

        if (theta_diff >= TWO_PI / N_ORI)
          continue;

        float bin_weight = 1.0f - N_ORI * 0.5f / M_PI_F * theta_diff;

        int hist_idx = (i - 1) * N_HIST * N_ORI + (j - 1) * N_ORI + (k - 1);
        hist[hist_idx] += hist_weight * bin_weight * contrib;
      }
    }
  }
}

/*
Main kernel: Compute 128-dimensional SIFT descriptors for oriented keypoints.
Matches CPU implementation compute_keypoint_descriptor() at lines 1760-1804.
*/
__kernel void compute_descriptors(
    __global const OrientedKeypoint *oriented_keypoints,
    __global int *oriented_counter, __global const float *grad_pyramid_buffer,
    int width, int height, int octave_idx, int scale_idx, float lambda_desc,
    float min_pix_dist, __global uchar *descriptors) {
  int kp_idx = get_global_id(0);
  if (kp_idx >= *oriented_counter)
    return;

  OrientedKeypoint okp = oriented_keypoints[kp_idx];
  GPUKeypoint kp = okp.kp;

  // Filter by octave and scale
  if (kp.octave != octave_idx || kp.scale != scale_idx)
    return;
  float theta = okp.orientation;

  float pix_dist = min_pix_dist * pown(2.0f, kp.octave);
  float patch_sigma = lambda_desc * kp.sigma;

  // Compute adaptive patch bounds (CPU lines 1769-1773)
  float half_size =
      M_SQRT2_F * lambda_desc * kp.sigma * (N_HIST + 1.0f) / N_HIST;
  int x_start = (int)round((kp.x - half_size) / pix_dist);
  int x_end = (int)round((kp.x + half_size) / pix_dist);
  int y_start = (int)round((kp.y - half_size) / pix_dist);
  int y_end = (int)round((kp.y + half_size) / pix_dist);

  // Build 4×4×8 histogram
  float hist[DESCRIPTOR_SIZE];
  for (int i = 0; i < DESCRIPTOR_SIZE; i++)
    hist[i] = 0.0f;

  float cos_t = cos(theta);
  float sin_t = sin(theta);

  // Sample adaptive patch (CPU lines 1778-1794)
  for (int m = x_start; m <= x_end; m++) {
    for (int n = y_start; n <= y_end; n++) {
      if (m < 0 || m >= width || n < 0 || n >= height)
        continue;

      // Normalized coordinates (CPU lines 1783-1784)
      float x_norm =
          ((m * pix_dist - kp.x) * cos_t + (n * pix_dist - kp.y) * sin_t) /
          kp.sigma;
      float y_norm =
          (-(m * pix_dist - kp.x) * sin_t + (n * pix_dist - kp.y) * cos_t) /
          kp.sigma;

      // Check patch bounds (CPU line 1787)
      if (fmax(fabs(x_norm), fabs(y_norm)) >
          lambda_desc * (N_HIST + 1.0f) / N_HIST)
        continue;

      // Sample gradients (CPU line 1789)
      float gx = grad_pyramid_buffer[2 * (n * width + m) + 0];
      float gy = grad_pyramid_buffer[2 * (n * width + m) + 1];

      // Compute orientation (CPU line 1790 - match fmod behavior exactly)
      float theta_mn = fmod(atan2(gy, gx) - theta + 4.0f * M_PI_F, TWO_PI);

      float grad_norm = sqrt(gx * gx + gy * gy);

      // Gaussian weighting (CPU line 1791)
      float dx = m * pix_dist - kp.x;
      float dy = n * pix_dist - kp.y;
      float weight =
          exp(-(dx * dx + dy * dy) / (2.0f * patch_sigma * patch_sigma));
      float contribution = weight * grad_norm;

      // Update histogram (CPU line 1794)
      update_histogram_linear(hist, x_norm, y_norm, contribution, theta_mn,
                              lambda_desc);
    }
  }

  // Normalize to unit length (CPU lines 1740-1746)
  float norm = 0.0f;
  for (int i = 0; i < DESCRIPTOR_SIZE; i++)
    norm += hist[i] * hist[i];
  norm = sqrt(norm);

  if (norm < 1e-6f) {
    for (int i = 0; i < DESCRIPTOR_SIZE; i++)
      descriptors[kp_idx * DESCRIPTOR_SIZE + i] = 0;
    return;
  }

  // Clamp to 0.2 × norm (CPU lines 1748-1751)
  float norm2 = 0.0f;
  for (int i = 0; i < DESCRIPTOR_SIZE; i++) {
    hist[i] = fmin(hist[i], 0.2f * norm);
    norm2 += hist[i] * hist[i];
  }
  norm2 = sqrt(norm2);

  // Quantize to uint8 (CPU lines 1753-1757)
  for (int i = 0; i < DESCRIPTOR_SIZE; i++) {
    float val = 512.0f * hist[i] / norm2;
    // Match CPU floor behavior
    descriptors[kp_idx * DESCRIPTOR_SIZE + i] = (uchar)min((int)val, 255);
  }
}
