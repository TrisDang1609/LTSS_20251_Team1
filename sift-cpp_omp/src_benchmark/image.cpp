#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <algorithm>

#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// --- Constructor/Destructor/Basic Methods ---
Image::Image(std::string file_path) {
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }
    size = width * height * channels;
    data = new float[size];
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4) channels = 3;
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c) : width{w}, height{h}, channels{c}, size{w*h*c}, data{new float[w*h*c]()} {}
Image::Image() : width{0}, height{0}, channels{0}, size{0}, data{nullptr} {}
Image::~Image() { delete[] this->data; }

Image::Image(const Image& other) : width{other.width}, height{other.height}, channels{other.channels}, size{other.size}, data{new float[other.size]} {
    for (int i = 0; i < size; i++) data[i] = other.data[i];
}

Image& Image::operator=(const Image& other) {
    if (this != &other) {
        delete[] data;
        width = other.width; height = other.height; channels = other.channels; size = other.size;
        data = new float[other.size];
        for (int i = 0; i < other.size; i++) data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other) : width{other.width}, height{other.height}, channels{other.channels}, size{other.size}, data{other.data} {
    other.data = nullptr; other.size = 0;
}

Image& Image::operator=(Image&& other) {
    delete[] data;
    data = other.data; width = other.width; height = other.height; channels = other.channels; size = other.size;
    other.data = nullptr; other.size = 0;
    return *this;
}

bool Image::save(std::string file_path) {
    unsigned char *out_data = new unsigned char[width*height*channels];
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = std::roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success) std::cerr << "Failed to save image: " << file_path << "\n";
    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val) {
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const {
    if (x < 0) x = 0;
    if (x >= width) x = width - 1;
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp() {
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        if (val > 1.0) val = 1.0;
        if (val < 0.0) val = 0.0;
        data[i] = val;
    }
}

// Helper local function
float map_coordinate(float new_max, float current_max, float coord) {
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

// --- RESIZE SERIAL ---
Image Image::resize_serial(int new_w, int new_h, Interpolation method) const {
    Image resized(new_w, new_h, this->channels);
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                float value;
                if (method == Interpolation::BILINEAR) value = bilinear_interpolate(*this, old_x, old_y, c);
                else value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}

// --- RESIZE OMP ---
Image Image::resize_omp(int new_w, int new_h, Interpolation method) const {
    Image resized(new_w, new_h, this->channels);
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                float value;
                if (method == Interpolation::BILINEAR) value = bilinear_interpolate(*this, old_x, old_y, c);
                else value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}

float bilinear_interpolate(const Image& img, float x, float y, int c) {
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c) {
    return img.get_pixel(std::round(x), std::round(y), c);
}

// --- RGB TO GRAYSCALE SERIAL ---
Image rgb_to_grayscale_serial(const Image& img) {
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red = img.get_pixel(x, y, 0);
            float green = img.get_pixel(x, y, 1);
            float blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
        }
    }
    return gray;
}

// --- RGB TO GRAYSCALE OMP ---
// Image rgb_to_grayscale_omp(const Image& img) {
//     assert(img.channels == 3);
//     Image gray(img.width, img.height, 1);
//     #pragma omp parallel for
//     for (int x = 0; x < img.width; x++) {
//         for (int y = 0; y < img.height; y++) {
//             float red = img.get_pixel(x, y, 0);
//             float green = img.get_pixel(x, y, 1);
//             float blue = img.get_pixel(x, y, 2);
//             gray.set_pixel(x, y, 0, 0.299f*red + 0.587f*green + 0.114f*blue);
//         }
//     }
//     return gray;
// }

Image rgb_to_grayscale_omp(const Image& img) {
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    
    int total_pixels = img.width * img.height;
    
    // Thay vì loop 2 chiều, ta flatten ra 1 chiều để vector hóa tốt hơn
    // Do cấu trúc Image lưu dữ liệu liên tiếp (row-major hoặc channel interleaved), 
    // ta cần cẩn thận việc truy xuất. 
    // Cách an toàn nhất với cấu trúc hiện tại vẫn là loop 2 chiều nhưng thêm simd.

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            // Lưu ý: Việc gọi hàm get_pixel bên trong loop có thể ngăn cản SIMD 
            // do nó chứa điều kiện if (kiểm tra biên).
            // Để tối ưu thực sự, nên truy cập trực tiếp vào mảng data:
            
            // float* data = img.data;
            // int idx = ...
            
            // Nhưng để an toàn và đơn giản, ta dùng code cũ và hy vọng compiler optimize:
            float red = img.get_pixel(x, y, 0);
            float green = img.get_pixel(x, y, 1);
            float blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299f*red + 0.587f*green + 0.114f*blue);
        }
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img) {
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

// --- GAUSSIAN BLUR SERIAL ---
Image gaussian_blur_serial(const Image& img, float sigma) {
    assert(img.channels == 1);
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        kernel.set_pixel(center+k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++) kernel.data[k] /= sum;

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);

    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum_val = 0;
            for (int k = 0; k < size; k++) {
                int dy = -center + k;
                sum_val += img.get_pixel(x, y+dy, 0) * kernel.data[k];
            }
            tmp.set_pixel(x, y, 0, sum_val);
        }
    }
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float sum_val = 0;
            for (int k = 0; k < size; k++) {
                int dx = -center + k;
                sum_val += tmp.get_pixel(x+dx, y, 0) * kernel.data[k];
            }
            filtered.set_pixel(x, y, 0, sum_val);
        }
    }
    return filtered;
}

// --- GAUSSIAN BLUR OMP ---
Image gaussian_blur_omp(const Image& img, float sigma) {
    assert(img.channels == 1);

    // 1. Chuẩn bị Kernel (Giữ nguyên)
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    Image kernel(size, 1, 1);
    float sum = 0;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        kernel.set_pixel(center+k, 0, 0, val);
        sum += val;
    }
    for (int k = 0; k < size; k++) kernel.data[k] /= sum;
    
    // Lấy con trỏ thô để truy cập nhanh (giảm overhead của object Image)
    const float* kernel_data = kernel.data;
    const float* img_data = img.data;
    int w = img.width;
    int h = img.height;

    Image tmp(w, h, 1);
    float* tmp_data = tmp.data;
    
    Image filtered(w, h, 1);
    float* filtered_data = filtered.data;

    // ==========================================
    // PASS 1: VERTICAL (Dọc)
    // ==========================================
    #pragma omp parallel for
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            float acc = 0.0f;
            
            // Kiểm tra xem pixel hiện tại có nằm trong vùng an toàn không
            // Vùng an toàn: kernel không bị trờm ra ngoài mép trên hoặc dưới
            if (y >= center && y < h - center) {
                // --- SAFE ZONE (FAST PATH) ---
                // Không cần if/else, truy cập mảng trực tiếp
                #pragma omp simd reduction(+:acc)
                for (int k = 0; k < size; k++) {
                    int dy = -center + k;
                    // Truy cập trực tiếp: (y + dy) * w + x
                    acc += img_data[(y + dy) * w + x] * kernel_data[k];
                }
            } else {
                // --- BORDER ZONE (SLOW PATH) ---
                // Cần kiểm tra biên (sử dụng logic kẹp giá trị y)
                for (int k = 0; k < size; k++) {
                    int dy = -center + k;
                    int cur_y = y + dy;
                    // Clamp (kẹp) giá trị y vào trong ảnh
                    if (cur_y < 0) cur_y = 0;
                    else if (cur_y >= h) cur_y = h - 1;
                    
                    acc += img_data[cur_y * w + x] * kernel_data[k];
                }
            }
            tmp_data[y * w + x] = acc;
        }
    }

    // ==========================================
    // PASS 2: HORIZONTAL (Ngang)
    // ==========================================
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        // Tính sẵn offset hàng để tránh nhân đi nhân lại trong vòng lặp con
        int row_offset = y * w; 
        
        for (int x = 0; x < w; x++) {
            float acc = 0.0f;

            // Kiểm tra vùng an toàn chiều ngang
            if (x >= center && x < w - center) {
                // --- SAFE ZONE (FAST PATH) ---
                // Đây là chỗ SIMD phát huy tác dụng mạnh nhất vì dữ liệu liền kề
                #pragma omp simd reduction(+:acc)
                for (int k = 0; k < size; k++) {
                    int dx = -center + k;
                    // Dữ liệu nằm ngang liền kề nhau trong bộ nhớ -> Cache friendly + SIMD tốt
                    acc += tmp_data[row_offset + (x + dx)] * kernel_data[k];
                }
            } else {
                // --- BORDER ZONE (SLOW PATH) ---
                for (int k = 0; k < size; k++) {
                    int dx = -center + k;
                    int cur_x = x + dx;
                    
                    if (cur_x < 0) cur_x = 0;
                    else if (cur_x >= w) cur_x = w - 1;

                    acc += tmp_data[row_offset + cur_x] * kernel_data[k];
                }
            }
            filtered_data[row_offset + x] = acc;
        }
    }

    return filtered;
}
void draw_point(Image& img, int x, int y, int size) {
    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2) {
    if (x2 < x1) { std::swap(x1, x2); std::swap(y1, y2); }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}