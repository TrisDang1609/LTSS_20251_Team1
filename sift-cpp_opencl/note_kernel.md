# How to use kernels

## IMAGE_RESIZE:
```cpp
// Example: Resize an image from 1920x1080 to 640x480
size_t global_work_size[3] = {640, 480, 3};  // width, height, channels
clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
```

// Bước 1: Load ảnh từ disk vào CPU (Image)
Image cpu_img("input.jpg");  
// ↑ Ảnh ở RAM, data = float array

// Bước 2: Upload từ CPU → GPU (chỉ 1 lần!)
GPUImage gpu_img = GPUImage::from_cpu(cpu_img);
// ↑ Copy dữ liệu từ cpu_img.data → gpu_img.gpu_buffer (VRAM)
// Thời gian: ~10ms cho 1920x1080

// Bước 3: Resize trên GPU (KHÔNG cần copy về CPU!)
GPUImage resized = gpu_img.resize(640, 480);
// ↑ Input: gpu_img.gpu_buffer (VRAM)
//   Output: resized.gpu_buffer (VRAM)
//   KHÔNG có data transfer CPU↔GPU!

// Bước 4: Tiếp tục xử lý trên GPU (vẫn KHÔNG cần CPU!)
GPUImage gray = resized.to_grayscale();  // GPU → GPU
GPUImage blurred = gray.gaussian_blur(1.6);  // GPU → GPU

// Bước 5: Download về CPU khi cần (chỉ 1 lần!)
Image result = blurred.to_cpu();
// ↑ Copy dữ liệu từ blurred.gpu_buffer → result.data
result.save("output.jpg");

CMAKE LIST EXAMPLE
add_executable(find_keypoints find_keypoints.cpp)

target_include_directories(find_keypoints PRIVATE ../src)

target_link_libraries(find_keypoints PRIVATE
                      img
                      sift
)

add_executable(match_features match_features.cpp)

target_include_directories(match_features PRIVATE ../src)

target_link_libraries(match_features PRIVATE
                      img
                      sift
)

SET_TARGET_PROPERTIES(find_keypoints PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET_TARGET_PROPERTIES(match_features PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)