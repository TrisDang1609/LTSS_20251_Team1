#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "image.hpp"
#include "sift.hpp"

using namespace std;
using namespace std::chrono;

int main()
{
    const int N = 5; // Số lần chạy để lấy trung bình
    cout << "Starting Benchmark (Runs = " << N << ")...\n";
    #ifdef _OPENMP
    cout << "OpenMP detected. Max threads: " << omp_get_max_threads() << "\n";
    #endif
    cout << fixed << setprecision(2);

    double total_serial_time = 0;
    double total_omp_time = 0;

    for (int i = 0; i < N; i++)
    {
        cout << "\nRun " << (i + 1) << "/" << N << "...\n";
        
        // Load ảnh (Không tính vào thời gian thuật toán)
        Image img("./../imgs/book_rotated.jpg");
        Image img2("./../imgs/book_in_scene.jpg");
        Image img_copy = img;   // Copy để dùng cho OMP
        Image img2_copy = img2; // Copy để dùng cho OMP

        // ==================== SERIAL ====================
        auto t1 = high_resolution_clock::now();
        
        std::vector<sift::Keypoint> kps1_s = sift::find_keypoints_and_descriptors_serial(img);
        std::vector<sift::Keypoint> kps2_s = sift::find_keypoints_and_descriptors_serial(img2);
        std::vector<std::pair<int, int>> matches_s = sift::find_keypoint_matches_serial(kps1_s, kps2_s);
        Image res_s = sift::draw_matches_serial(img, img2, kps1_s, kps2_s, matches_s);
        
        auto t2 = high_resolution_clock::now();
        double ms_serial = duration_cast<milliseconds>(t2 - t1).count();
        cout << "  [Serial] Time: " << ms_serial << " ms, Matches: " << matches_s.size() << "\n";
        total_serial_time += ms_serial;

        // ==================== OPENMP ====================
        auto t3 = high_resolution_clock::now();
        
        std::vector<sift::Keypoint> kps1_o = sift::find_keypoints_and_descriptors_omp(img_copy);
        std::vector<sift::Keypoint> kps2_o = sift::find_keypoints_and_descriptors_omp(img2_copy);
        std::vector<std::pair<int, int>> matches_o = sift::find_keypoint_matches_omp(kps1_o, kps2_o);
        Image res_o = sift::draw_matches_omp(img_copy, img2_copy, kps1_o, kps2_o, matches_o);
        
        auto t4 = high_resolution_clock::now();
        double ms_omp = duration_cast<milliseconds>(t4 - t3).count();
        cout << "  [OpenMP] Time: " << ms_omp << " ms, Matches: " << matches_o.size() << "\n";
        total_omp_time += ms_omp;

        // Chỉ lưu ảnh kết quả của lần chạy cuối
        if (i == N - 1) {
            res_o.save("book_matches_omp.jpg");
        }
    }

    double avg_serial = total_serial_time / N;
    double avg_omp = total_omp_time / N;
    double speedup = avg_serial / avg_omp;

    cout << "\n============= RESULTS =============\n";
    cout << "Avg Serial Time: " << avg_serial << " ms\n";
    cout << "Avg OpenMP Time: " << avg_omp << " ms\n";
    cout << "Speedup: " << speedup << "x\n";
    cout << "===================================\n";

    return 0;
}