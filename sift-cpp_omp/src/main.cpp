// #include <vector>
// #include "image.hpp"
// #include "sift.hpp"
// #include <chrono>
// using namespace std::chrono;


// int main()
// {
//     Image img("./../imgs/book_rotated.jpg");
//     Image img2("./../imgs/book_in_scene.jpg");
//     img = rgb_to_grayscale(img);
//     img2 = rgb_to_grayscale(img2);
//     std::vector<sift::Keypoint> kps1 = sift::find_keypoints_and_descriptors(img);
//     std::vector<sift::Keypoint> kps2 = sift::find_keypoints_and_descriptors(img2);
//     std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps1, kps2);
//     Image book_matches = sift::draw_matches(img, img2, kps1, kps2, matches);
//     book_matches.save("book_matches.jpg");
//     return 0;
// }

#include <vector>
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "image.hpp"
#include "sift.hpp"

using namespace std;
using namespace std::chrono;

int main()
{
    const int N = 10;   // số lần chạy benchmark
    cout << "Benchmark SIFT for N = " << N << " runs\n\n";

    long long gray_total = 0;
    long long sift1_total = 0;
    long long sift2_total = 0;
    long long match_total = 0;
    long long draw_total = 0;
    long long pipeline_total = 0;

    for (int i = 0; i < N; i++)
    {
        cout << "Run " << (i + 1) << "/" << N << "...\n";

        auto t_pipeline_start = high_resolution_clock::now();

        // Load images fresh every run (để đảm bảo công bằng)
        Image img("./../imgs/book_rotated.jpg");
        Image img2("./../imgs/book_in_scene.jpg");

        // ---- grayscale ----
        auto t_gray_start = high_resolution_clock::now();
        img = rgb_to_grayscale(img);
        img2 = rgb_to_grayscale(img2);
        auto t_gray_end = high_resolution_clock::now();
        gray_total += duration_cast<microseconds>(t_gray_end - t_gray_start).count();


        // ---- SIFT 1 ----
        auto t_sift1_start = high_resolution_clock::now();
        std::vector<sift::Keypoint> kps1 = sift::find_keypoints_and_descriptors(img);
        auto t_sift1_end = high_resolution_clock::now();
        sift1_total += duration_cast<microseconds>(t_sift1_end - t_sift1_start).count();


        // ---- SIFT 2 ----
        auto t_sift2_start = high_resolution_clock::now();
        std::vector<sift::Keypoint> kps2 = sift::find_keypoints_and_descriptors(img2);
        auto t_sift2_end = high_resolution_clock::now();
        sift2_total += duration_cast<microseconds>(t_sift2_end - t_sift2_start).count();


        // ---- Matching ----
        auto t_match_start = high_resolution_clock::now();
        std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(kps1, kps2);
        auto t_match_end = high_resolution_clock::now();
        match_total += duration_cast<microseconds>(t_match_end - t_match_start).count();


        // ---- Draw + save ----
        auto t_draw_start = high_resolution_clock::now();
        Image book_matches = sift::draw_matches(img, img2, kps1, kps2, matches);
        book_matches.save("book_matches_last_run.jpg"); // chỉ lưu file cuối cùng
        auto t_draw_end = high_resolution_clock::now();
        draw_total += duration_cast<microseconds>(t_draw_end - t_draw_start).count();


        // ---- pipeline total ----
        auto t_pipeline_end = high_resolution_clock::now();
        pipeline_total += duration_cast<microseconds>(t_pipeline_end - t_pipeline_start).count();
    }

    // ===== PRINT AVERAGES =====
    cout << "\n============= BENCHMARK RESULT =============\n";

    cout << "Average grayscale time: "   << gray_total   / N << " us\n";
    cout << "Average SIFT img1 time: "   << sift1_total  / N << " us\n";
    cout << "Average SIFT img2 time: "   << sift2_total  / N << " us\n";
    cout << "Average matching time: "    << match_total  / N << " us\n";
    cout << "Average draw+save time: "   << draw_total   / N << " us\n";
    cout << "-------------------------------------------\n";
    cout << "Average TOTAL pipeline time: " 
         << pipeline_total / N << " us\n";

    cout << "============================================\n";

    return 0;
}
