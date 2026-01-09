/**
 * AccuracyLogger.h
 *
 * Debug logging utility for CPU vs GPU accuracy validation.
 * Exports intermediate results (keypoints, descriptors) to files
 * for comparative analysis.
 *
 * Usage: Enable with -DENABLE_DEBUG_LOGGING during compilation
 *
 * This is an "Add-on" component - does not modify core logic.
 */

#ifndef ACCURACY_LOGGER_H
#define ACCURACY_LOGGER_H

#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace ORB_SLAM3
{
    namespace validation
    {

        /**
         * AccuracyLogger - Exports ORB extraction results for validation
         *
         * Output format (per frame):
         * FRAME <frame_id>
         * KEYPOINTS <count>
         * KP <x> <y> <response> <size> <angle> <octave>
         * ...
         * DESCRIPTOR_HASH <hash_value>  // For quick comparison
         * END_FRAME
         */
        class AccuracyLogger
        {
        public:
            enum class Mode
            {
                CPU,
                GPU
            };

            AccuracyLogger(const std::string &outputPath, Mode mode)
                : mode_(mode), frameCount_(0), totalKeypoints_(0)
            {
                std::string filename = outputPath + "/" +
                                       (mode == Mode::CPU ? "cpu_accuracy.log" : "gpu_accuracy.log");
                file_.open(filename, std::ios::out | std::ios::trunc);

                if (!file_.is_open())
                {
                    throw std::runtime_error("Failed to open accuracy log: " + filename);
                }

                // Write header
                file_ << "# ORB-SLAM3 Accuracy Validation Log\n";
                file_ << "# Mode: " << (mode == Mode::CPU ? "CPU" : "GPU") << "\n";
                file_ << "# Format: CSV-like per frame\n";
                file_ << "# ================================================\n";
                file_.flush();
            }

            ~AccuracyLogger()
            {
                if (file_.is_open())
                {
                    writeSummary();
                    file_.close();
                }
            }

            /**
             * Log keypoints and descriptors for a frame
             * @param frameId Frame number
             * @param keypoints Detected keypoints
             * @param descriptors ORB descriptors (optional, can be empty)
             * @param imageSize Image dimensions for normalization
             */
            void logFrame(int frameId,
                          const std::vector<cv::KeyPoint> &keypoints,
                          const cv::Mat &descriptors = cv::Mat(),
                          cv::Size imageSize = cv::Size(0, 0))
            {
                if (!file_.is_open())
                    return;

                frameCount_++;
                totalKeypoints_ += keypoints.size();

                file_ << "FRAME " << frameId << "\n";
                file_ << "IMAGE_SIZE " << imageSize.width << " " << imageSize.height << "\n";
                file_ << "KEYPOINTS " << keypoints.size() << "\n";

                // Log each keypoint (sorted by response for consistency)
                std::vector<cv::KeyPoint> sortedKps = keypoints;
                std::sort(sortedKps.begin(), sortedKps.end(),
                          [](const cv::KeyPoint &a, const cv::KeyPoint &b)
                          {
                              if (std::abs(a.response - b.response) > 0.001f)
                                  return a.response > b.response;
                              if (std::abs(a.pt.x - b.pt.x) > 0.001f)
                                  return a.pt.x < b.pt.x;
                              return a.pt.y < b.pt.y;
                          });

                // Log top N keypoints with full detail (configurable)
                const int MAX_DETAILED_KPS = 500;
                int detailedCount = std::min(static_cast<int>(sortedKps.size()), MAX_DETAILED_KPS);

                for (int i = 0; i < detailedCount; i++)
                {
                    const auto &kp = sortedKps[i];
                    file_ << "KP " << std::fixed << std::setprecision(2)
                          << kp.pt.x << " " << kp.pt.y << " "
                          << std::setprecision(4) << kp.response << " "
                          << std::setprecision(1) << kp.size << " "
                          << kp.angle << " " << kp.octave << "\n";
                }

                // Compute keypoint distribution statistics (octave histogram)
                std::vector<int> octaveHist(8, 0);
                for (const auto &kp : keypoints)
                {
                    int oct = std::max(0, std::min(7, kp.octave));
                    octaveHist[oct]++;
                }
                file_ << "OCTAVE_HIST";
                for (int c : octaveHist)
                    file_ << " " << c;
                file_ << "\n";

                // Compute spatial distribution (grid-based)
                if (imageSize.width > 0 && imageSize.height > 0)
                {
                    const int GRID_COLS = 8;
                    const int GRID_ROWS = 6;
                    std::vector<int> spatialGrid(GRID_COLS * GRID_ROWS, 0);
                    float cellW = imageSize.width / static_cast<float>(GRID_COLS);
                    float cellH = imageSize.height / static_cast<float>(GRID_ROWS);

                    for (const auto &kp : keypoints)
                    {
                        int gx = std::min(GRID_COLS - 1, static_cast<int>(kp.pt.x / cellW));
                        int gy = std::min(GRID_ROWS - 1, static_cast<int>(kp.pt.y / cellH));
                        spatialGrid[gy * GRID_COLS + gx]++;
                    }

                    file_ << "SPATIAL_GRID " << GRID_COLS << " " << GRID_ROWS;
                    for (int c : spatialGrid)
                        file_ << " " << c;
                    file_ << "\n";
                }

                // Descriptor statistics (if available)
                if (!descriptors.empty())
                {
                    file_ << "DESCRIPTORS " << descriptors.rows << " " << descriptors.cols << "\n";

                    // Compute a hash of the descriptors for quick comparison
                    // Using XOR-based checksum
                    uint64_t hash = 0;
                    for (int r = 0; r < descriptors.rows; r++)
                    {
                        const uchar *row = descriptors.ptr<uchar>(r);
                        for (int c = 0; c < descriptors.cols; c++)
                        {
                            hash ^= static_cast<uint64_t>(row[c]) << ((r + c) % 56);
                            hash = (hash << 7) | (hash >> 57); // Rotate
                        }
                    }
                    file_ << "DESCRIPTOR_HASH " << std::hex << hash << std::dec << "\n";

                    // Log first few descriptors for detailed comparison
                    const int MAX_DESC_LOG = 50;
                    int descToLog = std::min(descriptors.rows, MAX_DESC_LOG);
                    for (int r = 0; r < descToLog; r++)
                    {
                        file_ << "DESC " << r << " ";
                        const uchar *row = descriptors.ptr<uchar>(r);
                        for (int c = 0; c < descriptors.cols; c++)
                        {
                            file_ << std::setw(2) << std::setfill('0') << std::hex
                                  << static_cast<int>(row[c]);
                        }
                        file_ << std::dec << "\n";
                    }
                }

                file_ << "END_FRAME\n\n";

                // Flush periodically
                if (frameCount_ % 100 == 0)
                {
                    file_.flush();
                }
            }

            /**
             * Log matching results (for ORBmatcher validation)
             */
            void logMatches(int frameId1, int frameId2,
                            const std::vector<cv::DMatch> &matches,
                            int totalKps1, int totalKps2)
            {
                if (!file_.is_open())
                    return;

                file_ << "MATCH " << frameId1 << " " << frameId2 << "\n";
                file_ << "MATCH_COUNT " << matches.size() << "\n";
                file_ << "KP_COUNTS " << totalKps1 << " " << totalKps2 << "\n";

                // Log match indices
                for (const auto &m : matches)
                {
                    file_ << "M " << m.queryIdx << " " << m.trainIdx
                          << " " << std::fixed << std::setprecision(2) << m.distance << "\n";
                }
                file_ << "END_MATCH\n\n";
            }

            int getFrameCount() const { return frameCount_; }
            size_t getTotalKeypoints() const { return totalKeypoints_; }

        private:
            void writeSummary()
            {
                file_ << "\n# ================================================\n";
                file_ << "# SUMMARY\n";
                file_ << "# Frames: " << frameCount_ << "\n";
                file_ << "# Total Keypoints: " << totalKeypoints_ << "\n";
                file_ << "# Avg Keypoints/Frame: "
                      << (frameCount_ > 0 ? totalKeypoints_ / frameCount_ : 0) << "\n";
                file_ << "# ================================================\n";
            }

            std::ofstream file_;
            Mode mode_;
            int frameCount_;
            size_t totalKeypoints_;
        };

// Convenience macros for conditional logging
#ifdef ENABLE_DEBUG_LOGGING
#define ACCURACY_LOG_INIT_CPU(path)                                                  \
    static std::unique_ptr<ORB_SLAM3::validation::AccuracyLogger> s_accuracyLogger = \
        std::make_unique<ORB_SLAM3::validation::AccuracyLogger>(                     \
            path, ORB_SLAM3::validation::AccuracyLogger::Mode::CPU)

#define ACCURACY_LOG_INIT_GPU(path)                                                  \
    static std::unique_ptr<ORB_SLAM3::validation::AccuracyLogger> s_accuracyLogger = \
        std::make_unique<ORB_SLAM3::validation::AccuracyLogger>(                     \
            path, ORB_SLAM3::validation::AccuracyLogger::Mode::GPU)

#define ACCURACY_LOG_FRAME(frameId, keypoints, descriptors, imageSize) \
    if (s_accuracyLogger)                                              \
    s_accuracyLogger->logFrame(frameId, keypoints, descriptors, imageSize)

#define ACCURACY_LOG_MATCHES(f1, f2, matches, kps1, kps2) \
    if (s_accuracyLogger)                                 \
    s_accuracyLogger->logMatches(f1, f2, matches, kps1, kps2)
#else
#define ACCURACY_LOG_INIT_CPU(path)
#define ACCURACY_LOG_INIT_GPU(path)
#define ACCURACY_LOG_FRAME(frameId, keypoints, descriptors, imageSize)
#define ACCURACY_LOG_MATCHES(f1, f2, matches, kps1, kps2)
#endif

    } // namespace validation
} // namespace ORB_SLAM3

#endif // ACCURACY_LOGGER_H
