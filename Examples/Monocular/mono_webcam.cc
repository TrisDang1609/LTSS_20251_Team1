/**
 * This file is part of ORB-SLAM3
 *
 * Webcam example for monocular camera
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <System.h>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << endl
             << "Usage: ./mono_webcam path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

    // Open webcam
    cv::VideoCapture cap(0); // 0 is usually the built-in camera

    if (!cap.isOpened())
    {
        cerr << "ERROR: Cannot open webcam" << endl;
        return 1;
    }

    // Set camera resolution (optional, adjust as needed)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cout << "Webcam opened successfully" << endl;
    cout << "Resolution: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
         << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    cout << endl
         << "-------" << endl;
    cout << "Start processing webcam ..." << endl;
    cout << "Press 'q' to quit" << endl;

    cv::Mat im;

    // Main loop
    while (true)
    {
        // Capture frame
        cap >> im;

        if (im.empty())
        {
            cerr << "Failed to capture frame" << endl;
            break;
        }

        // Get timestamp (current time)
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        double tframe = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();

        // Resize if needed
        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe);

        // Check for 'q' key to quit
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) // 27 is ESC
            break;
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
