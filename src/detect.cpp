// Demo 1 — run the from-scratch ORB and SIFT detectors on an image and save
// keypoint visualizations.
//
//   Usage:  detect [image]          (default: assets/scene.jpg)
#include "custom_features.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "assets/scene.jpg";
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << path << "\n";
        return 1;
    }

    std::vector<cv::KeyPoint> orbKps  = cf::detectORB(img);
    std::vector<cv::KeyPoint> siftKps = cf::detectSIFT(img);
    std::cout << "custom ORB  keypoints: " << orbKps.size()  << "\n";
    std::cout << "custom SIFT keypoints: " << siftKps.size() << "\n";

    cv::Mat visOrb, visSift;
    const auto flags = cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
    cv::drawKeypoints(img, orbKps,  visOrb,  cv::Scalar(255, 64, 128), flags);
    cv::drawKeypoints(img, siftKps, visSift, cv::Scalar(255, 64, 128), flags);
    cv::imwrite("custom_orb_keypoints.png",  visOrb);
    cv::imwrite("custom_sift_keypoints.png", visSift);
    std::cout << "saved custom_orb_keypoints.png and custom_sift_keypoints.png\n";
    return 0;
}
