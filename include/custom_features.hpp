#pragma once
#include <opencv2/core.hpp>
#include <vector>

// From-scratch ORB / SIFT keypoint detectors, ported from the original Python
// implementation. They use OpenCV only for low-level primitives (resize, FAST,
// Harris, Gaussian blur, Sobel); the pyramid construction, scale-space extrema
// detection, edge filtering and orientation assignment are implemented by hand.
namespace cf {

// ORB-style detector: image pyramid -> FAST corners -> Harris edge filtering ->
// intensity-centroid orientation. Returns keypoints in full-resolution coords.
std::vector<cv::KeyPoint> detectORB(const cv::Mat& bgr);

// SIFT-style detector: Gaussian + DoG pyramids -> 3x3x3 scale-space extrema with
// contrast + Hessian edge filtering -> gradient-histogram orientation.
std::vector<cv::KeyPoint> detectSIFT(const cv::Mat& bgr);

} // namespace cf
