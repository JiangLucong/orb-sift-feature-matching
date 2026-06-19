// From-scratch ORB keypoint detector (port of the original orb.py).
//   image pyramid -> FAST corners -> Harris response filtering (top-N) ->
//   intensity-centroid orientation -> rescale keypoints to full resolution.
#include "custom_features.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <algorithm>
#include <cmath>

namespace cf {

static const double kScaleFactor = 1.4142135623730951; // sqrt(2)

// step 1: build an image pyramid by repeatedly downsampling
static std::vector<cv::Mat> buildImagePyramid(const cv::Mat& img, int numLevels = 8) {
    std::vector<cv::Mat> pyramid{ img };
    for (int level = 1; level < numLevels; ++level) {
        double s = 1.0 / std::pow(kScaleFactor, level);
        cv::Mat scaled;
        cv::resize(img, scaled, cv::Size(), s, s);
        pyramid.push_back(scaled);
    }
    return pyramid;
}

// step 2: detect FAST corners (with non-maximal suppression)
static std::vector<cv::KeyPoint> detectFast(const cv::Mat& img, int threshold = 20) {
    auto fast = cv::FastFeatureDetector::create(threshold);
    fast->setNonmaxSuppression(true);
    std::vector<cv::KeyPoint> keypoints;
    fast->detect(img, keypoints);
    return keypoints;
}

// step 3: score corners with the Harris response and keep the strongest top-N
static std::vector<cv::KeyPoint> harrisFilter(const cv::Mat& img, std::vector<cv::KeyPoint> kps,
        int blockSize = 3, int ksize = 3, double k = 0.06, int topN = 100) {
    cv::Mat imgF;  img.convertTo(imgF, CV_32F);
    cv::Mat harris;  cv::cornerHarris(imgF, harris, blockSize, ksize, k);
    for (auto& p : kps) {
        int x = (int)p.pt.x, y = (int)p.pt.y;
        if (x >= 0 && x < harris.cols && y >= 0 && y < harris.rows)
            p.response = harris.at<float>(y, x);
    }
    std::sort(kps.begin(), kps.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
    if ((int)kps.size() > topN) kps.resize(topN);
    return kps;
}

// step 4: assign orientation from the intensity centroid of a local patch
static void computeOrientation(const cv::Mat& img, std::vector<cv::KeyPoint>& kps, int patchRadius = 9) {
    const int w = img.cols, h = img.rows;
    for (auto& p : kps) {
        int x = (int)p.pt.x, y = (int)p.pt.y;
        int x1 = std::max(0, x - patchRadius), y1 = std::max(0, y - patchRadius);
        int x2 = std::min(w, x + patchRadius + 1), y2 = std::min(h, y + patchRadius + 1);
        double m10 = 0.0, m01 = 0.0;
        for (int yy = y1; yy < y2; ++yy)
            for (int xx = x1; xx < x2; ++xx) {
                double val = img.at<uchar>(yy, xx);
                m10 += (xx - x) * val;
                m01 += (yy - y) * val;
            }
        p.angle = (float)(std::atan2(m01, m10) * 180.0 / CV_PI);
    }
}

std::vector<cv::KeyPoint> detectORB(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Mat> pyramid = buildImagePyramid(gray);

    std::vector<cv::KeyPoint> all;
    for (size_t level = 0; level < pyramid.size(); ++level) {
        std::vector<cv::KeyPoint> kps = detectFast(pyramid[level]);
        kps = harrisFilter(pyramid[level], kps);
        computeOrientation(pyramid[level], kps);

        double sc = std::pow(kScaleFactor, (double)level);
        for (auto& p : kps) {                 // map back to original-image coordinates
            p.pt.x *= (float)sc;
            p.pt.y *= (float)sc;
            p.size  = (float)(5.0 * sc);
            p.octave = (int)level;
            all.push_back(p);
        }
    }
    return all;
}

} // namespace cf
