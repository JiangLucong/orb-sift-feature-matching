// From-scratch SIFT keypoint detector (port of the original sift.py).
//   Gaussian pyramid -> Difference-of-Gaussian pyramid -> 3x3x3 scale-space
//   extrema with contrast + Hessian edge filtering -> gradient-histogram
//   orientation assignment.
#include "custom_features.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace cf {

static const int    kNumOctaves = 3;
static const int    kNumScales  = 5;
static const double kSigma0     = 1.6;
static const double kK          = 1.4142135623730951; // sqrt(2)

// step 1: Gaussian pyramid — kNumScales+3 increasingly blurred images per octave
static std::vector<std::vector<cv::Mat>> buildGaussianPyramid(cv::Mat img) {
    std::vector<std::vector<cv::Mat>> pyramid;
    cv::Mat base = img.clone();
    for (int octave = 0; octave < kNumOctaves; ++octave) {
        std::vector<cv::Mat> imgs;
        for (int scale = 0; scale < kNumScales + 3; ++scale) {
            cv::Mat blur;
            if (scale == 0) {
                cv::GaussianBlur(base, blur, cv::Size(), kSigma0);
            } else {
                double sigmaPrev = kSigma0 * std::pow(kK, scale - 1);
                double sigmaDiff = std::sqrt(kK * kK - 1.0) * sigmaPrev;
                cv::GaussianBlur(imgs.back(), blur, cv::Size(), sigmaDiff);
            }
            imgs.push_back(blur);
        }
        pyramid.push_back(imgs);
        cv::Mat down;                              // next octave: downsample 3rd-to-last
        cv::pyrDown(imgs[imgs.size() - 3], down);
        base = down;
    }
    return pyramid;
}

// step 2: Difference-of-Gaussian pyramid
static std::vector<std::vector<cv::Mat>> buildDoGPyramid(
        const std::vector<std::vector<cv::Mat>>& gaussian) {
    std::vector<std::vector<cv::Mat>> dog;
    for (const auto& octave : gaussian) {
        std::vector<cv::Mat> dogs;
        for (size_t i = 1; i < octave.size(); ++i) {
            cv::Mat d;
            cv::subtract(octave[i], octave[i - 1], d);
            dogs.push_back(d);
        }
        dog.push_back(dogs);
    }
    return dog;
}

// step 3: scale-space extrema with contrast + Hessian edge-response filtering
struct RawKp { int x, y, octave, scale; };

static void detectInOctave(const std::vector<cv::Mat>& dog, int octave,
        double contrastThresh, double r, std::vector<RawKp>& out) {
    const double edgeThresh = (r + 1) * (r + 1) / r;
    for (int s = 1; s + 1 < (int)dog.size(); ++s) {
        const cv::Mat& prev = dog[s - 1];
        const cv::Mat& curr = dog[s];
        const cv::Mat& next = dog[s + 1];

        cv::Mat Dxx, Dyy, Dxy;                     // Hessian of the current DoG layer
        cv::Sobel(curr, Dxx, CV_64F, 2, 0, 3);
        cv::Sobel(curr, Dyy, CV_64F, 0, 2, 3);
        cv::Sobel(curr, Dxy, CV_64F, 1, 1, 3);

        const int h = curr.rows, w = curr.cols;
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                double val = (double)curr.at<uchar>(y, x);
                if (std::abs(val) <= contrastThresh) continue;       // low-contrast

                double trH  = Dxx.at<double>(y, x) + Dyy.at<double>(y, x);
                double detH = Dxx.at<double>(y, x) * Dyy.at<double>(y, x)
                            - Dxy.at<double>(y, x) * Dxy.at<double>(y, x);
                if (detH <= 0) detH = 1e-10;
                if (trH * trH / detH >= edgeThresh) continue;        // edge-like

                uchar c = curr.at<uchar>(y, x);                      // 3x3x3 extremum?
                bool isMax = true, isMin = true;
                for (int dy = -1; dy <= 1 && (isMax || isMin); ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        uchar p = prev.at<uchar>(y + dy, x + dx);
                        uchar u = curr.at<uchar>(y + dy, x + dx);
                        uchar n = next.at<uchar>(y + dy, x + dx);
                        uchar mx = std::max({p, u, n}), mn = std::min({p, u, n});
                        if (mx > c) isMax = false;
                        if (mn < c) isMin = false;
                    }
                if (isMax || isMin) out.push_back({x, y, octave, s});
            }
        }
    }
}

// step 4: dominant orientation from a Gaussian-weighted gradient histogram
static cv::KeyPoint orientKeypoint(const std::vector<std::vector<cv::Mat>>& gaussian,
        const RawKp& kp, int numBins = 36) {
    const cv::Mat& img = gaussian[kp.octave][kp.scale];
    const int w = img.cols, h = img.rows;
    double sigma  = kSigma0 * std::pow(2.0, kp.octave) * std::pow(kK, kp.scale);
    int    radius = (int)(3 * sigma);

    int x0 = std::max(0, kp.x - radius), x1 = std::min(w, kp.x + radius + 1);
    int y0 = std::max(0, kp.y - radius), y1 = std::min(h, kp.y + radius + 1);
    cv::Mat patch = img(cv::Range(y0, y1), cv::Range(x0, x1));

    cv::Mat gx, gy;
    cv::Sobel(patch, gx, CV_64F, 1, 0, 3);
    cv::Sobel(patch, gy, CV_64F, 0, 1, 3);

    std::vector<double> hist(numBins, 0.0);
    const double denom = 2.0 * (1.5 * sigma) * (1.5 * sigma);
    for (int j = 0; j < patch.rows; ++j)
        for (int i = 0; i < patch.cols; ++i) {
            double gxv = gx.at<double>(j, i), gyv = gy.at<double>(j, i);
            double mag = std::sqrt(gxv * gxv + gyv * gyv);
            double ori = std::fmod(std::atan2(gyv, gxv) * 180.0 / CV_PI + 360.0, 360.0);
            int px = x0 + i, py = y0 + j;
            double wgt = std::exp(-((px - kp.x) * (px - kp.x) + (py - kp.y) * (py - kp.y)) / denom);
            int bin = std::min(numBins - 1, (int)(ori / 360.0 * numBins));
            hist[bin] += mag * wgt;
        }
    int best = 0;
    for (int b = 1; b < numBins; ++b) if (hist[b] > hist[best]) best = b;

    cv::KeyPoint out;
    double sc = std::pow(2.0, kp.octave);
    out.pt     = cv::Point2f((float)(kp.x * sc), (float)(kp.y * sc));
    out.size   = (float)(2.0 * std::max(1.0, 3 * kSigma0 * sc * std::pow(kK, kp.scale)));
    out.angle  = (float)(best * (360.0 / numBins));
    out.octave = kp.octave;
    return out;
}

std::vector<cv::KeyPoint> detectSIFT(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    auto gaussian = buildGaussianPyramid(gray);
    auto dog      = buildDoGPyramid(gaussian);

    std::vector<RawKp> raw;
    for (int o = 0; o < (int)dog.size(); ++o)
        detectInOctave(dog[o], o, /*contrast*/ 20.0, /*r*/ 10.0, raw);

    std::vector<cv::KeyPoint> kps;
    kps.reserve(raw.size());
    for (const auto& r : raw) kps.push_back(orientKeypoint(gaussian, r));
    return kps;
}

} // namespace cf
