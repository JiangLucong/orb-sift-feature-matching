// Demo 2 — feature matching + object localization across two views.
//
//   detect ORB/SIFT features in an "object" image and a "scene" image, match
//   descriptors with a Lowe ratio test, then estimate a homography with RANSAC
//   and draw the localized object box in the scene.
//
//   Usage:  match [ORB|SIFT] [object] [scene]
//     - method defaults to ORB
//     - object defaults to assets/scene.jpg
//     - if no scene is given, a synthetic perspective-warped scene is generated
//       so the demo is self-contained.
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <string>
#include <vector>

// Build a synthetic scene: paste a perspective-warped copy of the object onto a
// larger canvas, so matching + localization can be demonstrated from one image.
static cv::Mat synthesizeScene(const cv::Mat& object) {
    const float W = (float)object.cols, H = (float)object.rows;
    std::vector<cv::Point2f> src = {{0, 0}, {W, 0}, {W, H}, {0, H}};
    std::vector<cv::Point2f> dst = {{W * 0.28f, H * 0.18f}, {W * 1.12f, H * 0.05f},
                                    {W * 1.22f, H * 1.15f}, {W * 0.08f, H * 1.05f}};
    cv::Mat warp = cv::getPerspectiveTransform(src, dst);
    cv::Mat scene;
    cv::warpPerspective(object, scene, warp,
                        cv::Size((int)(W * 1.35f), (int)(H * 1.35f)));
    return scene;
}

int main(int argc, char** argv) {
    int ai = 1;
    std::string method = "ORB";
    if (argc > ai && (std::string(argv[ai]) == "ORB" || std::string(argv[ai]) == "SIFT"))
        method = argv[ai++];

    std::string objectPath = (argc > ai) ? argv[ai++] : "assets/scene.jpg";
    cv::Mat object = cv::imread(objectPath);
    if (object.empty()) {
        std::cerr << "Cannot read object image: " << objectPath << "\n";
        return 1;
    }
    cv::Mat scene;
    if (argc > ai) scene = cv::imread(argv[ai++]);
    if (scene.empty()) scene = synthesizeScene(object);

    // 1) detect + describe
    cv::Ptr<cv::Feature2D> det;
    if (method == "SIFT") det = cv::SIFT::create();
    else                  det = cv::ORB::create(1500);
    std::vector<cv::KeyPoint> k1, k2;
    cv::Mat d1, d2;
    det->detectAndCompute(object, cv::noArray(), k1, d1);
    det->detectAndCompute(scene,  cv::noArray(), k2, d2);

    // 2) kNN match + Lowe ratio test
    int normType = (method == "SIFT") ? cv::NORM_L2 : cv::NORM_HAMMING;
    cv::BFMatcher matcher(normType);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);
    std::vector<cv::DMatch> good;
    for (auto& m : knn)
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) good.push_back(m[0]);

    std::cout << method << ": " << k1.size() << " vs " << k2.size()
              << " keypoints, " << good.size() << " good matches\n";

    // 3) homography (RANSAC) + localization
    cv::Mat vis;
    if (good.size() >= 4) {
        std::vector<cv::Point2f> p1, p2;
        for (auto& g : good) { p1.push_back(k1[g.queryIdx].pt); p2.push_back(k2[g.trainIdx].pt); }
        std::vector<uchar> mask;
        cv::Mat Hm = cv::findHomography(p1, p2, cv::RANSAC, 5.0, mask);
        int inliers = cv::countNonZero(mask);
        std::cout << "RANSAC inliers: " << inliers << "/" << (int)good.size() << "\n";

        std::vector<cv::DMatch> inlierMatches;
        for (size_t i = 0; i < good.size(); ++i) if (mask[i]) inlierMatches.push_back(good[i]);
        cv::drawMatches(object, k1, scene, k2, inlierMatches, vis, cv::Scalar(0, 255, 0),
                        cv::Scalar(160, 160, 160), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        if (!Hm.empty()) {
            std::vector<cv::Point2f> corners = {{0, 0}, {(float)object.cols, 0},
                                                {(float)object.cols, (float)object.rows},
                                                {0, (float)object.rows}};
            std::vector<cv::Point2f> projected;
            cv::perspectiveTransform(corners, projected, Hm);
            for (auto& pt : projected) pt.x += object.cols; // offset into the scene panel
            for (int i = 0; i < 4; ++i)
                cv::line(vis, projected[i], projected[(i + 1) % 4], cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
        }
    } else {
        std::cout << "Not enough matches for homography.\n";
        cv::drawMatches(object, k1, scene, k2, good, vis);
    }

    cv::imwrite("matching_result.png", vis);
    std::cout << "saved matching_result.png\n";
    return 0;
}
