#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
 
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return -1;
    }

    cv::UMat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("test", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}