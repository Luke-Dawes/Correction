#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    
    cv::VideoCapture cap(0);

    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        cv::imshow("Camera", frame);
        if (cv::waitKey(30) >= 0) {
            break;
		}
    }


    cv::waitKey(0);
    return 0;
}
