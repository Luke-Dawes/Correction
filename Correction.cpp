#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Create a blank black image (400x200 pixels)
    cv::Mat image = cv::Mat::zeros(200, 400, CV_8UC3);

    // 2. Add some text to the image
    std::string text = "OpenCV " + std::string(CV_VERSION) + " works!";
    cv::putText(image, text, cv::Point(50, 100),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 3. Show the image in a window
    cv::imshow("OpenCV Test", image);

    std::cout << "Successfully loaded OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "Press any key to close the window..." << std::endl;

    // 4. Wait for a key press to close
    cv::waitKey(0);
    return 0;
}
