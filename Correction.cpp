//error [ERROR:0@0.390] global obsensor_uvc_stream_channel.cpp:163 cv::obsensor::getStreamChannelGroup Camera index out of range Failed to capture frame cause by antivirus

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>

int main() {
    
    cv::VideoCapture cap(0);

    cv::UMat frame;

	ov::Core core;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }

		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        cv::imshow("Camera", frame);
        if (cv::waitKey(30) >= 0) {
            break;
		}
    }
    cv::waitKey(0);
    return 0;
}
