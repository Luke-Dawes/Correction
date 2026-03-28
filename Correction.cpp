#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

int main() {
    try {
        ov::Core core;

        // --- 1. SETUP FACE DETECTION (300x300 input) ---
        auto det_model = core.read_model("face-detection-retail-0005.xml");
        ov::preprocess::PrePostProcessor det_ppp(det_model);
        det_ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        det_ppp.input().model().set_layout("NCHW");
        det_model = det_ppp.build();
        auto det_compiled = core.compile_model(det_model, "GPU");
        auto det_request = det_compiled.create_infer_request();

        // --- 2. SETUP LANDMARKS (48x48 input) ---
        auto lm_model = core.read_model("landmarks-regression-retail-0009.xml");
        ov::preprocess::PrePostProcessor lm_ppp(lm_model);
        lm_ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        lm_ppp.input().model().set_layout("NCHW");
        lm_model = lm_ppp.build();
        auto lm_compiled = core.compile_model(lm_model, "GPU");
        auto lm_request = lm_compiled.create_infer_request();

        // --- 3. VIDEO LOOP ---
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) return -1;

        cv::Mat frame;
        while (cap.read(frame)) {
            // STEP A: Prepare Detection Input
            cv::Mat det_resized;
            cv::resize(frame, det_resized, cv::Size(300, 300));
            ov::Tensor det_input(ov::element::u8, { 1, 300, 300, 3 }, det_resized.data);

            det_request.set_input_tensor(det_input);
            det_request.infer();

            auto det_output = det_request.get_output_tensor();
            const float* detections = det_output.data<const float>();

            for (int i = 0; i < 200; i++) {
                float confidence = detections[i * 7 + 2];
                if (confidence > 0.7f) {
                    // Calculate Box Coordinates relative to original frame
                    int x_min = std::max(0, (int)(detections[i * 7 + 3] * frame.cols));
                    int y_min = std::max(0, (int)(detections[i * 7 + 4] * frame.rows));
                    int x_max = std::min(frame.cols, (int)(detections[i * 7 + 5] * frame.cols));
                    int y_max = std::min(frame.rows, (int)(detections[i * 7 + 6] * frame.rows));

                    cv::Rect face_rect(x_min, y_min, x_max - x_min, y_max - y_min);
                    if (face_rect.width < 10 || face_rect.height < 10) continue;

                    // STEP B: Prepare Landmarks Input (Crop -> Resize to 48x48)
                    cv::Mat face_crop = frame(face_rect);
                    cv::Mat lm_resized;
                    cv::resize(face_crop, lm_resized, cv::Size(48, 48));

                    ov::Tensor lm_input(ov::element::u8, { 1, 48, 48, 3 }, lm_resized.data);
                    lm_request.set_input_tensor(lm_input);
                    lm_request.infer();

                    auto lm_output = lm_request.get_output_tensor();
                    const float* lm_data = lm_output.data<const float>();

                    // STEP C: Draw Results
                    cv::rectangle(frame, face_rect, cv::Scalar(0, 255, 0), 2);
                    for (int j = 0; j < 5; j++) {
                        // Landmarks are normalized 0-1 within the crop
                        float lx = lm_data[j * 2] * face_rect.width + face_rect.x;
                        float ly = lm_data[j * 2 + 1] * face_rect.height + face_rect.y;
                        cv::circle(frame, cv::Point((int)lx, (int)ly), 3, cv::Scalar(0, 0, 255), -1);
                    }
                }
            }

            cv::imshow("OpenVINO Face + Landmarks", frame);
            if (cv::waitKey(1) == 'q') break;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
