#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// -------------------------------
// FACE ALIGNMENT
// -------------------------------
cv::Mat get_config_warp(const float* lm_ptr, const cv::Rect& face_rect, int size) {
    float scale = (float)size / 112.0f;

    std::vector<cv::Point2f> dst_pts = {
        {38.2946f * scale, 51.6963f * scale},  // left eye
        {73.5318f * scale, 51.5014f * scale},  // right eye
        {56.0252f * scale, 71.7366f * scale}   // nose
    };

    std::vector<cv::Point2f> src_pts;
    for (int i = 0; i < 3; i++) {
        src_pts.emplace_back(
            lm_ptr[i * 2] * face_rect.width + face_rect.x,
            lm_ptr[i * 2 + 1] * face_rect.height + face_rect.y
        );
    }

    return cv::getAffineTransform(src_pts, dst_pts);
}

// -------------------------------
// CHW -> HWC (for Inswapper output)
// -------------------------------
cv::Mat nchw_to_hwc(const float* data, int H, int W) {
    cv::Mat img(H, W, CV_8UC3);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img.at<cv::Vec3b>(y, x)[c] =
                (uchar)std::clamp(data[c * H * W + y * W + x] * 255.0f, 0.0f, 255.0f);
    return img;
}

int main() {
    try {
        ov::Core core;

        // -------------------------------
        // LOAD MODELS
        // -------------------------------

        // Detection
        auto det_model = core.read_model("face-detection-retail-0005.xml");
        {
            ov::preprocess::PrePostProcessor ppp(det_model);
            ppp.input().tensor()
                .set_element_type(ov::element::u8)
                .set_layout("NHWC")
                .set_color_format(ov::preprocess::ColorFormat::BGR);
            ppp.input().preprocess()
                .convert_element_type(ov::element::f32)
                .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            ppp.input().model().set_layout("NCHW");
            det_model = ppp.build();
        }

        // Landmarks
        
        auto lm_model = core.read_model("landmarks-regression-retail-0009.xml");
        {
            ov::preprocess::PrePostProcessor ppp(lm_model);
            ppp.input().tensor()
                .set_element_type(ov::element::u8)
                .set_layout("NHWC")
                .set_color_format(ov::preprocess::ColorFormat::BGR);
            ppp.input().preprocess()
                .convert_element_type(ov::element::f32)
                .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            ppp.input().model().set_layout("NCHW");
            lm_model = ppp.build();
        }

        // ArcFace
		std::cout << "Loading ArcFace model and preparing PPP...\n";
        auto arc_model = core.read_model("arc.onnx");
        {
            for (auto& input : arc_model->inputs())
                std::cout << "ArcFace input: " << input.get_any_name() << std::endl;

            ov::preprocess::PrePostProcessor ppp(arc_model);
            ppp.input("input_1").tensor()
                .set_element_type(ov::element::u8)
                .set_layout("NHWC")
                .set_color_format(ov::preprocess::ColorFormat::BGR);
            ppp.input("input_1").preprocess()
                .convert_element_type(ov::element::f32)
                .scale({ 255.0f,255.0f,255.0f });
            arc_model = ppp.build();
        }

        // Compile models
        auto det_compiled = core.compile_model(det_model, "GPU");
        auto lm_compiled = core.compile_model(lm_model, "GPU");
        auto arc_compiled = core.compile_model(arc_model, "GPU");

        auto det_req = det_compiled.create_infer_request();
        auto lm_req = lm_compiled.create_infer_request();
        auto arc_req = arc_compiled.create_infer_request();

        // -------------------------------
        // LOAD SOURCE EMBEDDING
        // -------------------------------

		std::cout << "Loading source image and extracting ArcFace embedding...\n";
        
        cv::Mat source = cv::imread("source.jpg");
        cv::resize(source, source, cv::Size(112, 112));

        ov::Tensor src_tensor(ov::element::u8, { 1,(size_t)source.rows,(size_t)source.cols,3 }, source.data);
        arc_req.set_input_tensor(src_tensor);
        arc_req.infer();

        std::vector<float> source_id(512);
        std::memcpy(source_id.data(), arc_req.get_output_tensor().data<float>(), 512 * sizeof(float));
        std::cout << "ArcFace embedding ready.\n";

        // -------------------------------
        // LOAD SWAPPER (ONNX) without PPP layout changes
        // -------------------------------

        auto swp_model = core.read_model("swapper.onnx"); //error here which i cant be bothered to fix ==========================================

        for (auto& input : swp_model->inputs()) {
            std::cout << "Swapper input node: " << input.get_any_name() << std::endl;
        }

        auto swp_compiled = core.compile_model(swp_model, "GPU");
        auto swp_req = swp_compiled.create_infer_request();

		std::cout << "Swapper model loaded. Starting camera...\n"; 

        // -------------------------------
        // CAMERA LOOP
        // -------------------------------
        cv::VideoCapture cap(0);
        cv::Mat frame;

        while (cap.read(frame)) {
            if (!frame.isContinuous()) frame = frame.clone();

            // DETECTION
            ov::Tensor det_tensor(ov::element::u8, { 1,(size_t)frame.rows,(size_t)frame.cols,3 }, frame.data);
            det_req.set_input_tensor(det_tensor);
            det_req.infer();

            auto det_out = det_req.get_output_tensor().data<const float>();
            size_t num = det_req.get_output_tensor().get_shape()[2];

            for (size_t i = 0; i < num; i++) {
                float conf = det_out[i * 7 + 2];
                if (conf < 0.7f) continue;

                int x1 = std::max(0, (int)(det_out[i * 7 + 3] * frame.cols));
                int y1 = std::max(0, (int)(det_out[i * 7 + 4] * frame.rows));
                int x2 = std::min(frame.cols, (int)(det_out[i * 7 + 5] * frame.cols));
                int y2 = std::min(frame.rows, (int)(det_out[i * 7 + 6] * frame.rows));

                cv::Rect face_rect(x1, y1, x2 - x1, y2 - y1);
                if (face_rect.width < 40) continue;

                // LANDMARKS
                cv::Mat face = frame(face_rect).clone();
                ov::Tensor lm_tensor(ov::element::u8, { 1,(size_t)face.rows,(size_t)face.cols,3 }, face.data);
                lm_req.set_input_tensor(lm_tensor);
                lm_req.infer();
                const float* lm_ptr = lm_req.get_output_tensor().data<const float>();

                // ALIGN
                cv::Mat warp_m = get_config_warp(lm_ptr, face_rect, 128);
                cv::Mat aligned;
                cv::warpAffine(frame, aligned, warp_m, cv::Size(128, 128));

                // SWAPPER (manual NCHW conversion)
                cv::Mat resized;
                cv::resize(aligned, resized, cv::Size(128, 128));
                resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

                std::vector<float> chw_data(3 * 128 * 128);
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < 128; y++)
                        for (int x = 0; x < 128; x++)
                            chw_data[c * 128 * 128 + y * 128 + x] = resized.at<cv::Vec3f>(y, x)[c];

                ov::Tensor swp_img(ov::element::f32, { 1,3,128,128 }, chw_data.data());
                ov::Tensor swp_id(ov::element::f32, { 1,512 }, source_id.data());
                swp_req.set_input_tensor(0, swp_img);
                swp_req.set_input_tensor(1, swp_id);
                swp_req.infer();

                const float* out = swp_req.get_output_tensor().data<const float>();
                cv::Mat swapped = nchw_to_hwc(out, 128, 128);

                // PASTE BACK
                cv::Mat inv_m, patch;
                cv::invertAffineTransform(warp_m, inv_m);
                cv::warpAffine(swapped, patch, inv_m, frame.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

                cv::Mat gray, mask;
                cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
                cv::threshold(gray, mask, 1, 255, cv::THRESH_BINARY);

                patch.copyTo(frame, mask);
            }

            cv::imshow("Face Swap OpenVINO", frame);
            if (cv::waitKey(1) == 'q') break;
        }

    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }
    return 0;
}