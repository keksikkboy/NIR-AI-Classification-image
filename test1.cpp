#include <Windows.h>
#include <NuiApi.h>
#include <opencv2/opencv.hpp>
#include <iostream>

INuiSensor* sensor = nullptr;
HANDLE colorStreamHandle = nullptr;

bool initKinect()
{
    int sensorCount = 0;
    if (FAILED(NuiGetSensorCount(&sensorCount)) || sensorCount < 1) {
        std::cerr << "No Kinect sensors found!" << std::endl;
        return false;
    }

    if (FAILED(NuiCreateSensorByIndex(0, &sensor))) {
        std::cerr << "Failed to create Kinect sensor!" << std::endl;
        return false;
    }

    if (FAILED(sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR))) {
        std::cerr << "Failed to initialize Kinect color stream!" << std::endl;
        return false;
    }

    if (FAILED(sensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION_640x480,
        0, 2, NULL, &colorStreamHandle)))
    {
        std::cerr << "Failed to open color stream!" << std::endl;
        return false;
    }

    return true;
}

cv::Mat getColorFrame()
{
    NUI_IMAGE_FRAME imageFrame;
    if (FAILED(sensor->NuiImageStreamGetNextFrame(colorStreamHandle, 0, &imageFrame))) {
        return cv::Mat();
    }

    INuiFrameTexture* pTexture = imageFrame.pFrameTexture;
    NUI_LOCKED_RECT lockedRect;
    pTexture->LockRect(0, &lockedRect, NULL, 0);

    cv::Mat frame;
    if (lockedRect.Pitch != 0) {
        cv::Mat temp(cv::Size(640, 480), CV_8UC4, lockedRect.pBits);
        frame = temp.clone();
    }

    pTexture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(colorStreamHandle, &imageFrame);

    return frame;
}

// Проверка "круглости" контура
bool isCircle(const std::vector<cv::Point>& contour)
{
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    if (perimeter == 0) return false;
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
    return (circularity > 0.7); // эмпирически: ближе к 1 — круг
}

int main()
{
    if (!initKinect()) return -1;
    std::cout << "Kinect initialized. Press ESC to exit." << std::endl;

    while (true) {
        cv::Mat frame = getColorFrame();
        if (!frame.empty()) {
            cv::Mat bgr;
            cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);

            cv::Mat hsv;
            cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

            // Маска красного
            cv::Mat maskRed1, maskRed2, maskRed;
            cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), maskRed1);
            cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), maskRed2);
            cv::bitwise_or(maskRed1, maskRed2, maskRed);

            // Маска синего
            cv::Mat maskBlue;
            cv::inRange(hsv, cv::Scalar(100, 120, 70), cv::Scalar(130, 255, 255), maskBlue);

            // Немного морфологии для чистоты
            cv::morphologyEx(maskRed, maskRed, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2);
            cv::morphologyEx(maskBlue, maskBlue, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2);

            // Поиск контуров
            std::vector<std::vector<cv::Point>> contoursRed, contoursBlue;
            cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::findContours(maskBlue, contoursBlue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Анализ красных контуров
            for (auto& contour : contoursRed) {
                if (cv::contourArea(contour) > 500 && isCircle(contour)) {
                    cv::Rect box = cv::boundingRect(contour);
                    cv::rectangle(bgr, box, cv::Scalar(0, 0, 255), 2);
                    cv::putText(bgr, "STOP", cv::Point(box.x, box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                }
            }

            // Анализ синих контуров
            for (auto& contour : contoursBlue) {
                if (cv::contourArea(contour) > 500 && isCircle(contour)) {
                    cv::Rect box = cv::boundingRect(contour);
                    cv::rectangle(bgr, box, cv::Scalar(255, 0, 0), 2);
                    cv::putText(bgr, "RIGHT", cv::Point(box.x, box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
                }
            }

            // Вывод результатов
            cv::imshow("Kinect RGB", bgr);
            cv::imshow("Mask Red", maskRed);
            cv::imshow("Mask Blue", maskBlue);
        }

        if (cv::waitKey(30) == 27) break;
    }

    sensor->NuiShutdown();
    return 0;
}