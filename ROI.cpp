#include <Windows.h>
#include <NuiApi.h>
#include <opencv2/opencv.hpp>
#include <iostream>

INuiSensor* sensor = nullptr;
HANDLE colorStreamHandle = nullptr;
HANDLE depthStreamHandle = nullptr;

//инициализация кинепта
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

    if (FAILED(sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH))) {
        std::cerr << "Failed to initialize Kinect streams!" << std::endl;
        return false;
    }

    // Поток цвета
    if (FAILED(sensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION_640x480,
        0, 2, NULL, &colorStreamHandle)))
    {
        std::cerr << "Failed to open color stream!" << std::endl;
        return false;
    }

    // Поток глубины
    if (FAILED(sensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_DEPTH,
        NUI_IMAGE_RESOLUTION_640x480,
        0, 2, NULL, &depthStreamHandle)))
    {
        std::cerr << "Failed to open depth stream!" << std::endl;
        return false;
    }

    return true;
}

//получение кадра цвета
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

//получение кадра глубины (в миллиметрах)
cv::Mat getDepthFrame()
{
    NUI_IMAGE_FRAME depthFrame;
    if (FAILED(sensor->NuiImageStreamGetNextFrame(depthStreamHandle, 0, &depthFrame))) {
        return cv::Mat();
    }

    INuiFrameTexture* pTexture = depthFrame.pFrameTexture;
    NUI_LOCKED_RECT lockedRect;
    pTexture->LockRect(0, &lockedRect, NULL, 0);

    cv::Mat depthMat;
    if (lockedRect.Pitch != 0) {
        USHORT* pBuffer = (USHORT*)lockedRect.pBits;
        depthMat = cv::Mat(480, 640, CV_16U);

        //раскодировка глубины
        for (int y = 0; y < depthMat.rows; ++y) {
            USHORT* rowPtr = depthMat.ptr<USHORT>(y);
            for (int x = 0; x < depthMat.cols; ++x) {
                USHORT raw = pBuffer[y * depthMat.cols + x];
                USHORT depthMM = raw >> 3; // реальные миллиметры
                rowPtr[x] = depthMM;
            }
        }
    }

    pTexture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(depthStreamHandle, &depthFrame);

    return depthMat;
}

//проверка на контур, мы ищем круглый контур как для знака, кф 0,7 даёт камере считать знак под углом перспективы
bool isCircle(const std::vector<cv::Point>& contour)
{
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    if (perimeter == 0) return false;
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
    return (circularity > 0.7);
}

//расчёт средней глубины (в метрах) в ROI
double getAverageDepth(const cv::Mat& depth, const cv::Rect& roi)
{
    if (depth.empty()) return 0.0;

    int x1 = std::max(0, roi.x);
    int y1 = std::max(0, roi.y);
    int x2 = std::min(depth.cols, roi.x + roi.width);
    int y2 = std::min(depth.rows, roi.y + roi.height);

    if (x2 <= x1 || y2 <= y1) return 0.0;

    long long sum = 0;
    int count = 0;

    for (int y = y1; y < y2; ++y) {
        const USHORT* row = depth.ptr<USHORT>(y);
        for (int x = x1; x < x2; ++x) {
            USHORT d = row[x];
            if (d > 300 && d < 10000) { // фильтруем шум
                sum += d;
                ++count;
            }
        }
    }

    if (count == 0) return 0.0;
    double mean_mm = static_cast<double>(sum) / count;
    return mean_mm / 1000.0; // метры
}

int main()
{
    if (!initKinect()) return -1;
    std::cout << "Kinect initialized. Press ESC to exit." << std::endl;

    while (true) {
        cv::Mat color = getColorFrame();
        cv::Mat depth = getDepthFrame();
        if (color.empty() || depth.empty()) continue;

        cv::Mat bgr;
        cv::cvtColor(color, bgr, cv::COLOR_BGRA2BGR);

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        //маски для знаков, по основному признаку красный и синий
        cv::Mat maskRed1, maskRed2, maskRed, maskBlue;
        cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), maskRed1);
        cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), maskRed2);
        cv::bitwise_or(maskRed1, maskRed2, maskRed);
        cv::inRange(hsv, cv::Scalar(100, 120, 70), cv::Scalar(130, 255, 255), maskBlue);

        cv::morphologyEx(maskRed, maskRed, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2);
        cv::morphologyEx(maskBlue, maskBlue, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contoursRed, contoursBlue;
        cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(maskBlue, contoursBlue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        //знак стопа
        for (auto& contour : contoursRed) {
            if (cv::contourArea(contour) > 500 && isCircle(contour)) {
                cv::Rect box = cv::boundingRect(contour);
                double distance = getAverageDepth(depth, box);

                cv::rectangle(bgr, box, cv::Scalar(0, 0, 255), 2);
                std::ostringstream ss;
                ss.precision(2);
                ss << std::fixed << "STOP (" << distance << " m)";
                cv::putText(bgr, ss.str(), cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
        }

        //знак налево
        for (auto& contour : contoursBlue) {
            if (cv::contourArea(contour) > 500 && isCircle(contour)) {
                cv::Rect box = cv::boundingRect(contour);
                double distance = getAverageDepth(depth, box);

                cv::rectangle(bgr, box, cv::Scalar(255, 0, 0), 2);
                std::ostringstream ss;
                ss.precision(2);
                ss << std::fixed << "LEFT (" << distance << " m)";
                cv::putText(bgr, ss.str(), cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            }
        }

        //визуализация глубины
        cv::Mat depthVis;
        depth.convertTo(depthVis, CV_8U, 255.0 / 4000.0);
        cv::imshow("Depth", depthVis);

        cv::imshow("Kinect RGB", bgr);

        if (cv::waitKey(30) == 27) break;
    }

    sensor->NuiShutdown();
    return 0;
}