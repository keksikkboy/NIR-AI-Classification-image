//подключение библиотек
#include <Windows.h>
#include <NuiApi.h>
#include <opencv2/opencv.hpp>
#include <iostream>

INuiSensor* sensor = nullptr;
HANDLE colorStreamHandle = nullptr;
HANDLE depthStreamHandle = nullptr;

//инициализируем кинект, если не подрубился выводит сообщения об ошибке или о том что не найден
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

    //подрубаем поток цвета
    if (FAILED(sensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION_640x480,
        0, 2, NULL, &colorStreamHandle)))
    {
        std::cerr << "Failed to open color stream!" << std::endl;
        return false;
    }

    //подрубаем датчик глубины
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

//получаем цветовые кадры
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

//получаем кадры глубины
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

        //раскадровка глубины 
        for (int y = 0; y < depthMat.rows; ++y) {
            USHORT* rowPtr = depthMat.ptr<USHORT>(y);
            for (int x = 0; x < depthMat.cols; ++x) {
                USHORT raw = pBuffer[y * depthMat.cols + x];
                USHORT depthMM = raw >> 3; 
                rowPtr[x] = depthMM;
            }
        }
    }

    pTexture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(depthStreamHandle, &depthFrame);

    return depthMat;
}

int main()
{
    if (!initKinect()) return -1;
    std::cout << "Kinect initialized. Press ESC to exit." << std::endl;

    while (true) {
        cv::Mat color = getColorFrame();
        cv::Mat depth = getDepthFrame();
        if (color.empty() || depth.empty()) continue;

        //визуализация глубины
        cv::Mat depthVis;
        depth.convertTo(depthVis, CV_8U, 255.0 / 4000.0);
        cv::imshow("Depth", depthVis);

        cv::Mat bgr;
        cv::cvtColor(color, bgr, cv::COLOR_BGRA2BGR);
        cv::imshow("Kinect RGB", bgr);

        if (cv::waitKey(30) == 27) break;
    }

    sensor->NuiShutdown();
    return 0;
}