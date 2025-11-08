#include <Windows.h>
#include <NuiApi.h>
#include <opencv2/opencv.hpp>
#include <iostream>

INuiSensor* sensor = nullptr;
HANDLE colorStreamHandle = nullptr;

//инициализация Kinect
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

//получение кадра и преобразование в cv::Mat
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
        //BGRA → OpenCV матрица
        cv::Mat temp(cv::Size(640, 480), CV_8UC4, lockedRect.pBits);
        frame = temp.clone(); //делаем копию, т.к. буфер освободится после UnlockRect
    }

    pTexture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(colorStreamHandle, &imageFrame);

    return frame;
}

int main()
{
    if (!initKinect()) {
        return -1;
    }

    std::cout << "Kinect initialized. Press ESC to exit." << std::endl;

    while (true) {
        cv::Mat frame = getColorFrame();
        if (!frame.empty()) {
            //оригинальное RGB изображение
            cv::imshow("Kinect RGB", frame);

            //Конвертация в серые оттенки
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
            cv::imshow("Gray", gray);


        }

        if (cv::waitKey(30) == 27) break; // ESC для выхода
    }

    sensor->NuiShutdown();
    return 0;
}