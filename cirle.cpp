#include <Windows.h>
#include <NuiApi.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

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
        // BGRA → OpenCV матрица
        cv::Mat temp(cv::Size(640, 480), CV_8UC4, lockedRect.pBits);
        frame = temp.clone(); // делаем копию, т.к. буфер освободится после UnlockRect
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
       
            cv::imshow("Kinect RGB", frame);

         
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
            cv::imshow("Gray Scale", gray);



     
            cv::Mat bgr, hsv;
            cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
            cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

     
            cv::Mat mask1, mask2, mask;
            cv::inRange(hsv, cv::Scalar(0, 80, 80),     cv::Scalar(10, 255, 255), mask1);
            cv::inRange(hsv, cv::Scalar(160, 80, 80),   cv::Scalar(179, 255, 255), mask2);
            cv::bitwise_or(mask1, mask2, mask);

            
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(mask.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

     
            cv::Mat drawFrame = bgr.clone();

            // анализ контуров по критерию круглости
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = cv::contourArea(contours[i]);
                
                // отсечение мелкого шума 
                if (area < 500.0)
                    continue;

                
                double perimeter = cv::arcLength(contours[i], true);
                if (perimeter <= 0.0)
                    continue;

               
                double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);

           
                if (circularity > 0.7) {
                    // минимальная окружность, описывающая контур
                    cv::Point2f center;
                    float radius;
                    cv::minEnclosingCircle(contours[i], center, radius);

                    //рисуем зелёный контур
                    cv::circle(drawFrame, center, (int)radius, cv::Scalar(0, 255, 0), 3);
                    
                   
                }
            }

            cv::imshow("Detected circular signs", drawFrame);
            cv::imshow("Red mask", mask);
        }

        if (cv::waitKey(30) == 27) break; // экспейп для выхода
    }

    sensor->NuiShutdown();
    return 0;
}