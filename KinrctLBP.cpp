#include <Windows.h>
#include <NuiApi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

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
        NUI_IMAGE_TYPE_COLOR,          // Цветной поток
        NUI_IMAGE_RESOLUTION_640x480,  // Разрешение
        0, 2, NULL, &colorStreamHandle)))
    {
        std::cerr << "Failed to open color stream!" << std::endl;
        return false;
    }

    return true;
}

//получение одного кадра и преобразование в cv::Mat
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

//функция для выделения красных областей
cv::Mat detectRedRegions(const cv::Mat& frame)
{
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    //диапазоны для красного цвета (HSV)
    cv::Mat red_mask1, red_mask2;
    cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), red_mask1);
    cv::inRange(hsv, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), red_mask2);

    cv::Mat red_mask;
    cv::bitwise_or(red_mask1, red_mask2, red_mask);

    //морфологические операции для улучшения маски
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel);

    return red_mask;
}

//функция для выделения синих областей
cv::Mat detectBlueRegions(const cv::Mat& frame)
{
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    //диапазон для синего цвета (HSV)
    cv::Mat blue_mask;
    cv::inRange(hsv, cv::Scalar(100, 150, 50), cv::Scalar(140, 255, 255), blue_mask);

    //морфологические операции для улучшения маски
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(blue_mask, blue_mask, cv::MORPH_OPEN, kernel);

    return blue_mask;
}

//функция для создания LBP-признаков
cv::Mat computeLBP(const cv::Mat& gray)
{
    cv::Mat lbp = cv::Mat::zeros(gray.size(), gray.type());

    for (int i = 1; i < gray.rows - 1; i++) {
        for (int j = 1; j < gray.cols - 1; j++) {
            unsigned char center = gray.at<unsigned char>(i, j);
            unsigned char code = 0;

            code |= (gray.at<unsigned char>(i - 1, j - 1) > center) << 7;
            code |= (gray.at<unsigned char>(i - 1, j) > center) << 6;
            code |= (gray.at<unsigned char>(i - 1, j + 1) > center) << 5;
            code |= (gray.at<unsigned char>(i, j + 1) > center) << 4;
            code |= (gray.at<unsigned char>(i + 1, j + 1) > center) << 3;
            code |= (gray.at<unsigned char>(i + 1, j) > center) << 2;
            code |= (gray.at<unsigned char>(i + 1, j - 1) > center) << 1;
            code |= (gray.at<unsigned char>(i, j - 1) > center) << 0;

            lbp.at<unsigned char>(i, j) = code;
        }
    }

    return lbp;
}

//функция для анализа формы и классификации знаков
std::string classifySign(const cv::Mat& region, const cv::Mat& original, const std::string& color)
{
    //находим контуры
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(region.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        //фильтрация по площади
        double area = cv::contourArea(contour);
        if (area < 500) continue; // слишком маленькие области игнорируем для уменьшения ложных

        // аппроксимация контура
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);

        //анализ формы
        if (approx.size() == 8) {
            if (color == "red") {
                return "STOP";
            }
        }
        else if (approx.size() >= 4 && approx.size() <= 6) { // прямоугольник/квадрат - знак направо
            cv::RotatedRect rect = cv::minAreaRect(contour);
            float aspect_ratio = std::max(rect.size.width, rect.size.height) /
                std::min(rect.size.width, rect.size.height);

            if (aspect_ratio < 1.5 && color == "blue") { // квадратная форма
                return "RIGHT";
            }
        }
    }

    return "UNKNOWN";
}

//основная функция детектирования знаков
void detectTrafficSigns(const cv::Mat& frame, cv::Mat& output)
{
    output = frame.clone();

    //детектируем красные и синие области
    cv::Mat red_mask = detectRedRegions(frame);
    cv::Mat blue_mask = detectBlueRegions(frame);

    //конвертируем в градации серого для LBP
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    //вычисляем LBP
    cv::Mat lbp = computeLBP(gray);

    //анализируем красные области
    std::vector<std::vector<cv::Point>> red_contours;
    cv::findContours(red_mask, red_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : red_contours) {
        double area = cv::contourArea(contour);
        if (area > 500) {
            cv::Rect bbox = cv::boundingRect(contour);
            cv::rectangle(output, bbox, cv::Scalar(0, 0, 255), 2);

            std::string sign_type = classifySign(red_mask, frame, "red");
            if (sign_type != "UNKNOWN") {
                cv::putText(output, sign_type, cv::Point(bbox.x, bbox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
        }
    }

    //анализируем синие области
    std::vector<std::vector<cv::Point>> blue_contours;
    cv::findContours(blue_mask, blue_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : blue_contours) {
        double area = cv::contourArea(contour);
        if (area > 500) {
            cv::Rect bbox = cv::boundingRect(contour);
            cv::rectangle(output, bbox, cv::Scalar(255, 0, 0), 2);

            std::string sign_type = classifySign(blue_mask, frame, "blue");
            if (sign_type != "UNKNOWN") {
                cv::putText(output, sign_type, cv::Point(bbox.x, bbox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            }
        }
    }

    //отображаем маски для отладки
    cv::imshow("Red Mask", red_mask);
    cv::imshow("Blue Mask", blue_mask);
    cv::imshow("LBP", lbp);
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
            //конвертируем из BGRA в BGR для OpenCV
            cv::Mat bgr_frame;
            cv::cvtColor(frame, bgr_frame, cv::COLOR_BGRA2BGR);

          
            cv::imshow("Kinect RGB", bgr_frame);

            //конвертация в серые оттенки
            cv::Mat gray;
            cv::cvtColor(bgr_frame, gray, cv::COLOR_BGR2GRAY);
            cv::imshow("Gray Scale", gray);

            //детектирование дорожных знаков
            cv::Mat detected_frame;
            detectTrafficSigns(bgr_frame, detected_frame);
            cv::imshow("Traffic Sign Detection", detected_frame);
        }

        if (cv::waitKey(30) == 27) break; // ESC для выхода
    }

    sensor->NuiShutdown();
    return 0;
}