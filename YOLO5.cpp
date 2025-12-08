#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

// Структура для хранения результатов детекции
struct Detection {
    int classId;
    float confidence;
    Rect boundingBox;
    string className;
};

class TrafficSignDetector {
private:
    Net net;
    vector<string> classNames;
    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    
    // Цвета для рисования разных классов знаков
    vector<Scalar> colors = {
        Scalar(0, 255, 0),     // зелёный
        Scalar(255, 0, 0),     // синий
        Scalar(0, 0, 255),     // красный
        Scalar(255, 255, 0),   // жёлтый
        Scalar(255, 0, 255)    // фиолетовый
    };

public:
    TrafficSignDetector() {}
    
    // Загрузка классов из файла
    bool loadClassNames(const string& classNamesPath) {
        ifstream ifs(classNamesPath);
        if (!ifs.is_open()) {
            cerr << "Ошибка: не удаётся открыть файл с классами: " 
                 << classNamesPath << endl;
            return false;
        }
        
        string line;
        while (getline(ifs, line)) {
            if (!line.empty()) {
                classNames.push_back(line);
            }
        }
        
        cout << "Загружено " << classNames.size() << " классов знаков" << endl;
        return true;
    }
    
    // Загрузка предобученной модели YOLOv5
    bool loadModel(const string& modelPath) {
        try {
            net = readNetFromONNX(modelPath);
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            // Для использования GPU раскомментируйте следующие строки:
            // net.setPreferableBackend(DNN_BACKEND_CUDA);
            // net.setPreferableTarget(DNN_TARGET_CUDA);
            
            cout << "Модель успешно загружена: " << modelPath << endl;
            return true;
        } catch (const exception& e) {
            cerr << "Ошибка при загрузке модели: " << e.what() << endl;
            return false;
        }
    }
    
    // Предварительная обработка изображения
    Mat preprocessImage(const Mat& image, int inputSize = 640) {
        Mat blob;
        
        // Преобразование в blob с нормализацией
        blobFromImage(image, blob, 1.0/255.0, Size(inputSize, inputSize),
                      Scalar(0, 0, 0), true, false);
        
        return blob;
    }
    
    // Постобработка результатов сети
    vector<Detection> postprocess(const Mat& image, 
                                  const vector<Mat>& outputs,
                                  int inputSize = 640) {
        vector<Detection> detections;
        
        int imageHeight = image.rows;
        int imageWidth = image.cols;
        
        float scaleX = (float)imageWidth / inputSize;
        float scaleY = (float)imageHeight / inputSize;
        
        // Обработка выходов сети
        for (const auto& output : outputs) {
            float* data = (float*)output.data;
            
            for (int i = 0; i < output.rows; i++) {
                // Получение координат центра, размеров и уверенности
                float x = data[i * output.cols + 0] * scaleX;
                float y = data[i * output.cols + 1] * scaleY;
                float w = data[i * output.cols + 2] * scaleX;
                float h = data[i * output.cols + 3] * scaleY;
                
                // Преобразование в координаты углов
                int left = max(0, (int)(x - w / 2));
                int top = max(0, (int)(y - h / 2));
                int right = min(imageWidth, (int)(x + w / 2));
                int bottom = min(imageHeight, (int)(y + h / 2));
                
                // Получение максимальной уверенности класса
                float maxConf = 0;
                int classId = -1;
                
                for (int j = 4; j < output.cols; j++) {
                    float confidence = data[i * output.cols + j];
                    if (confidence > maxConf) {
                        maxConf = confidence;
                        classId = j - 4;
                    }
                }
                
                // Фильтрация по порогу уверенности
                if (maxConf >= confThreshold && classId >= 0 
                    && classId < classNames.size()) {
                    Detection det;
                    det.classId = classId;
                    det.confidence = maxConf;
                    det.boundingBox = Rect(left, top, right - left, bottom - top);
                    det.className = classNames[classId];
                    detections.push_back(det);
                }
            }
        }
        
        // Применение NMS (Non-Maximum Suppression)
        vector<Rect> boxes;
        vector<float> confidences;
        for (const auto& det : detections) {
            boxes.push_back(det.boundingBox);
            confidences.push_back(det.confidence);
        }
        
        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        
        vector<Detection> finalDetections;
        for (int idx : indices) {
            finalDetections.push_back(detections[idx]);
        }
        
        return finalDetections;
    }
    
    // Основной метод детекции на изображении
    vector<Detection> detect(const Mat& image) {
        if (net.empty()) {
            cerr << "Модель не загружена!" << endl;
            return vector<Detection>();
        }
        
        // Предварительная обработка
        Mat blob = preprocessImage(image);
        
        // Задание входных данных
        net.setInput(blob);
        
        // Получение имён выходных слоёв
        vector<String> outNames = net.getUnconnectedOutLayersNames();
        
        // Прямой проход через сеть
        vector<Mat> outputs;
        net.forward(outputs, outNames);
        
        // Постобработка результатов
        return postprocess(image, outputs);
    }
    
    // Рисование результатов на изображении
    void drawDetections(Mat& image, const vector<Detection>& detections) {
        for (const auto& det : detections) {
            // Выбор цвета для класса
            Scalar color = colors[det.classId % colors.size()];
            
            // Рисование прямоугольника
            rectangle(image, det.boundingBox, color, 2);
            
            // Подготовка текста с названием класса и уверенностью
            string label = det.className + ": " 
                          + to_string((int)(det.confidence * 100)) + "%";
            
            // Размер текста
            int baseLine = 0;
            Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 
                                       0.6, 1, &baseLine);
            
            // Рисование фона для текста
            rectangle(image, 
                     Point(det.boundingBox.x, det.boundingBox.y - textSize.height - 5),
                     Point(det.boundingBox.x + textSize.width, det.boundingBox.y),
                     color, -1);
            
            // Рисование текста
            putText(image, label, 
                   Point(det.boundingBox.x, det.boundingBox.y - 5),
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        }
    }
};

// ============ ГЛАВНАЯ ФУНКЦИЯ ============
int main(int argc, char* argv[]) {
    cout << "=== Детектор дорожных знаков (YOLOv5 + OpenCV) ===" << endl;
    
    // Пути к файлам моделей (замените на ваши пути)
    string modelPath = "yolov5s-traffic.onnx";      // Скачайте предобученную модель
    string classNamesPath = "traffic_classes.txt";  // Файл с названиями классов
    string videoInput = "0";                        // Камера (0) или путь к видео
    
    // Создание детектора
    TrafficSignDetector detector;
    
    // Загрузка модели и классов
    if (!detector.loadClassNames(classNamesPath)) {
        return -1;
    }
    
    if (!detector.loadModel(modelPath)) {
        return -1;
    }
    
    // Открытие видеопотока с камеры или файла
    VideoCapture cap;
    if (videoInput == "0") {
        cap.open(0);  // Встроенная камера
    } else {
        cap.open(videoInput);  // Видеофайл
    }
    
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удаётся открыть видеопоток!" << endl;
        return -1;
    }
    
    cout << "Видеопоток открыт успешно. Нажмите 'q' для выхода." << endl;
    
    // Основной цикл обработки кадров
    Mat frame, resized;
    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();
    
    while (true) {
        // Захват кадра
        if (!cap.read(frame)) {
            cerr << "Ошибка при чтении кадра" << endl;
            break;
        }
        
        frameCount++;
        
        // Изменение размера для более быстрой обработки (опционально)
        resize(frame, resized, Size(640, 480));
        
        // Детекция знаков
        vector<Detection> detections = detector.detect(resized);
        
        // Рисование результатов
        detector.drawDetections(resized, detections);
        
        // Отображение количества найденных знаков
        string statusText = "Знаков найдено: " + to_string(detections.size());
        putText(resized, statusText, Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        
        // Расчёт FPS
        if (frameCount % 30 == 0) {
            auto currentTime = chrono::high_resolution_clock::now();
            double fps = 30.0 / chrono::duration<double>(
                currentTime - startTime).count();
            startTime = currentTime;
            cout << "FPS: " << fixed << setprecision(1) << fps << endl;
        }
        
        // Отображение FPS на кадре
        putText(resized, "FPS: " + to_string((int)30), Point(10, 60),
               FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 1);
        
        // Показ кадра
        imshow("Детектор дорожных знаков", resized);
        
        // Нажмите 'q' для выхода
        if (waitKey(1) == 'q') {
            break;
        }
    }
    
    // Освобождение ресурсов
    cap.release();
    destroyAllWindows();
    
    cout << "Программа завершена." << endl;
    return 0;
}