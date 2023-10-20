import torch
from mss import mss
import cv2
import numpy
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Задаем параметры считывания окна
mon = {"top": 220, "left": 150, "width": 800, "height": 450}
print("Основной кадр создан")

# Задаем координаты и размеры ROI
rois = [
    {"roi_x": 60, "roi_y": 0, "roi_width": 160, "roi_height": 115},
    {"roi_x": 408, "roi_y": 17, "roi_width": 222, "roi_height": 105},
]
print("Rois созданы")

# Создаем объект для считывания экрана
sct = mss()
while True:
    # Считываем кадр
    img = numpy.asarray(sct.grab(mon))
    print("Кадр создан")
    print(img)
    
        
    im = Image.fromarray(img)
    results = model(im, size=640)  # Обнаружение объектов на кадре

    for roi in rois:
        x1, y1, roi_width, roi_height = roi["roi_x"], roi["roi_y"], roi["roi_width"], roi["roi_height"]
        y2 = y1 + roi_height
        x2 = x1 + roi_width

        roi_frame = numpy.zeros((roi_height, roi_width, 4), dtype=numpy.uint8)  # Создать пустой ROI с альфа-каналом
        roi_frame[:, :, 3] = 0  # Установить все значения альфа-канала в 0

        for result in results.xyxy[0].numpy():
            label = int(result[5])  # Class label
            confidence = result[4]
            if confidence > 0.4:  # Порог уверенности
                x, y, w, h = map(int, result[:4])

                # Преобразование координат относительно ROI
                x_on_full_frame = x + x1
                y_on_full_frame = y + y1
                w_on_full_frame = x_on_full_frame + w
                h_on_full_frame = y_on_full_frame + h

                roi_frame = cv2.rectangle(roi_frame, (x_on_full_frame, y_on_full_frame), (w_on_full_frame, h_on_full_frame), (0, 255, 0), 2)  # Нарисовать ограничивающий прямоугольник
                text = f"Class: {label}, Confidence: {confidence:.2f}"
                roi_frame = cv2.putText(roi_frame, text, (x_on_full_frame, y_on_full_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Нарисовать текст

        img[y1:y2, x1:x2] = roi_frame  # Обновить ROI на полном кадре

    cv2.imshow("Object Detection", img)


    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break






