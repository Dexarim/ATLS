import torch
from mss import mss
import cv2
import numpy
from PIL import Image
import threading
from multiprocessing import Queue



model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
frame_queue = Queue()



def screen_recorder():
    # Задаем параметры считывания окна
    mon = {"top": 260, "left": 157, "width": 640, "height": 360}
    fps = 0
    frame_count = 0

    # Считывание окна
    with mss() as sct:
        while True:
            img = numpy.asarray(sct.grab(mon))
            frame_count += 1
            if frame_count % 5 == 0:  # Обработка кадра
                frame_queue.put(img) 
            fps += 1
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


def object_detection():
    while True:
        frame = frame_queue.get()
        if frame is None:
            continue
        im = Image.fromarray(frame)
        results = model(im, size=640)  #обнаружение объектов на кадре

        for result in results.xyxy[0].numpy():
            label = int(result[5])  # Class label
            confidence = result[4] 
            if confidence > 0.4:  # порог уверенности
                x1, y1, x2, y2 = map(int, result[:4])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Нарисовать ограничивающий прямоугольник
                text = f"Class: {label}, Confidence: {confidence:.2f}"
                frame = cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Нарисовать текст

        cv2.imshow("Object Detection", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    screen_thread = threading.Thread(target=screen_recorder)
    object_detection_thread = threading.Thread(target=object_detection)

    screen_thread.start()
    object_detection_thread.start()

    screen_thread.join()
    object_detection_thread.join()