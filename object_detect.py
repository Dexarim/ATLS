import torch
from mss import mss
import cv2
import time
import numpy
from PIL import Image
import threading
from multiprocessing import Queue

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
frame_queue = Queue()
frame_count = 0

def screen_recorder_and_object_detection(region):
    global frame_count
    fps = 0

    with mss() as sct:
        while True:
            img = numpy.asarray(sct.grab(region))
            frame_count += 1

            if frame_count % 3 == 0:  # Обработка каждого третьего кадра
                im = Image.fromarray(img)
                results = model(im, size=640)

                for result in results.xyxy[0].numpy():
                    label = int(result[5])
                    confidence = result[4]

                    if confidence > 0.3:
                        x1, y1, x2, y2 = map(int, result[:4])
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"Class: {label}, Confidence: {confidence:.2f}"
                        img = cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Object Detection", img)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    regions = [
        {"top": 277, "left": 507, "width": 205, "height": 95},
        {"top": 260, "left": 157, "width": 640, "height": 360},
    ]

    screen_threads = []
    for region in regions:
        screen_thread = threading.Thread(target=screen_recorder_and_object_detection, args=(region,))
        screen_threads.append(screen_thread)
        screen_thread.start()

    for thread in screen_threads:
        thread.join()
