import torch
import mss
import cv2
import time
import numpy as np
from PIL import Image
import threading
from multiprocessing import Queue

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Create a queue for passing frames from screen_recorder to the processing thread
frame_queue = Queue()

def screen_recorder():
    # Parameters for capturing a region of the screen
    mon = {"top": 260, "left": 157, "width": 640, "height": 360}
    fps = 0
    sct = mss.mss()

    def capture_screen():
        nonlocal fps
        while True:
            img = np.asarray(sct.grab(mon))
            fps += 1
            frame_queue.put(img)  # Put the frame into the queue

    def display_fps():
        while True:
            print(f"Записано кадров в секунду: {fps}")
            time.sleep(1)

    screen_thread = threading.Thread(target=capture_screen)
    fps_thread = threading.Thread(target=display_fps)

    screen_thread.start()
    fps_thread.start()

    screen_thread.join()
    fps_thread.join()

def object_detection():
    while True:
        frame = frame_queue.get()  # Get a frame from the queue
        im = Image.fromarray(frame)
        results = model(im, size=640)  # Perform object detection on the frame

        for result in results.xyxy[0].numpy():
            label = int(result[5])  # Class label
            confidence = result[4]  # Confidence score
            if confidence > 0.5:  # You can adjust this confidence threshold
                x1, y1, x2, y2 = map(int, result[:4])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a bounding box
                text = f"Class: {label}, Confidence: {confidence:.2f}"
                frame = cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Draw text

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    screen_thread = threading.Thread(target=screen_recorder)
    object_detection_thread = threading.Thread(target=object_detection)

    screen_thread.start()
    object_detection_thread.start()

    screen_thread.join()
    object_detection_thread.join()
