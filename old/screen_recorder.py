import cv2
import mss
import time
import numpy
import threading

def screen_recorder():
    # задаем параметры считывания окна
    # top отступ по выстое
    # left отступ по левой части экрана
    # width размеры считывания окна по длине
    # height размеры считывания окна по высоте
    mon = {"top": 260, "left": 157, "width": 640, "height": 360}
    title = "[test screen]"
    fps = 0
    sct = mss.mss()
    
    # считывания окна
    def capture_screen():
        nonlocal fps
        while True:
            img = numpy.asarray(sct.grab(mon))
            fps += 1
            cv2.imshow(title, img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    # вывод окна
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

if __name__ == "__main__":
    screen_recorder()
