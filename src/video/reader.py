import cv2
import time


class VideoReader:
    def __init__(self, source, width=None, height=None):
        self.cap = cv2.VideoCapture(source)

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.prev_time = time.time()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0

        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

        return frame, fps

    def release(self):
        self.cap.release()
