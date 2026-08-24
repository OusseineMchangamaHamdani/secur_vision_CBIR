import cv2
import os
import time

os.makedirs("captures", exist_ok=True)


class CaptureManager:

    def __init__(self):

        self.start_time = None

    def start(self):

        if self.start_time is None:
            self.start_time = time.time()

    def reset(self):

        self.start_time = None

    def elapsed(self):

        if self.start_time is None:
            return 0

        return time.time() - self.start_time

    def capture(self, frame):

        filename = f"captures/{int(time.time())}.jpg"

        cv2.imwrite(filename, frame)

        return filename