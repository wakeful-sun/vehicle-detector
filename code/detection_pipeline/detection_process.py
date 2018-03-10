import matplotlib.pyplot as plt
import cv2
import numpy as np


class DetectionProcess:

    def __init__(self, image_handler_fn):
        self.image_handler_fn = image_handler_fn

    def run_on_video(self, video_path):
        video = cv2.VideoCapture(video_path)

        while video.isOpened():
            _, bgr_frame = video.read()

            if not isinstance(bgr_frame, np.ndarray):
                # workaround to handle end of video stream.
                break

            frame = self.image_handler_fn(bgr_frame)
            cv2.imshow("output", frame)

            key = cv2.waitKey(1) & 0xFF
            # stop video on ESC key pressed
            if key == 27:
                break

        video.release()
        cv2.destroyAllWindows()

    def run_on_image(self, image_path):
        def convert(image):
            return image[..., [2, 1, 0]]

        bgr_frame = cv2.imread(image_path)
        frame = self.image_handler_fn(bgr_frame)

        rgb_frame = convert(frame)
        plt.imshow(rgb_frame)
        plt.show()
