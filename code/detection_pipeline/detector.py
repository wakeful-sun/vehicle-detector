import cv2
import numpy as np
import time


class Detector:

    def __init__(self, classifier, sliding_windows_factory, car_class):
        self.car_class = car_class
        self.classifier = classifier
        self.sliding_windows_factory = sliding_windows_factory
        self.color = (255, 0, 0)
        self.thick = 6

    def get_hot_windows(self, bgr_image):
        sliding_windows = self.sliding_windows_factory.create(bgr_image.shape)

        hot_windows = []
        for window in sliding_windows:
            (x_min, y_min), (x_max, y_max) = window
            test_img = bgr_image[y_min:y_max, x_min:x_max]

            if self.classifier.predict(test_img) == self.car_class:
                hot_windows.append(window)

        return hot_windows

    def detect(self, bgr_image):
        t_start = time.time()

        resulting_image = np.copy(bgr_image)
        hot_windows = self.get_hot_windows(resulting_image)
        for car_window in hot_windows:
            p1, p2 = car_window
            cv2.rectangle(resulting_image, p1, p2, self.color, self.thick)

        print("Detection time: {:.4f} seconds".format(time.time() - t_start))
        return resulting_image
