import cv2
import numpy as np
import time


class Detector:

    def __init__(self, classifier, search_windows, heat_map, car_class):
        self.classifier = classifier
        self.search_windows = search_windows
        self.heat_map = heat_map
        self.car_class = car_class

        self.color = (255, 0, 0)
        self.thick = 6

    def _select_car_search_positive_windows(self, bgr_image):
        hot_windows = []
        for window in self.search_windows:
            (x_min, y_min), (x_max, y_max) = window
            test_img = bgr_image[y_min:y_max, x_min:x_max]

            if self.classifier.predict(test_img) == self.car_class:
                hot_windows.append(window)

        return hot_windows

    def detect(self, bgr_image):
        t_start = time.time()

        resulting_image = np.copy(bgr_image)
        car_windows = self._select_car_search_positive_windows(resulting_image)

        self.heat_map.update(car_windows)
        cars_boundaries = self.heat_map.get_cars_boundaries()
        self.draw_boxes(resulting_image, cars_boundaries, self.color, self.thick)

        print("Detection time: {:.4f} seconds".format(time.time() - t_start))
        return resulting_image

    @staticmethod
    def draw_boxes(image, boxes, line_color, line_thick):
        for box in boxes:
            p1, p2 = box
            cv2.rectangle(image, p1, p2, line_color, line_thick)
