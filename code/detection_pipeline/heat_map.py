import numpy as np
from scipy.ndimage.measurements import label
from collections import deque


class HeatMap:

    def __init__(self, height, width, threshold=2):
        self.height = height
        self.width = width
        self.threshold = threshold

        self.heat_step = 1
        self.color_min = 0
        self.color_max = 255
        self.detections_cache = deque([], maxlen=10)

    def update(self, car_windows):
        heat_map = np.zeros((self.height, self.width)).astype(np.float)
        self._add_heat(heat_map, car_windows, self.heat_step)
        self.detections_cache.appendleft(heat_map)

    def get_cars_boundaries(self):
        heat_map = np.average(self.detections_cache, axis=0)
        self._apply_threshold(heat_map, self.threshold)
        heat_map = np.clip(heat_map, self.color_min, self.color_max)

        labels_map, cars_amount = label(heat_map)
        cars_boundaries = []

        for car_number in range(1, cars_amount + 1):
            y_indexes, x_indexes = (labels_map == car_number).nonzero()
            car_x_indexes = np.array(x_indexes)
            car_y_indexes = np.array(y_indexes)

            p1 = np.min(car_x_indexes), np.min(car_y_indexes)
            p2 = np.max(car_x_indexes), np.max(car_y_indexes)
            cars_boundaries.append((p1, p2))

        return cars_boundaries

    @staticmethod
    def _add_heat(heat_map, car_windows, heat_step):
        for window in car_windows:
            (x_min, y_min), (x_max, y_max) = window
            heat_map[y_min:y_max, x_min:x_max] += heat_step
        return heat_map

    @staticmethod
    def _apply_threshold(heat_map, threshold):
        heat_map[heat_map <= threshold] = 0
        return heat_map