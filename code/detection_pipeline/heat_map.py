import numpy as np
from scipy.ndimage.measurements import label


class HeatMap:

    def __init__(self, height, width, threshold=1):
        self.height = height
        self.width = width
        self.threshold = threshold

        self.heat_step = 1
        self.color_min = 0
        self.color_max = 255
        self.heat_map = np.zeros((height, width)).astype(np.float)
        self.labels = label(self.heat_map)

    @property
    def detected_cars_amount(self):
        return self.labels[1]

    def update(self, car_windows):
        self._add_heat(self.heat_map, car_windows, self.heat_step)
        self._apply_threshold(self.heat_map, self.threshold)
        self.heat_map = np.clip(self.heat_map, self.color_min, self.color_max)

        self.labels = label(self.heat_map)

    def get_cars_boundaries(self):
        labels_map, cars_amount = self.labels
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