import numpy as np


class SearchWindowsFactory:

    @staticmethod
    def _get_min_max(min_max_array, default_max):
        return [0 if min_max_array[0] is None else min_max_array[0],
                default_max if min_max_array[1] is None else min_max_array[1]]

    @staticmethod
    def _get_search_region_size(min_max_array):
        return min_max_array[1] - min_max_array[0]

    @staticmethod
    def _get_step_size(window_size, overlap_percentage):
        assert 0 <= overlap_percentage < 1, "overlap should be in [0-1) boundaries"
        return np.int(window_size * (1 - overlap_percentage))

    @staticmethod
    def _get_windows_amount(area_size, window_size, step_size):
        assert area_size >= window_size, "can't split image on windows, area is too small"
        windows_amount = 1 + np.int((area_size - window_size) / step_size)
        return windows_amount

    @staticmethod
    def _get_window_min_max(start, step_size, window_size, window_number):
        min = start + step_size * window_number
        max = min + window_size
        return min, max

    def create(self, max_height, max_width, configs):
        windows = []
        [windows.extend(self.create_windows_for_config(max_height, max_width, config)) for config in configs]
        return windows

    def create_windows_for_config(self, max_height, max_width, config):
        window_size_x, window_size_y = config["xy_window"]
        overlap_x, overlap_y = config["xy_overlap"]

        x_boundaries = self._get_min_max(config["x_start_stop"], max_width)
        y_boundaries = self._get_min_max(config["y_start_stop"], max_height)

        search_region_size_x = self._get_search_region_size(x_boundaries)
        search_region_size_y = self._get_search_region_size(y_boundaries)

        x_step_size = self._get_step_size(window_size_x, overlap_x)
        y_step_size = self._get_step_size(window_size_y, overlap_y)

        nx_windows = self._get_windows_amount(search_region_size_x, window_size_x, x_step_size)
        ny_windows = self._get_windows_amount(search_region_size_y, window_size_y, y_step_size)

        x_start = x_boundaries[0]
        y_start = y_boundaries[0]

        window_list = []
        for y_window_number in range(ny_windows):
            y_min, y_max = self._get_window_min_max(y_start, y_step_size, window_size_y, y_window_number)
            for x_window_number in range(nx_windows):
                x_min, x_max = self._get_window_min_max(x_start, x_step_size, window_size_x, x_window_number)
                window_list.append(((x_min, y_min), (x_max, y_max)))

        print("Window size: {}, Total amount of search windows: {}.".format((window_size_x, window_size_y),
                                                                            len(window_list)))
        return window_list
